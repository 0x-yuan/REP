"""Multi-replica orchestrator for inference-farm.

Use case
--------
One Modal account hosts N independent replicas of the SAME model under
unique app names (`sglang-slave-<model>-r0`, `-r1`, ...). All replicas
share the Modal Volumes (`inference-farm-data`, `sglang-hf-cache`,
`sglang-slave-checkpoints`) so checkpoints and inbox files are visible
to every replica, but each replica has its OWN progress / fc Dicts so
the master can track them independently.

The orchestrator pulls files from a thread-safe queue and dispatches one
file at a time to whichever replica is free. This gives natural load
balancing without any pre-sharding work — replicas that finish early
just grab the next file.

A background **inbox watcher** thread periodically rescans `inbox/` so
files dropped into the inbox AFTER startup are picked up automatically.
You can therefore stage new work mid-run by `cp`-ing a JSONL into the
inbox or `rm`-ing a queued file you no longer want. See
`master/queue_cli.py` for an ergonomic CLI on top of this directory.

For workloads with one giant inbox file you want to fan out, use the
`--shard-files-min-rows` flag: any inbox file >= that many rows is split
into N parts (one per replica) BEFORE being added to the queue.

Two run modes:
  * **drain (default)** — exit after the inbox is empty AND all workers
    are idle for `--drain-quiet-s` consecutive seconds. This is the
    correct mode for "process my current backlog and stop".
  * **watch (`--watch`)** — never auto-exit; sleep waiting for new files.
    Stop the runner with SIGINT.

Replica deployment
------------------
Each replica is a separate Modal app, deployed with REPLICA_ID set:

    MODEL_KEY=qwen3-8b REPLICA_ID=r0 \\
        MODAL_PROFILE=<your-modal-profile> \\
        uv run modal deploy inference-farm/slave/app.py
    MODEL_KEY=qwen3-8b REPLICA_ID=r1 \\
        MODAL_PROFILE=<your-modal-profile> \\
        uv run modal deploy inference-farm/slave/app.py
    # ...

`master/run_pipeline.sh REPLICAS=N` automates this.

Usage
-----
    MODAL_PROFILE=<your-modal-profile> uv run python -u \\
        inference-farm/master/multi_replica_runner.py \\
        --model qwen3-8b --replicas 4

The orchestrator picks up apps named
`sglang-slave-<model>-r0`...`sglang-slave-<model>-r{N-1}`.
"""
from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import shutil
import sys
import threading
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_SERVER_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SERVER_ROOT / "slave"))

from naming import app_name as _slave_app_name  # noqa: E402
from queue_runner import (  # noqa: E402
    DEFAULT_STALL_TIMEOUT_S,
    INBOX,
    PROCESSED,
    RESULT,
    _count_lines,
    _list_inbox_files,
    process_one,
)


def _replica_app_name(model: str, replica_id: str) -> str:
    return _slave_app_name(model, replica_id)


def _shard_jsonl(src: Path, n_parts: int, dst_dir: Path) -> list[Path]:
    """Split a JSONL file into N roughly-equal-line shards.

    Output names: `<stem>__shard{i}of{n}.jsonl` so they sort stably and
    cannot collide with the original file. Returns the new shard paths.
    The original file is moved to PROCESSED to prevent re-queue.
    """
    n_total = _count_lines(src)
    if n_total == 0:
        return []
    if n_parts <= 1:
        return [src]

    base = src.stem
    # Extract the part after the model__ prefix so we keep the model__ prefix
    # on the shard files (the inbox lister filters by `model__*.jsonl`).
    if "__" in base:
        model_prefix, batch_part = base.split("__", 1)
    else:
        model_prefix, batch_part = "", base
    if not model_prefix:
        raise ValueError(
            f"shard source must have a model__ prefix in its name; got {src.name!r}"
        )

    # Even split with the last shard absorbing the remainder.
    rows_per_shard = n_total // n_parts
    extra = n_total - rows_per_shard * n_parts

    shard_paths: list[Path] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    with src.open() as f_in:
        for shard_idx in range(n_parts):
            n_take = rows_per_shard + (1 if shard_idx < extra else 0)
            if n_take == 0:
                continue
            shard_name = (
                f"{model_prefix}__{batch_part}__shard{shard_idx}of{n_parts}.jsonl"
            )
            shard_path = dst_dir / shard_name
            with shard_path.open("w") as f_out:
                for _ in range(n_take):
                    line = f_in.readline()
                    if not line:
                        break
                    f_out.write(line)
            shard_paths.append(shard_path)

    # Move the original to PROCESSED with an .original.jsonl suffix so
    # nothing else picks it up. Use shutil.move which is atomic on the same
    # filesystem.
    PROCESSED.mkdir(parents=True, exist_ok=True)
    src_archived = PROCESSED / f"{src.stem}.preshard.jsonl"
    shutil.move(str(src), str(src_archived))
    return shard_paths


def _maybe_shard_inputs(
    files: list[Path],
    n_replicas: int,
    min_rows: int,
    tag: str,
) -> list[Path]:
    """For any file with >= min_rows lines, split it into n_replicas shards
    (in-place in INBOX). Returns the post-shard list."""
    if n_replicas <= 1 or min_rows <= 0:
        return files
    expanded: list[Path] = []
    for f in files:
        try:
            n = _count_lines(f)
        except Exception:
            expanded.append(f)
            continue
        if n >= min_rows:
            print(
                f"{tag} pre-sharding {f.name} ({n} rows) into {n_replicas} parts",
                flush=True,
            )
            shards = _shard_jsonl(f, n_replicas, INBOX)
            for s in shards:
                print(f"{tag}   → {s.name} ({_count_lines(s)} rows)", flush=True)
            expanded.extend(shards)
        else:
            expanded.append(f)
    return expanded


def _worker_loop(
    *,
    replica_id: str,
    model: str,
    file_q: queue_mod.Queue,
    poll_interval: float,
    max_retries: int,
    stall_timeout_s: int,
    failures: list[tuple[str, str]],
    stop_event: threading.Event | None = None,
    in_flight: set[str] | None = None,
    in_flight_lock: threading.Lock | None = None,
    busy: dict[str, str] | None = None,
    status_print_interval_s: float | None = None,
) -> None:
    """One worker thread = one replica.

    Pulls files from the shared queue, sends them to its own slave app,
    and keeps going until `stop_event` is set AND the queue is drained.
    `in_flight` / `busy` / `_lock` are optional shared state for the
    inbox watcher so it knows which files are already enqueued or being
    processed.
    """
    app_name = _replica_app_name(model, replica_id)
    tag = f"[{replica_id}/{model}]"
    print(f"{tag} worker start; app={app_name}", flush=True)

    Slave = modal.Cls.from_name(app_name, "SGLangSlave")
    slave = Slave()
    try:
        info = slave.ping.remote()
        print(
            f"{tag} ping ← phase={info.get('_state', {}).get('phase', '?')}",
            flush=True,
        )
    except Exception as e:
        print(f"{tag} ping failed: {e}", flush=True)
        failures.append((replica_id, f"ping_failed: {e!r}"))
        return

    while True:
        # Drain-then-stop: only exit when stop is signalled AND nothing
        # is left in the queue. This lets the watcher request a graceful
        # shutdown without dropping work in progress.
        if stop_event is not None and stop_event.is_set() and file_q.empty():
            print(f"{tag} queue empty + stop signalled; worker exiting", flush=True)
            return

        try:
            # `timeout` makes the worker wake periodically to re-check
            # stop_event so a watch-mode shutdown converges in bounded time.
            jsonl_path = file_q.get(timeout=max(1.0, poll_interval))
        except queue_mod.Empty:
            if stop_event is None:
                # Legacy semantics — exit immediately on empty queue.
                print(f"{tag} queue empty; worker exiting", flush=True)
                return
            continue

        # If the file was deleted from inbox while sitting in the queue,
        # _process_one will skip it cleanly. We still need to release the
        # in_flight slot below.
        try:
            if busy is not None:
                busy[replica_id] = jsonl_path.name
            process_one(
                slave,
                jsonl_path,
                tag,
                poll_interval,
                max_retries,
                stall_timeout_s=stall_timeout_s,
                status_print_interval_s=status_print_interval_s,
            )
        except SystemExit as e:
            # process_one raises SystemExit on max_retries exhausted.
            print(f"{tag} {jsonl_path.name} FAILED: {e}", flush=True)
            failures.append((replica_id, f"{jsonl_path.name}: {e}"))
        except Exception as e:
            print(f"{tag} {jsonl_path.name} EXCEPTION: {e!r}", flush=True)
            failures.append((replica_id, f"{jsonl_path.name}: {e!r}"))
        finally:
            if in_flight is not None and in_flight_lock is not None:
                with in_flight_lock:
                    in_flight.discard(jsonl_path.name)
            if busy is not None:
                busy[replica_id] = ""
            file_q.task_done()


def _watcher_loop(
    *,
    model: str,
    file_q: queue_mod.Queue,
    in_flight: set[str],
    in_flight_lock: threading.Lock,
    stop_event: threading.Event,
    scan_interval: float,
    tag: str,
    pre_shard_min_rows: int,
    n_replicas: int,
) -> None:
    """Periodically rescan INBOX for new `<model>__*.jsonl` files and put
    them on the queue. Files already enqueued or in-flight are skipped.

    On startup, this thread also performs the optional pre-sharding pass —
    so a giant inbox file gets split BEFORE its shards are queued.
    """
    print(
        f"{tag} watcher start; scan_interval={scan_interval}s "
        f"shard_min_rows={pre_shard_min_rows}",
        flush=True,
    )
    first_pass = True
    while not stop_event.is_set():
        try:
            files = _list_inbox_files(model)
        except Exception as e:
            print(f"{tag} watcher inbox-scan error: {e!r}", flush=True)
            files = []

        # Pre-shard pass — only on first scan (subsequent additions assume
        # the user already sized them appropriately).
        if first_pass and pre_shard_min_rows > 0 and n_replicas > 1:
            files = _maybe_shard_inputs(
                files, n_replicas, pre_shard_min_rows, tag
            )
        first_pass = False

        added = 0
        with in_flight_lock:
            for f in files:
                if not f.exists():
                    continue
                if f.name in in_flight:
                    continue
                in_flight.add(f.name)
                file_q.put(f)
                added += 1
                print(f"{tag} watcher enqueued: {f.name}", flush=True)
        if added == 0 and stop_event.is_set():
            break
        # `wait` returns True when the event is set; that lets us stop
        # promptly without sleeping out the full interval.
        stop_event.wait(timeout=scan_interval)
    print(f"{tag} watcher exiting", flush=True)


def _summary_loop(
    *,
    file_q: queue_mod.Queue,
    in_flight: set[str],
    in_flight_lock: threading.Lock,
    busy: dict[str, str],
    stop_event: threading.Event,
    interval_s: float,
    tag: str,
) -> None:
    """Print a periodic global summary: queued, in-flight per replica,
    total seen. Strictly informational — does not change scheduling."""
    while not stop_event.is_set():
        stop_event.wait(timeout=interval_s)
        if stop_event.is_set():
            break
        with in_flight_lock:
            n_in_flight = len(in_flight)
            snapshot = sorted(in_flight)
        n_queued = file_q.qsize()
        busy_lines = ", ".join(
            f"{rid}={(name or '<idle>')}" for rid, name in busy.items()
        )
        print(
            f"{tag} summary: queued={n_queued} tracked={n_in_flight} "
            f"busy=[{busy_lines}] tracked_files={snapshot[:5]}"
            f"{'...' if len(snapshot) > 5 else ''}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--replicas",
        type=int,
        required=True,
        help="N — orchestrator expects deployed apps "
        "sglang-slave-<model>-r0 ... -r{N-1}",
    )
    parser.add_argument("--poll-interval", default=10.0, type=float)
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument(
        "--stall-timeout-s",
        type=int,
        default=DEFAULT_STALL_TIMEOUT_S,
        help="seconds without progress before queue_runner cancels & respawns "
        "(env SGLANG_STALL_TIMEOUT_S also accepted)",
    )
    parser.add_argument(
        "--shard-files-min-rows",
        type=int,
        default=0,
        help="if > 0, any inbox file with >= this many rows is pre-sharded "
        "into <replicas> parts before queueing (helps when one big file "
        "would otherwise pin to one replica)",
    )
    parser.add_argument(
        "--replica-ids",
        default=None,
        help="comma-separated explicit replica IDs (default: r0,r1,...,r{N-1})",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stay alive after the inbox drains — keep polling for newly "
        "added files. Without this flag (drain mode), exit when the inbox "
        "has been empty AND all workers have been idle for "
        "--drain-quiet-s seconds.",
    )
    parser.add_argument(
        "--inbox-scan-interval-s",
        type=float,
        default=5.0,
        help="how often the watcher rescans inbox/ for new files",
    )
    parser.add_argument(
        "--summary-interval-s",
        type=float,
        default=60.0,
        help="how often to print a global queued/in-flight summary "
        "(0 disables)",
    )
    parser.add_argument(
        "--drain-quiet-s",
        type=float,
        default=15.0,
        help="in drain mode, exit after the inbox has been empty AND all "
        "workers have been idle for this many consecutive seconds. Lower = "
        "exits faster after the last file finishes; raise it on slow shared "
        "filesystems where new file appearance lags.",
    )
    parser.add_argument(
        "--status-print-interval-s",
        type=float,
        default=None,
        help="forwarded to queue_runner._process_one — how often to print a "
        "progress line per replica even when n_done has not changed. Default "
        "uses SGLANG_STATUS_PRINT_INTERVAL_S env or 30s.",
    )
    args = parser.parse_args()

    if args.replicas < 1:
        raise SystemExit("--replicas must be >= 1")

    if args.replica_ids:
        replica_ids = [s.strip() for s in args.replica_ids.split(",") if s.strip()]
        if len(replica_ids) != args.replicas:
            raise SystemExit(
                f"--replica-ids has {len(replica_ids)} entries but "
                f"--replicas={args.replicas}"
            )
    else:
        replica_ids = [f"r{i}" for i in range(args.replicas)]

    tag = f"[mr/{args.model}]"
    print(
        f"{tag} start; inbox={INBOX} result={RESULT} "
        f"replicas={replica_ids} stall_timeout_s={args.stall_timeout_s} "
        f"mode={'watch' if args.watch else 'drain'}",
        flush=True,
    )

    file_q: queue_mod.Queue = queue_mod.Queue()
    in_flight: set[str] = set()
    in_flight_lock = threading.Lock()
    busy: dict[str, str] = {rid: "" for rid in replica_ids}
    stop_event = threading.Event()
    failures: list[tuple[str, str]] = []

    # SIGINT / SIGTERM → set stop_event, drain gracefully.
    import signal

    def _sig(_n, _f):
        if not stop_event.is_set():
            print(f"{tag} signal received; requesting graceful shutdown", flush=True)
            stop_event.set()

    try:
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except (ValueError, OSError):
        # Not on the main thread (e.g. running under tests) — skip.
        pass

    watcher = threading.Thread(
        target=_watcher_loop,
        kwargs=dict(
            model=args.model,
            file_q=file_q,
            in_flight=in_flight,
            in_flight_lock=in_flight_lock,
            stop_event=stop_event,
            scan_interval=max(1.0, args.inbox_scan_interval_s),
            tag=tag,
            pre_shard_min_rows=args.shard_files_min_rows,
            n_replicas=len(replica_ids),
        ),
        name="inbox-watcher",
        daemon=True,
    )
    watcher.start()

    summary_thread: threading.Thread | None = None
    if args.summary_interval_s > 0:
        summary_thread = threading.Thread(
            target=_summary_loop,
            kwargs=dict(
                file_q=file_q,
                in_flight=in_flight,
                in_flight_lock=in_flight_lock,
                busy=busy,
                stop_event=stop_event,
                interval_s=args.summary_interval_s,
                tag=tag,
            ),
            name="summary",
            daemon=True,
        )
        summary_thread.start()

    workers: list[threading.Thread] = []
    for rid in replica_ids:
        t = threading.Thread(
            target=_worker_loop,
            kwargs=dict(
                replica_id=rid,
                model=args.model,
                file_q=file_q,
                poll_interval=args.poll_interval,
                max_retries=args.max_retries,
                stall_timeout_s=args.stall_timeout_s,
                failures=failures,
                stop_event=stop_event,
                in_flight=in_flight,
                in_flight_lock=in_flight_lock,
                busy=busy,
                status_print_interval_s=args.status_print_interval_s,
            ),
            name=f"worker-{rid}",
            daemon=False,
        )
        t.start()
        workers.append(t)

    # Drive the drain decision on the main thread.
    if args.watch:
        print(
            f"{tag} watch mode: orchestrator will not auto-exit. "
            f"Use SIGINT/SIGTERM to stop; in-flight files are completed first.",
            flush=True,
        )
        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=5.0)
        except KeyboardInterrupt:
            stop_event.set()
    else:
        # Drain mode: wait until inbox empty AND queue empty AND no in-flight
        # for `drain_quiet_s` consecutive seconds.
        print(
            f"{tag} drain mode: exit after {args.drain_quiet_s}s of "
            "(inbox empty AND queue empty AND no in-flight)",
            flush=True,
        )
        quiet_start: float | None = None
        try:
            while not stop_event.is_set():
                # Snapshot state.
                with in_flight_lock:
                    tracked = len(in_flight)
                    n_busy = sum(1 for v in busy.values() if v)
                    inbox_files = _list_inbox_files(args.model)
                    inbox_count = sum(1 for p in inbox_files if p.exists())
                quiet = (tracked == 0 and n_busy == 0 and inbox_count == 0)
                now = time.time()
                if quiet:
                    if quiet_start is None:
                        quiet_start = now
                    elif now - quiet_start >= args.drain_quiet_s:
                        print(
                            f"{tag} drain quiet for {int(now - quiet_start)}s "
                            "→ requesting shutdown",
                            flush=True,
                        )
                        stop_event.set()
                        break
                else:
                    quiet_start = None
                stop_event.wait(timeout=2.0)
        except KeyboardInterrupt:
            stop_event.set()

    for t in workers:
        t.join()

    if failures:
        print(f"{tag} {len(failures)} failures:", flush=True)
        for rid, msg in failures:
            print(f"{tag}   {rid}: {msg}", flush=True)
        sys.exit(2)

    print(f"{tag} all files processed; queue empty", flush=True)


if __name__ == "__main__":
    main()
