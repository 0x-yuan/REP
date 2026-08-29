"""Per-model inbox queue runner for inference-farm.

Points at:
    - inbox/ result/ processed/ inside `inference-farm/`
    - app `sglang-slave-<model>` (class `SGLangSlave`)

Pipeline per file is identical: upload → spawn → poll → fetch → archive.

Usage (one process per model):
    MODAL_PROFILE=<your-modal-profile> uv run python -u \
        inference-farm/master/queue_runner.py --model qwen3-1p7b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

# Add slave/ to sys.path so we can import the shared naming helpers
# (the master process never runs inside Modal; it just borrows slave/
# as a single source of truth for resource names).
_SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVER_ROOT / "slave"))
from naming import (  # noqa: E402
    app_name as _slave_app_name,
    data_volume_name,
)

INBOX = _SERVER_ROOT / "inbox"
RESULT = _SERVER_ROOT / "result"
PROCESSED = _SERVER_ROOT / "processed"
DATA_VOL_NAME = data_volume_name()
POST_SPAWN_GRACE_S = 90
DEFAULT_STALL_TIMEOUT_S = int(os.environ.get("SGLANG_STALL_TIMEOUT_S", "1200"))
# How often to print a status line even when n_done hasn't ticked. Without
# this, large chunk_size jobs go silent for minutes between chunk boundaries.
DEFAULT_STATUS_PRINT_INTERVAL_S = float(
    os.environ.get("SGLANG_STATUS_PRINT_INTERVAL_S", "30")
)


# ---------- progress formatting helpers ----------

def _fmt_dur(seconds: float | int | None) -> str:
    """Compact h/m/s. Returns '?' for None / negative."""
    if seconds is None or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _eta_seconds(n_done: int, n_total: int, elapsed_s: float) -> float | None:
    """Linear-extrapolation ETA. Returns None if unknown."""
    if n_done <= 0 or n_total <= 0 or elapsed_s <= 0 or n_done >= n_total:
        return None
    rate = n_done / elapsed_s
    if rate <= 0:
        return None
    return (n_total - n_done) / rate


def _progress_bar(n_done: int, n_total: int, width: int = 20) -> str:
    if n_total <= 0:
        return "[" + "?" * width + "]"
    n_done = max(0, min(n_done, n_total))
    filled = int(round(width * n_done / n_total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_status_line(
    *,
    tag: str,
    rec: dict,
    spawned_at: float,
    file_name: str | None = None,
) -> str:
    """One-line dense status: progress bar + counts + elapsed + eta + tps +
    chunk_active_s if mid-chunk."""
    n_done = rec.get("n_done", 0) or 0
    n_total = rec.get("n_total", 0) or 0
    pct = (100.0 * n_done / n_total) if n_total else 0.0
    elapsed = max(0.0, time.time() - spawned_at)
    eta = _eta_seconds(n_done, n_total, elapsed)
    tps = rec.get("tps_decode")
    cur = rec.get("current_row_id")
    status = rec.get("status", "?")
    phase = (rec.get("_state") or {}).get("phase", "?")
    bar = _progress_bar(n_done, n_total)

    chunk_idx = rec.get("chunk_index")
    chunk_n_chunks = rec.get("chunk_n_chunks")
    chunk_active = rec.get("chunk_active_s")
    chunk_str = ""
    if (
        chunk_idx is not None and chunk_idx >= 0
        and chunk_n_chunks
        and chunk_active is not None
        and chunk_active > 0
    ):
        chunk_str = (
            f" chunk={chunk_idx + 1}/{chunk_n_chunks}@{int(chunk_active)}s"
        )

    # heartbeat staleness
    hb_at = rec.get("heartbeat_at")
    hb_str = ""
    if hb_at:
        hb_age = time.time() - float(hb_at)
        if hb_age > 30:
            hb_str = f" hb_age={int(hb_age)}s"

    file_str = f" file={file_name}" if file_name else ""
    return (
        f"{tag}{file_str} {bar} {n_done}/{n_total} ({pct:.1f}%) "
        f"elapsed={_fmt_dur(elapsed)} eta={_fmt_dur(eta)} "
        f"tps={tps if tps is not None else '?'} "
        f"phase={phase} status={status}{chunk_str}{hb_str} "
        f"cur={cur}"
    )


def _stable_batch_id(name: str) -> str:
    # `sglang_` namespace so checkpoints don't collide with any other
    # engine's checkpoints (different engine, can't reuse).
    return f"sglang_{hashlib.sha1(name.encode()).hexdigest()[:16]}"


def _list_inbox_files(model: str) -> list[Path]:
    return sorted(
        p
        for p in INBOX.glob(f"{model}__*.jsonl")
        if not p.name.endswith(".meta.jsonl")
    )


def _count_lines(p: Path) -> int:
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def _upload_to_volume(local: Path, remote_path: str, tag: str) -> None:
    sz_gb = local.stat().st_size / 1e9
    print(
        f"{tag} upload {local.name} ({sz_gb:.2f} GB) → "
        f"{DATA_VOL_NAME}:/{remote_path}",
        flush=True,
    )
    cmd = [
        "modal",
        "volume",
        "put",
        DATA_VOL_NAME,
        str(local),
        remote_path,
        "--force",
    ]
    rc = subprocess.run(cmd, check=False)
    if rc.returncode != 0:
        raise RuntimeError(
            f"{tag} modal volume put failed (exit={rc.returncode})"
        )


def _archive(jsonl_path: Path, tag: str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    meta_path = jsonl_path.with_suffix(".meta.jsonl")
    moved: list[str] = []
    for src in (jsonl_path, meta_path):
        if src.exists():
            shutil.move(str(src), str(PROCESSED / src.name))
            moved.append(src.name)
    print(f"{tag} archived: {moved}", flush=True)


def _is_already_done(jsonl_path: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        n_in = _count_lines(jsonl_path)
        n_out = _count_lines(out_path)
    except Exception:
        return False
    return n_out == n_in and n_in > 0


def process_one(
    slave,
    jsonl_path: Path,
    tag: str,
    poll_interval: float,
    max_retries: int,
    stall_timeout_s: int | None = None,
    status_print_interval_s: float | None = None,
) -> None:
    """Public wrapper around _process_one for use by multi_replica_runner.

    `slave` may belong to any replica app — the function only needs the
    inference_from_volume / progress / get_results methods which are the
    same on every SGLangSlave instance.
    """
    return _process_one(
        slave,
        jsonl_path,
        tag,
        poll_interval,
        max_retries,
        stall_timeout_s=stall_timeout_s,
        status_print_interval_s=status_print_interval_s,
    )


def _process_one(
    slave,
    jsonl_path: Path,
    tag: str,
    poll_interval: float,
    max_retries: int,
    stall_timeout_s: int | None = None,
    status_print_interval_s: float | None = None,
) -> None:
    out_path = RESULT / jsonl_path.name
    if not jsonl_path.exists():
        print(f"{tag} skip {jsonl_path.name}: removed from inbox", flush=True)
        return
    if _is_already_done(jsonl_path, out_path):
        print(
            f"{tag} skip {jsonl_path.name}: already in result/ "
            f"({_count_lines(out_path)} rows)",
            flush=True,
        )
        _archive(jsonl_path, tag)
        return

    bid = _stable_batch_id(jsonl_path.name)
    print(f"{tag} processing {jsonl_path.name} batch_id={bid}", flush=True)

    vol_remote = f"inbox/{jsonl_path.name}"
    _upload_to_volume(jsonl_path, vol_remote, tag)

    n_total = _count_lines(jsonl_path)
    last_done = -1
    last_print_at = 0.0
    STALL_TIMEOUT_S = stall_timeout_s if stall_timeout_s is not None else DEFAULT_STALL_TIMEOUT_S
    PRINT_EVERY_S = (
        float(status_print_interval_s)
        if status_print_interval_s is not None
        else DEFAULT_STATUS_PRINT_INTERVAL_S
    )
    for attempt in range(max_retries + 1):
        if attempt > 0:
            backoff = min(30 * (2 ** (attempt - 1)), 300)
            print(
                f"{tag} retry {attempt}/{max_retries} after {backoff}s "
                f"(slave auto-resumes from checkpoint)",
                flush=True,
            )
            time.sleep(backoff)

        # multi-replica: bump chunk_size so SGLang continuously batches
        # across many rows. Default chunk_size = max_running_requests=16 means
        # each 16-row chunk is wall-bottlenecked by its slowest req hitting
        # max_tokens. Setting chunk_size=128 lets the engine refill the 16
        # concurrent slots from a 128-row pool, paying tail latency once per
        # 128 rows instead of once per 16.
        import os as _os
        _cs = int(_os.environ.get("EXP_B7_CHUNK_SIZE", "128"))
        fc = slave.inference_from_volume.spawn(
            vol_path=vol_remote, batch_id=bid, chunk_size=_cs,
        )
        spawned_at = time.time()
        last_progress_at = spawned_at
        last_print_at = 0.0
        print(
            f"{tag} spawn fc={fc.object_id} file={jsonl_path.name} "
            f"n_total={n_total} (attempt {attempt + 1}/{max_retries + 1})",
            flush=True,
        )

        while True:
            try:
                rec = slave.progress.remote(bid)
            except Exception as e:
                print(f"{tag} poll error: {e}; sleep 5s", flush=True)
                time.sleep(5)
                continue

            status = rec.get("status", "?")
            n_done = rec.get("n_done", 0)
            n_tot = rec.get("n_total", n_total)
            submitted_at = rec.get("submitted_at") or 0
            elapsed = time.time() - spawned_at
            now = time.time()

            progress_changed = n_done != last_done
            terminal = status in {"done", "failed"}
            interval_due = (now - last_print_at) >= PRINT_EVERY_S
            if progress_changed or terminal or interval_due:
                print(
                    _format_status_line(
                        tag=tag,
                        rec=rec,
                        spawned_at=spawned_at,
                        file_name=jsonl_path.name,
                    ),
                    flush=True,
                )
                last_print_at = now
                if progress_changed:
                    last_progress_at = now
                    last_done = n_done

            if status == "done":
                results = slave.get_results.remote(bid)
                results = [
                    (
                        r
                        if isinstance(r, dict)
                        else {
                            "id": "?",
                            "outputs": [],
                            "prompt_tokens": 0,
                            "error": "null_row_in_checkpoint",
                        }
                    )
                    for r in results
                ]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_err = sum(1 for r in results if r.get("error"))
                print(
                    f"{tag} wrote {len(results)} rows → {out_path.name}  "
                    f"(ok={len(results) - n_err}, err={n_err})",
                    flush=True,
                )
                _archive(jsonl_path, tag)
                return

            if status == "failed":
                if (
                    submitted_at < spawned_at - 1
                    or elapsed < POST_SPAWN_GRACE_S
                ):
                    time.sleep(poll_interval)
                    continue
                err = rec.get("error", "(no error msg)")
                print(
                    f"{tag} FAILED at n={n_done}/{n_tot}: {err[:200]}",
                    flush=True,
                )
                break

            stall_elapsed = time.time() - last_progress_at
            if (
                stall_elapsed > STALL_TIMEOUT_S
                and elapsed > POST_SPAWN_GRACE_S
            ):
                print(
                    f"{tag} STALLED at n={n_done}/{n_tot} for "
                    f"{int(stall_elapsed)}s; cancelling fc and respawning",
                    flush=True,
                )
                try:
                    fc.cancel()
                except Exception as e:
                    print(f"{tag} fc.cancel() error: {e}", flush=True)
                break

            time.sleep(poll_interval)

    raise SystemExit(
        f"{tag} {jsonl_path.name} still failing after {max_retries} "
        f"retries; checkpoint preserved (batch_id={bid})."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--poll-interval", default=10.0, type=float)
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stay alive after inbox drains, polling for new files. "
        "Without this flag, exit when the inbox is empty.",
    )
    parser.add_argument(
        "--inbox-scan-interval-s",
        type=float,
        default=5.0,
        help="how often to rescan inbox/ for new files",
    )
    parser.add_argument(
        "--drain-quiet-s",
        type=float,
        default=15.0,
        help="exit after the inbox has been empty for this many consecutive "
        "seconds (drain mode only)",
    )
    parser.add_argument(
        "--status-print-interval-s",
        type=float,
        default=None,
        help="how often to reprint a progress line even when n_done has not "
        "changed; default uses SGLANG_STATUS_PRINT_INTERVAL_S env or 30s",
    )
    args = parser.parse_args()

    tag = f"[queue/{args.model}]"
    mode = "watch" if args.watch else "drain"
    print(
        f"{tag} start; inbox={INBOX} result={RESULT} mode={mode}",
        flush=True,
    )

    Slave = modal.Cls.from_name(_slave_app_name(args.model), "SGLangSlave")
    slave = Slave()
    info = slave.ping.remote()
    print(f"{tag} ping ← phase={info['_state']['phase']}", flush=True)

    # Source of truth = inbox/. _process_one archives on success (file
    # disappears from inbox). On unrecoverable failure it raises SystemExit.
    # That means a file in inbox that we've never processed is always
    # eligible — no need for a separate `seen` set.
    quiet_start: float | None = None
    while True:
        files = _list_inbox_files(args.model)
        if files:
            print(
                f"{tag} picked up {len(files)} files: "
                f"{[f.name for f in files]}",
                flush=True,
            )
            for f in files:
                if not f.exists():
                    print(
                        f"{tag} skip {f.name}: removed before processing",
                        flush=True,
                    )
                    continue
                _process_one(
                    slave,
                    f,
                    tag,
                    args.poll_interval,
                    args.max_retries,
                    status_print_interval_s=args.status_print_interval_s,
                )
            quiet_start = None
            continue

        now = time.time()
        if quiet_start is None:
            quiet_start = now
        elif not args.watch and (now - quiet_start) >= args.drain_quiet_s:
            print(
                f"{tag} inbox quiet {int(now - quiet_start)}s; exiting",
                flush=True,
            )
            break
        time.sleep(max(1.0, args.inbox_scan_interval_s))

    print(f"{tag} all files processed; queue empty", flush=True)


if __name__ == "__main__":
    main()
