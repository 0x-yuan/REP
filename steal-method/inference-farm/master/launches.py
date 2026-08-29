"""CLI for inference-farm pipeline launches.

A "launch" is one invocation of `master/run_pipeline.sh`. The script
generates a unique LAUNCH_ID (8-char hex by default) and writes a registry
entry to `inference-farm/launches/<launch_id>.json`. Every line in the
launch's log file `/tmp/sglang_pipeline_<launch_id>.log` is prefixed with
`[L:<launch_id>]` so N parallel launches can stream into the same terminal
without their output mixing.

This CLI reads the registry directory + the launch logs to give an operator
view that survives across terminals / sessions.

Subcommands:
    list                       — show every registered launch, with alive/dead status
    tail <launch_id> [--replica r0]
                               — `tail -F` the launch's log; --replica filters
                                 inner per-replica lines
    status <launch_id>         — query each replica's progress Dict via Modal
                                 (uses model + replicas from the registry)
    stop <launch_id>           — SIGTERM the recorded pid; the bash script's
                                 EXIT trap will mark the registry as exited
    gc                         — drop registry entries whose pid is dead AND
                                 status=exited (run periodically to keep the
                                 registry tidy)

The CLI never modifies the inbox/result/processed dirs; it only reads the
launches/ registry and (for `status`) talks to Modal.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SGLANG_DIR = _HERE.parent
LAUNCHES_DIR = SGLANG_DIR / "launches"

# Single source of truth for Modal resource names (handles EXP_ID prefixing).
# Lives in slave/, importable from master/ via the sibling-package layout.
if str(SGLANG_DIR) not in sys.path:
    sys.path.insert(0, str(SGLANG_DIR))
from slave import naming  # noqa: E402


# --- registry I/O ----------------------------------------------------------


def _registry_path(launch_id: str) -> Path:
    return LAUNCHES_DIR / f"{launch_id}.json"


def _read_registry(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _list_registry() -> list[dict]:
    if not LAUNCHES_DIR.exists():
        return []
    out: list[dict] = []
    for p in sorted(LAUNCHES_DIR.glob("*.json")):
        rec = _read_registry(p)
        if rec is None:
            continue
        rec["_path"] = str(p)
        out.append(rec)
    return out


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM  # exists but not ours = alive


def _resolve_launch(prefix: str) -> dict:
    """Resolve `prefix` to exactly one registry entry. Accepts either a full
    launch_id or a unique prefix."""
    all_entries = _list_registry()
    matches = [
        r
        for r in all_entries
        if r.get("launch_id") == prefix
        or (r.get("launch_id") or "").startswith(prefix)
    ]
    if not matches:
        raise SystemExit(f"no launch matches {prefix!r}")
    if len(matches) > 1:
        ids = ", ".join(r["launch_id"] for r in matches)
        raise SystemExit(
            f"prefix {prefix!r} is ambiguous; matches: {ids}"
        )
    return matches[0]


# --- formatting ------------------------------------------------------------


def _fmt_age(started_at: str | None) -> str:
    if not started_at:
        return "?"
    try:
        # ISO 8601 like 2026-05-09T18:05:33Z
        from datetime import datetime, timezone

        dt = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        delta = (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return "?"
    if delta < 60:
        return f"{int(delta)}s"
    if delta < 3600:
        return f"{int(delta // 60)}m{int(delta % 60):02d}s"
    if delta < 86400:
        return f"{int(delta // 3600)}h{int((delta % 3600) // 60):02d}m"
    return f"{int(delta // 86400)}d{int((delta % 86400) // 3600):02d}h"


def _runtime_state(rec: dict) -> str:
    status = rec.get("status", "?")
    pid = rec.get("pid")
    alive = _pid_alive(pid)
    if status == "running" and alive:
        return "RUNNING"
    if status == "running" and not alive:
        return "ZOMBIE"  # bash crashed without trap firing
    if status == "exited":
        rc = rec.get("exit_code")
        return f"EXITED({rc})" if rc is not None else "EXITED"
    return status.upper()


# --- subcommands -----------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    entries = _list_registry()
    if not entries:
        print("(no launches registered)")
        return 0
    if args.active:
        entries = [
            r
            for r in entries
            if r.get("status") == "running" and _pid_alive(r.get("pid"))
        ]
        if not entries:
            print("(no active launches)")
            return 0
    print(
        f"# {len(entries)} launch(es); registry={LAUNCHES_DIR}"
    )
    print(
        f"{'LAUNCH_ID':<10}  {'STATE':<14} {'MODEL':<14} {'PROFILE':<14} "
        f"{'R':<3} {'WATCH':<5} {'AGE':<10} PID"
    )
    for r in entries:
        lid = r.get("launch_id", "?")
        state = _runtime_state(r)
        model = r.get("model", "?")
        profile = r.get("profile", "?")
        n_repl = r.get("replicas", "?")
        watch = "1" if r.get("watch") else "0"
        age = _fmt_age(r.get("started_at"))
        pid = r.get("pid", "?")
        print(
            f"{lid:<10}  {state:<14} {model:<14} {profile:<14} "
            f"{str(n_repl):<3} {watch:<5} {age:<10} {pid}"
        )
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    rec = _resolve_launch(args.launch_id)
    log_path = rec.get("log_path")
    if not log_path:
        raise SystemExit(f"launch {rec['launch_id']!r} has no log_path")
    log = Path(log_path)
    # tail -F follows the file even if it doesn't exist yet (just-launched).
    cmd = ["tail", "-F", "-n", str(args.lines), str(log)]
    if args.replica:
        # Per-replica filter: only inner lines tagged [r0/...] survive.
        # Use `grep --line-buffered` to avoid pipe stalling.
        # Use a fixed string match to avoid regex escaping surprises.
        tag = f"[{args.replica}/"
        ps = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            grep = subprocess.Popen(
                ["grep", "--line-buffered", "-F", tag],
                stdin=ps.stdout,
            )
            assert ps.stdout is not None
            ps.stdout.close()  # let grep see EOF if tail exits
            grep.wait()
        finally:
            ps.terminate()
        return grep.returncode if grep.returncode in (0, 1) else 0
    os.execvp("tail", cmd)
    return 0  # unreachable


def cmd_status(args: argparse.Namespace) -> int:
    rec = _resolve_launch(args.launch_id)
    try:
        import modal
    except ImportError as e:
        raise SystemExit(f"modal SDK not installed: {e}")

    model = rec["model"]
    n_repl = int(rec.get("replicas", 1))
    # exp_id is part of the registry written by run_pipeline.sh; older entries
    # may not have it (treat as legacy / empty).
    exp_id = rec.get("exp_id") or ""
    print(
        f"# launch {rec['launch_id']}  state={_runtime_state(rec)}  "
        f"exp_id={exp_id or '<unset>'}  model={model}  "
        f"profile={rec.get('profile')}  replicas={n_repl}  "
        f"log={rec.get('log_path')}"
    )

    replicas = (
        [""] if n_repl <= 1 else [f"r{i}" for i in range(n_repl)]
    )
    for rid in replicas:
        # Route through slave/naming.py so the dict name follows the same
        # rules the slave used at deploy time (EXP_ID prefix + replica suffix).
        dict_name = naming.progress_dict_name(model, replica_id=rid, exp_id=exp_id)
        try:
            d = modal.Dict.from_name(dict_name)
            keys = list(d.keys())
        except Exception as e:
            print(
                f"  [{rid or '(legacy)'}] Dict {dict_name!r} not reachable: {e!r}"
            )
            continue
        if not keys:
            print(f"  [{rid or '(legacy)'}] no in-flight batches")
            continue
        for k in keys:
            try:
                pr = dict(d[k])
            except Exception as e:
                print(f"  [{rid or '(legacy)'}] {k}: read error: {e!r}")
                continue
            status = pr.get("status", "?")
            n_done = pr.get("n_done", 0)
            n_total = pr.get("n_total", 0)
            pct = (100.0 * n_done / n_total) if n_total else 0.0
            chunk_idx = pr.get("chunk_index", -1)
            chunk_n = pr.get("chunk_n_chunks", 0)
            chunk_active = pr.get("chunk_active_s", 0.0)
            hb_at = pr.get("heartbeat_at")
            hb_age = (time.time() - float(hb_at)) if hb_at else None
            print(
                f"  [{rid or '(legacy)'}] batch={k} status={status} "
                f"n={n_done}/{n_total} ({pct:.1f}%) "
                f"chunk={(chunk_idx + 1) if chunk_idx >= 0 else '-'}/{chunk_n} "
                f"chunk_active={int(chunk_active)}s "
                f"hb_age={int(hb_age) if hb_age is not None else '?'}s"
            )
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    rec = _resolve_launch(args.launch_id)
    pid = rec.get("pid")
    if not pid:
        raise SystemExit(f"launch {rec['launch_id']!r} has no recorded pid")
    if not _pid_alive(pid):
        print(f"launch {rec['launch_id']} pid={pid} already dead")
        return 0
    sig = signal.SIGKILL if args.force else signal.SIGTERM
    name = "SIGKILL" if args.force else "SIGTERM"
    try:
        os.kill(pid, sig)
        print(f"sent {name} to launch {rec['launch_id']} pid={pid}")
    except ProcessLookupError:
        print(f"pid {pid} disappeared before signal")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    entries = _list_registry()
    n_dropped = 0
    for r in entries:
        path = Path(r["_path"])
        status = r.get("status")
        pid = r.get("pid")
        # Drop "exited" entries unconditionally; drop "running" entries whose
        # pid is dead (= bash crashed before the EXIT trap could fire).
        is_zombie = status == "running" and not _pid_alive(pid)
        is_clean_exit = status == "exited"
        if not (is_zombie or is_clean_exit):
            continue
        if not args.yes:
            print(
                f"would drop {path.name} (status={status}, pid={pid}, alive={_pid_alive(pid)})"
            )
            n_dropped += 1
            continue
        try:
            path.unlink()
            print(f"dropped {path.name}")
            n_dropped += 1
        except FileNotFoundError:
            pass
    if n_dropped == 0:
        print("(nothing to gc)")
    elif not args.yes:
        print(
            f"# would drop {n_dropped} entries; rerun with --yes to actually delete"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="launches", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="show registered launches")
    p.add_argument(
        "--active",
        action="store_true",
        help="only show launches whose status=running AND pid is alive",
    )
    p.set_defaults(func=cmd_list)

    p = sub.add_parser(
        "tail", help="tail -F the launch log; --replica filters inner lines"
    )
    p.add_argument("launch_id", help="full LAUNCH_ID or a unique prefix")
    p.add_argument(
        "--replica",
        default=None,
        help="filter to lines tagged [<replica>/...] (e.g. r0)",
    )
    p.add_argument(
        "-n",
        "--lines",
        type=int,
        default=20,
        help="initial lines to print (forwarded to tail -n)",
    )
    p.set_defaults(func=cmd_tail)

    p = sub.add_parser(
        "status",
        help="live progress for the launch's replicas (Modal Dict reads)",
    )
    p.add_argument("launch_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "stop",
        help="SIGTERM the bash pipeline (its EXIT trap marks the registry exited)",
    )
    p.add_argument("launch_id")
    p.add_argument(
        "--force",
        action="store_true",
        help="SIGKILL instead of SIGTERM (skips graceful drain)",
    )
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser(
        "gc",
        help="drop registry entries whose process is dead",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="actually delete (default: dry-run)",
    )
    p.set_defaults(func=cmd_gc)

    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
