"""CLI for inference-farm inbox queue management.

The actual queue is the contents of `inference-farm/inbox/` filtered by
filename prefix `<model>__*.jsonl`. The runners (queue_runner /
multi_replica_runner) periodically rescan this directory, so anything
you add or remove here is picked up live (within the configured
inbox-scan interval, default 5s).

This CLI just wraps the directory operations with the model-prefix
convention enforced and a `list / add / remove / clear / status` UX.

Usage:
    # Show pending work for a model.
    python inference-farm/master/queue_cli.py list qwen3-8b

    # Queue a file (must already follow the `<model>__<batch>.jsonl`
    # naming convention OR pass --rename-as).
    python inference-farm/master/queue_cli.py add qwen3-8b path/to/run.jsonl
    python inference-farm/master/queue_cli.py add qwen3-8b run.jsonl \\
        --rename-as run42

    # Drop a queued file (it must NOT yet be in flight).
    python inference-farm/master/queue_cli.py remove qwen3-8b 'qwen3-8b__run_*.jsonl'

    # Drop all queued files for a model.
    python inference-farm/master/queue_cli.py clear qwen3-8b

    # Live status from the slave's progress dict (requires Modal auth).
    python inference-farm/master/queue_cli.py status qwen3-8b
    python inference-farm/master/queue_cli.py status qwen3-8b --replicas 4
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import time
from pathlib import Path

_SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVER_ROOT / "slave"))
from naming import progress_dict_name  # noqa: E402

INBOX = _SERVER_ROOT / "inbox"
PROCESSED = _SERVER_ROOT / "processed"
RESULT = _SERVER_ROOT / "result"


def _model_inbox_files(model: str) -> list[Path]:
    """Pending inbox files for `model`, excluding any `.meta.jsonl` companions."""
    if not INBOX.exists():
        return []
    return sorted(
        p
        for p in INBOX.glob(f"{model}__*.jsonl")
        if not p.name.endswith(".meta.jsonl")
    )


def _count_lines(p: Path) -> int:
    try:
        n = 0
        with p.open("rb") as f:
            for _ in f:
                n += 1
        return n
    except Exception:
        return -1


def _validate_name(model: str, name: str) -> None:
    if not name.startswith(f"{model}__"):
        raise SystemExit(
            f"file name {name!r} must start with '{model}__' so the runner's "
            f"inbox filter picks it up. Pass --rename-as <batch_name> to fix."
        )
    if not name.endswith(".jsonl") or name.endswith(".meta.jsonl"):
        raise SystemExit(
            f"file name {name!r} must end with '.jsonl' (and not '.meta.jsonl')."
        )


def cmd_list(args: argparse.Namespace) -> int:
    files = _model_inbox_files(args.model)
    if not files:
        print(f"(inbox empty for model {args.model!r})")
        return 0
    print(
        f"# {len(files)} file(s) queued for {args.model} "
        f"(first to be picked up at the top):"
    )
    for i, f in enumerate(files, start=1):
        n_lines = _count_lines(f)
        sz_mb = f.stat().st_size / 1e6
        n_str = f"{n_lines} rows" if n_lines >= 0 else "? rows"
        print(f"  {i:3d}. {f.name}  ({n_str}, {sz_mb:.2f} MB)")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"source file not found: {src}")

    if args.rename_as:
        dest_name = f"{args.model}__{args.rename_as}.jsonl"
    else:
        dest_name = src.name
    _validate_name(args.model, dest_name)

    INBOX.mkdir(parents=True, exist_ok=True)
    dst = INBOX / dest_name
    if dst.exists():
        if not args.force:
            raise SystemExit(
                f"refusing to overwrite existing inbox file: {dst}\n"
                "(pass --force to replace it, or remove it first)"
            )
        dst.unlink()

    if args.move:
        shutil.move(str(src), str(dst))
        op = "moved"
    else:
        shutil.copy2(str(src), str(dst))
        op = "copied"
    n_lines = _count_lines(dst)
    sz_mb = dst.stat().st_size / 1e6
    print(f"{op}: {src} → {dst}  ({n_lines} rows, {sz_mb:.2f} MB)")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    pattern = args.pattern
    if "/" in pattern or "\\" in pattern:
        raise SystemExit("pattern must be a filename glob, not a path")
    files = _model_inbox_files(args.model)
    matched = [f for f in files if fnmatch.fnmatch(f.name, pattern)]
    if not matched:
        print(
            f"(no inbox files match pattern {pattern!r} for model {args.model!r})"
        )
        return 1
    if not args.yes:
        print(f"# would remove {len(matched)} file(s) for {args.model}:")
        for f in matched:
            print(f"  - {f.name}")
        print("(re-run with --yes to actually remove them)")
        return 0
    for f in matched:
        try:
            f.unlink()
            print(f"removed: {f.name}")
        except FileNotFoundError:
            print(f"already gone: {f.name}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    files = _model_inbox_files(args.model)
    if not files:
        print(f"(inbox empty for model {args.model!r})")
        return 0
    if not args.yes:
        print(f"# would clear {len(files)} file(s) for {args.model}:")
        for f in files:
            print(f"  - {f.name}")
        print("(re-run with --yes to actually clear)")
        return 0
    for f in files:
        try:
            f.unlink()
            print(f"removed: {f.name}")
        except FileNotFoundError:
            pass
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Query each replica's progress Dict via Modal and dump a summary."""
    try:
        import modal
    except ImportError as e:
        raise SystemExit(f"modal SDK not installed: {e}")

    replicas = (
        [""] if args.replicas <= 0 else [f"r{i}" for i in range(args.replicas)]
    )
    print(
        f"# status for {args.model} (replicas={replicas or ['(legacy)']}); "
        f"queue = {len(_model_inbox_files(args.model))} pending file(s)"
    )

    for rid in replicas:
        dict_name = progress_dict_name(args.model, rid)
        try:
            d = modal.Dict.from_name(dict_name)
            keys = list(d.keys())
        except Exception as e:
            print(f"  [{rid or '(legacy)'}] Dict {dict_name!r} not reachable: {e!r}")
            continue
        if not keys:
            print(f"  [{rid or '(legacy)'}] no in-flight batches")
            continue
        for k in keys:
            try:
                rec = dict(d[k])
            except Exception as e:
                print(f"  [{rid or '(legacy)'}] {k}: read error: {e!r}")
                continue
            status = rec.get("status", "?")
            n_done = rec.get("n_done", 0)
            n_total = rec.get("n_total", 0)
            pct = (100.0 * n_done / n_total) if n_total else 0.0
            chunk_idx = rec.get("chunk_index", -1)
            chunk_n = rec.get("chunk_n_chunks", 0)
            chunk_active = rec.get("chunk_active_s", 0.0)
            hb_at = rec.get("heartbeat_at")
            hb_age = (time.time() - float(hb_at)) if hb_at else None
            print(
                f"  [{rid or '(legacy)'}] batch={k} status={status} "
                f"n={n_done}/{n_total} ({pct:.1f}%) "
                f"chunk={chunk_idx + 1 if chunk_idx >= 0 else '-'}/{chunk_n} "
                f"chunk_active={int(chunk_active)}s "
                f"hb_age={int(hb_age) if hb_age is not None else '?'}s"
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="queue_cli", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="show queued inbox files for a model")
    p.add_argument("model")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("add", help="copy/move a file into the inbox")
    p.add_argument("model")
    p.add_argument("file", help="source JSONL path")
    p.add_argument(
        "--rename-as",
        default=None,
        help="treat the source file as <model>__<rename-as>.jsonl in the inbox",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="move (not copy) the source file",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite if the inbox already has a file with this name",
    )
    p.set_defaults(func=cmd_add)

    p = sub.add_parser(
        "remove", help="remove inbox files matching a filename glob"
    )
    p.add_argument("model")
    p.add_argument(
        "pattern",
        help="filename glob, e.g. 'qwen3-8b__run_*.jsonl'",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="actually delete (default: dry-run)",
    )
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("clear", help="remove all inbox files for a model")
    p.add_argument("model")
    p.add_argument(
        "--yes",
        action="store_true",
        help="actually delete (default: dry-run)",
    )
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser(
        "status", help="live status from the slave's progress Dict (Modal)"
    )
    p.add_argument("model")
    p.add_argument(
        "--replicas",
        type=int,
        default=0,
        help="number of replicas to query (0 = legacy single-replica)",
    )
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
