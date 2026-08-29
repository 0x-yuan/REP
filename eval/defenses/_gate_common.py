"""Shared resumable driver for the input-gate scripts (ProB / KAD / guard).

A gate looks at the *user turn* of each REP prompt row (or a benign slice of it,
see ``prompt_lib.benign_text``), calls a detector, and appends one JSON decision
per row to ``--out``. Rows already present in ``--out`` are skipped.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from defenses.prompt_lib import benign_text, parse, row_idx

MODES = ("rep", "benign", "benign_long")


def gate_argparser(doc: str, modes=MODES, concurrency: int = 24, limit: int = 0) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--prompts", required=True, help="REP prompt rows jsonl (rendered Qwen3 chat)")
    ap.add_argument("--out", required=True, help="decisions jsonl (append / resume)")
    ap.add_argument("--mode", choices=modes, default="rep",
                    help="rep = full [q;S] user turn; benign = bare question; "
                         "benign_long = worked-examples block only (~30K)")
    ap.add_argument("--limit", type=int, default=limit)
    ap.add_argument("--concurrency", type=int, default=concurrency)
    return ap


def load_rows(path: str, limit: int) -> list[dict]:
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    return rows[:limit] if limit else rows


def load_done(path: str, key: str | None = None) -> dict[str, dict]:
    """idx -> decision for rows already in ``path`` (optionally only rows having ``key``)."""
    done = {}
    p = Path(path)
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if key is None or key in r:
                done[str(r["idx"])] = r
    return done


def run_gate(args, fn: Callable[[str], dict], tag: str, done_key: str | None = None) -> list[dict]:
    """Apply ``fn(text) -> dict`` to every not-yet-done row; return all decisions."""
    rows = load_rows(args.prompts, args.limit)
    done = load_done(args.out, done_key)
    tasks = []
    for r in rows:
        idx = row_idx(r)
        if idx in done:
            continue
        _sys, user = parse(r["prompt"])
        tasks.append((idx, benign_text(user, args.mode)))
    print(f"[{tag}:{args.mode}] total={len(rows)} done={len(done)} todo={len(tasks)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(fn, t): i for (i, t) in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            res["idx"] = futs[fut]
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if k % 10 == 0:
                print(f"  {k}/{len(tasks)}", flush=True)
    return list(load_done(args.out).values())
