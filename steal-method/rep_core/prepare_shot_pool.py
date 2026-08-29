"""Build a shadow demonstration pool file for the REP builders from a local JSONL.

The release already ships the pools under ``public/data/shot_pool/`` (the
deterministic seed-7 50-row draws). This utility rebuilds such a file from your
own shadow-model harvest: drop truncated traces, draw
``random.Random(7).sample(non_truncated, 50)`` (the seed the paper reports, so
growing k appends shots without reshuffling) and write::

    public/data/shot_pool/<config>.jsonl.gz

Input rows (JSONL, one per shadow-model demo)::

    {"id": ..., "x": <question>, "trace": <reasoning, <think> tags stripped>,
     "answer": ..., "truncated": false}

Output rows: ``{id, question, think, answer, truncated: false}``.

Run::

    python prepare_shot_pool.py --pool my_pool.jsonl --config qwen3_14b
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

N_DEMOS = 50
SAMPLE_SEED = 7

_HERE = Path(__file__).resolve()
# vendored location the builders read from (public/data/shot_pool/)
OUT_DIR = _HERE.parents[2] / "data" / "shot_pool"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_pool(pool_path: Path, config: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(pool_path)
    print(f"[shot_pool] {len(rows)} raw rows from {pool_path}", flush=True)

    kept = [
        {
            "id": r["id"],
            "question": r["x"],
            "think": r["trace"],   # <think>-stripped shadow reasoning
            "answer": r["answer"],
            "truncated": False,
        }
        for r in rows if not bool(r.get("truncated", False))
    ]
    if len(kept) < N_DEMOS:
        raise SystemExit(
            f"[shot_pool] {config}: only {len(kept)} non-truncated rows (<{N_DEMOS})"
        )

    sampled = random.Random(SAMPLE_SEED).sample(kept, N_DEMOS)
    out_path = out_dir / f"{config}.jsonl.gz"
    with gzip.open(out_path, "wt") as f:
        for row in sampled:
            f.write(json.dumps(row) + "\n")
    print(f"[shot_pool] wrote {len(sampled)} demos -> {out_path}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", type=Path, required=True,
                    help="local JSONL(.gz) of shadow demos {id, x, trace, answer, truncated}")
    ap.add_argument("--config", default="qwen3_14b",
                    help="output name / shadow-model config (e.g. qwen3_14b, qwen3_32b)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    build_pool(args.pool, args.config, args.out_dir)


if __name__ == "__main__":
    main()
