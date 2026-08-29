"""REP distillation corpus — step 2: write the inference-farm inbox JSONL.

Concatenates the rendered cell prefix (``build_prompts.py``) with each query
and writes one inbox row per question::

    {"id": "b2::<prompt_id>", "prompt": <prefix+question+post_marker>,
     "max_tokens": 12800, "temperature": 0.0, "_meta": {...}}

Filename convention is ``<victim_key>__<batch>.jsonl`` (the farm keys the
victim model off the prefix). Decoding is greedy; ``max_tokens`` defaults to
12 800 — the published corpus re-ran the ``finish_reason == "length"`` tail at
32k and 80k (``--max-tokens`` + ``--only-ids``), and the top-up harvest used
24 000.

``--shards N`` splits the file into N contiguous chunks
(``<victim>__<batch>__shardNN.jsonl``) so replicas keep prefix-cache locality.

Run::

    python assemble_inbox.py                                # qwen3-14b__distill10k.jsonl
    python assemble_inbox.py --queries prompts/queries/topup_25k.jsonl \\
        --batch topup25k --max-tokens 24000 --shards 25
    python assemble_inbox.py --only-ids retry_ids.txt --batch distill10k_v4 --max-tokens 32000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXP_DIR = _HERE.parent
sys.path.insert(0, str(EXP_DIR))

from corpus_lib import ID_PREFIX  # noqa: E402

VICTIM_KEY = "qwen3-14b"
MAX_NEW_TOKENS = 12800
TEMPERATURE = 0.0
DEFAULT_INBOX = _HERE.parents[2] / "inference-farm" / "inbox"


def make_row(cell: dict, q: dict, *, max_tokens: int, temperature: float,
             victim: str) -> dict:
    return {
        "id": f"{ID_PREFIX}{q['prompt_id']}",
        "prompt": cell["prefix_text"] + q["question"] + cell["post_marker"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "_meta": {
            "cell_id": cell["cell_id"], "K": cell["K"], "wrap": cell["wrap"],
            "victim": victim, "shot_pool": cell["shot_pool_victim"],
            "prompt_id": q["prompt_id"], "source_index": q["source_index"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path, default=None,
                    help="cell JSON from build_prompts.py (default: the single file in prompts/cells/)")
    ap.add_argument("--queries", type=Path, default=EXP_DIR / "prompts" / "queries" / "qwen3_14b.jsonl")
    ap.add_argument("--victim-key", default=VICTIM_KEY, help="farm registry key (default: %(default)s)")
    ap.add_argument("--batch", default="distill10k", help="batch tag in the filename")
    ap.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--shards", type=int, default=1, help="contiguous shards (default: 1)")
    ap.add_argument("--only-ids", type=Path, default=None,
                    help="text file of prompt_ids to keep (one per line) — for length re-runs")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows only")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_INBOX)
    args = ap.parse_args()

    cell_path = args.cell
    if cell_path is None:
        cands = sorted((EXP_DIR / "prompts" / "cells").glob("*.json"))
        if len(cands) != 1:
            raise SystemExit(f"pass --cell; found {len(cands)} cell files")
        cell_path = cands[0]
    cell = json.loads(cell_path.read_text())
    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    if args.only_ids:
        keep = {l.strip() for l in args.only_ids.read_text().splitlines() if l.strip()}
        queries = [q for q in queries if q["prompt_id"] in keep]
    if args.limit:
        queries = queries[: args.limit]
    print(f"[assemble] cell={cell['cell_id']} prefix_tok={cell['n_prefix_tokens']} "
          f"queries={len(queries)}", flush=True)

    victim = cell["cell_id"].split("_")[2] if cell["cell_id"].startswith("distill_C_") else args.victim_key
    rows = [make_row(cell, q, max_tokens=args.max_tokens, temperature=args.temperature,
                     victim=victim) for q in queries]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate prompt_ids in queries")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    for i in range(max(1, args.shards)):
        lo, hi = i * n // args.shards, (i + 1) * n // args.shards
        name = (f"{args.victim_key}__{args.batch}.jsonl" if args.shards == 1
                else f"{args.victim_key}__{args.batch}__shard{i:02d}.jsonl")
        out = args.out_dir / name
        with out.open("w") as f:
            for r in rows[lo:hi]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[assemble] {out}  rows={hi - lo}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
