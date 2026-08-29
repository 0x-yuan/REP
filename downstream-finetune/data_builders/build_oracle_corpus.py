"""Oracle (no-attack) teacher corpus for ``configs/oracle_{14b,32b}.py``.

The oracle reference row of Table 2 trains on the victim's OWN internal
``<think>`` trace, collected without any REP prompt: each of the 10k clean-
corpus questions is sent as a bare user turn (no system prompt, no
demonstrations) with Qwen3 thinking enabled, and the returned trace becomes
the training target. Two sub-commands:

``prompts``  — clean corpus questions -> inference-farm inbox rows
              ``{"id": prompt_id, "messages": [{"role": "user", "content": q}],
              "max_tokens": 20000, "temperature": 0.7, "top_p": 0.8, "seed": 7,
              "enable_thinking": true}`` (Qwen3 thinking-mode sampling defaults),
              sharded into ``--shards`` files.
``assemble`` — farm outputs -> oracle rows (``_builders.oracle_row``): r1 =
              inside ``<think>``, answer = last ``\\boxed{}`` after it,
              ``completion`` re-synthesised as
              ``<think>r1</think>\\n\\nr1\\n\\n**Final Answer**\\n\\boxed{ans}``.
              Rows without a trace or a boxed answer are dropped (all kept rows
              are ``structural=True``); ``answer_match`` vs ``gold_boxed`` is
              recorded but NOT filtered on.

Source questions: ``--source`` = local JSONL(.gz) with
``{prompt_id, source_index, question[, gold_boxed]}``. Default = the vendored
``data/openthoughts_10k/questions.jsonl.gz``; the clean corpus from
``steal-method/experiments/distill_corpus/`` also works.

Run::

    python build_oracle_corpus.py prompts --shards 10
    python build_oracle_corpus.py assemble --results ../../steal-method/inference-farm/result \\
        --out oracle_q3_14b.jsonl [--push-to <org>/<repo>]   # push is off by default
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import _builders as B  # noqa: E402

DEFAULT_SOURCE = _HERE.parents[2] / "data" / "openthoughts_10k" / "questions.jsonl.gz"
MODEL_KEY = "qwen3-14b"
BATCH_PREFIX = "oracle"
DEFAULT_INBOX = _HERE.parents[2] / "steal-method" / "inference-farm" / "inbox"


def load_source(src: str) -> list[dict]:
    p = Path(src)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"[oracle] source rows={len(rows)}", flush=True)
    return rows


def prompt_rows(source: list[dict], *, max_tokens: int, temperature: float,
                top_p: float, seed: int) -> list[dict]:
    rows, seen = [], set()
    for i, ex in enumerate(source):
        q = (ex.get("question") or "").strip()
        if not q:
            continue
        pid = ex.get("prompt_id") or f"openthoughts_{i:06d}"
        if pid in seen:
            continue
        seen.add(pid)
        rows.append({
            "id": pid,
            "messages": [{"role": "user", "content": q}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
            "enable_thinking": True,
            "add_generation_prompt": True,
        })
    return rows


def cmd_prompts(args) -> None:
    rows = prompt_rows(load_source(args.source), max_tokens=args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, args.shards)
    chunks: list[list[dict]] = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        chunks[i % n].append(r)
    for idx, chunk in enumerate(chunks):
        out = args.out_dir / f"{args.model_key}__{args.batch}_part-{idx:02d}-of-{n:02d}.jsonl"
        with out.open("w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[oracle] wrote {out.name} ({len(chunk)} rows)", flush=True)
    print(f"[oracle] {len(rows)} prompt rows across {n} shard(s) -> {args.out_dir}", flush=True)


def iter_results(paths: list[Path], pattern: str):
    files: list[Path] = []
    for p in paths:
        files += sorted(p.rglob(pattern)) if p.is_dir() else [p]
    for fp in files:
        with fp.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass


def cmd_assemble(args) -> None:
    lookup = {}
    for ex in load_source(args.source):
        if ex.get("prompt_id"):
            lookup[ex["prompt_id"]] = {"source_index": ex.get("source_index"),
                                       "question": ex.get("question"),
                                       "gold_boxed": ex.get("gold_boxed")}
    stats = {k: 0 for k in ("considered", "row_error", "no_think", "no_boxed",
                            "missing_in_source", "truncated", "kept",
                            "answer_match_true", "answer_match_false")}
    out_rows, seen = [], set()
    for rec in iter_results(args.results, f"*{args.batch}*.jsonl"):
        stats["considered"] += 1
        pid = rec.get("id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        meta = lookup.get(pid)
        if meta is None:
            stats["missing_in_source"] += 1
            continue
        row, why = B.oracle_row(rec, meta, victim=args.victim)
        if row is None:
            stats[why] += 1
            continue
        stats["truncated"] += row["finish_reason"] == "length"
        stats["kept"] += 1
        stats["answer_match_true" if row["answer_match"] else "answer_match_false"] += 1
        out_rows.append(row)
    out_rows.sort(key=lambda r: r["prompt_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[oracle] wrote {args.out} ({len(out_rows)} rows)\n{json.dumps(stats, indent=2)}", flush=True)
    if args.push_to and out_rows:
        from datasets import Dataset  # lazy
        Dataset.from_list(out_rows).push_to_hub(args.push_to, private=args.private)
        print(f"[oracle] pushed -> {args.push_to}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                    help="local questions JSONL(.gz) (default: vendored 10k questions)")
    ap.add_argument("--victim", default="qwen3_14b")
    ap.add_argument("--model-key", default=MODEL_KEY, help="farm registry key (default: %(default)s)")
    ap.add_argument("--batch", default=BATCH_PREFIX, help="batch tag in file names")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts", help="write no-attack inbox shards")
    p.add_argument("--shards", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_INBOX)
    p.set_defaults(fn=cmd_prompts)

    a = sub.add_parser("assemble", help="farm outputs -> oracle corpus")
    a.add_argument("--results", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--push-to", default=None)
    a.add_argument("--private", action="store_true")
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
