"""Trace-summary teacher data for ``configs/summary_{14b,32b}.py`` (step 1 of 2).

The *summary* control row replaces the oracle internal trace with a short
solution written by ``Qwen2.5-7B-Instruct``: for every oracle-corpus row the
summariser gets the question and the trace ``r1`` under the fixed prompt in
``_builders.SUMMARY_SYSTEM`` / ``SUMMARY_USER_TEMPLATE`` and must end with the
same ``\\boxed{}``. Two sub-commands:

``prompts`` — oracle corpus (``build_oracle_corpus.py assemble`` output or its
             Hub id) -> one inbox file for the farm's ``qwen2p5-7b-instruct``
             key: ``{"id": prompt_id, "messages": [system, user],
             "max_tokens": 8192, "temperature": 0.7, "top_p": 0.8, "seed": 7,
             "enable_thinking": false}``.
``merge``   — farm outputs -> the oracle corpus with four extra columns
             ``summary, summary_completion_tokens, summary_finish_reason,
             summary_model`` (left-join on prompt_id; missing summaries become
             ``""`` / 0). Pre-existing ``summary*`` columns are replaced.

Then wrap the summaries into training rows with
``build_variant_datasets.py --kind summary_answer``.

Run::

    python build_summary_dataset.py prompts --source oracle_q3_14b.jsonl --batch sum_q3_14b
    python build_summary_dataset.py merge --source oracle_q3_14b.jsonl \\
        --results ../../steal-method/inference-farm/result/qwen2p5-7b-instruct__sum_q3_14b.jsonl \\
        --out oracle_q3_14b_with_summary.jsonl [--push-to <org>/<repo>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import _builders as B  # noqa: E402

MODEL_KEY = "qwen2p5-7b-instruct"
DEFAULT_INBOX = _HERE.parents[2] / "steal-method" / "inference-farm" / "inbox"


def load_source(src: str) -> list[dict]:
    if Path(src).exists():
        rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
    else:
        from datasets import load_dataset  # lazy
        print(f"[summary] loading {src} …", flush=True)
        rows = [dict(r) for r in load_dataset(src, split="train")]
    print(f"[summary] source rows={len(rows)}", flush=True)
    return rows


def prompt_rows(source: list[dict], *, max_tokens: int, temperature: float,
                top_p: float, seed: int) -> list[dict]:
    rows, seen = [], set()
    for ex in source:
        pid = ex.get("prompt_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        q = (ex.get("question") or "").strip()
        r1 = (ex.get("r1") or "").strip()
        if not q or not r1:
            continue
        rows.append({
            "id": pid,
            "messages": B.summary_messages(q, r1),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
            "enable_thinking": False,
            "add_generation_prompt": True,
        })
    return rows


def cmd_prompts(args) -> None:
    rows = prompt_rows(load_source(args.source), max_tokens=args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.model_key}__{args.batch}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[summary] wrote {out} ({len(rows)} rows)", flush=True)


def load_summaries(paths: list[Path], model_key: str) -> tuple[dict[str, dict], dict]:
    out: dict[str, dict] = {}
    st = {"seen": 0, "dupe": 0, "empty": 0, "truncated_length": 0}
    files: list[Path] = []
    for p in paths:
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    for fp in files:
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("id")
            if not pid:
                continue
            st["seen"] += 1
            s = B.extract_summary(rec, model_key)
            if s is None:
                st["empty"] += 1
                continue
            st["truncated_length"] += s["summary_finish_reason"] == "length"
            if pid in out:
                st["dupe"] += 1
                continue
            out[pid] = s
    st["kept"] = len(out)
    return out, st


def merge_rows(base: list[dict], summaries: dict[str, dict]) -> list[dict]:
    merged = []
    for r in base:
        r = {k: v for k, v in r.items() if k not in B.SUMMARY_COLUMNS}
        s = summaries.get(r.get("prompt_id"), {})
        r["summary"] = s.get("summary") or ""
        r["summary_completion_tokens"] = int(s.get("summary_completion_tokens") or 0)
        r["summary_finish_reason"] = s.get("summary_finish_reason") or ""
        r["summary_model"] = s.get("summary_model") or ""
        merged.append(r)
    return merged


def cmd_merge(args) -> None:
    base = load_source(args.source)
    summaries, st = load_summaries(args.results, args.model_key)
    print(f"[summary] results: {st}", flush=True)
    if not summaries:
        raise SystemExit("no summaries extracted")
    merged = merge_rows(base, summaries)
    n_filled = sum(1 for r in merged if r["summary"])
    print(f"[summary] merged: filled={n_filled}/{len(merged)} missing={len(merged) - n_filled}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[summary] wrote {args.out}", flush=True)
    if args.push_to:
        from datasets import Dataset  # lazy
        Dataset.from_list(merged).push_to_hub(args.push_to, private=args.private)
        print(f"[summary] pushed -> {args.push_to}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="oracle corpus: local JSONL or Hub id")
    ap.add_argument("--model-key", default=MODEL_KEY, help="summariser farm key (default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts", help="write the summariser inbox file")
    p.add_argument("--batch", default="sum_q3_14b")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_INBOX)
    p.set_defaults(fn=cmd_prompts)

    m = sub.add_parser("merge", help="attach summaries to the oracle corpus")
    m.add_argument("--results", type=Path, nargs="+", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--push-to", default=None)
    m.add_argument("--private", action="store_true")
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
