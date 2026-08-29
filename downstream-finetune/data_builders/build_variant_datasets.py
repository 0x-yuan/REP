"""Answer-only / summary+answer / ideal-think+answer training datasets
(Table 2 control rows; ``configs/{answer_only,summary}_{14b,32b}.py``).

Takes an oracle-corpus source (``build_oracle_corpus.py assemble`` output,
with the ``summary`` column attached by ``build_summary_dataset.py merge``
for ``summary_answer``) and emits one training dataset per ``--kind``:

| kind                 | assistant target                                   |
|----------------------|----------------------------------------------------|
| ``answer_only``      | ``**Final Answer**\\n\\boxed{gold}``                 |
| ``summary_answer``   | ``row.summary`` (boxed tail appended if missing)   |
| ``ideal_think_answer``| ``row.r1`` (boxed tail appended if missing)       |

Every output row carries the canonical-loader columns
(``prompt_id, question, completion, structural, victim, …``) with
``completion = "<think>\\n</think>\\n\\n{target}"`` so the engine's
post-``</think>`` extraction returns ``target`` verbatim, plus preview /
provenance columns (``target, summary, original_completion, original_r1,
original_r2, variant, teacher``). Non-structural or empty rows are dropped.

Run::

    python build_variant_datasets.py --source oracle_q3_14b_with_summary.jsonl \\
        --kind answer_only --victim qwen3_14b --out ans_q3_14b.jsonl
    python build_variant_datasets.py --source <org>/<oracle-repo> --kind summary_answer \\
        --victim qwen3_32b --out sum_q3_32b.jsonl --push-to <org>/<repo>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import _builders as B  # noqa: E402


def load_source(src: str) -> list[dict]:
    if Path(src).exists():
        rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
    else:
        from datasets import load_dataset  # lazy
        print(f"[variant] loading {src} …", flush=True)
        rows = [dict(r) for r in load_dataset(src, split="train")]
    print(f"[variant] source rows={len(rows)}", flush=True)
    return rows


def build(rows: list[dict], kind: str, victim: str) -> tuple[list[dict], dict]:
    out, st = [], {"struct_drop": 0, "empty_q": 0, "no_target": 0, "kept": 0}
    for r in rows:
        row, why = B.variant_row(r, kind, victim=victim)
        if row is None:
            st[why] += 1
            continue
        st["kept"] += 1
        out.append(row)
    return out, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="oracle corpus: local JSONL or Hub id")
    ap.add_argument("--kind", choices=B.VARIANT_KINDS, required=True)
    ap.add_argument("--victim", default="qwen3_14b", help="teacher label (qwen3_14b | qwen3_32b)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--push-to", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    rows, st = build(load_source(args.source), args.kind, args.victim)
    print(f"[variant:{args.kind}] {st}", flush=True)
    if not rows:
        raise SystemExit("built 0 rows — abort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[variant:{args.kind}] wrote {args.out} ({len(rows)} rows)", flush=True)
    if args.push_to:
        from datasets import Dataset  # lazy
        Dataset.from_list(rows).push_to_hub(args.push_to, private=args.private)
        print(f"[variant:{args.kind}] pushed -> {args.push_to}", flush=True)


if __name__ == "__main__":
    main()
