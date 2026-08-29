"""Round-trip coverage check for the dataset scorers.

For every row in each test split:
1.  canonical_gold = scorer.extract_gold(gold_text)
        — must be non-None on every row.

2.  synthetic_gen  = f"The final answer is \\boxed{{<canonical_gold>}}."
    result        = scorer.score(synthetic_gen, gold_text, meta)
        — must yield answer_match == 1.0 on every row.

Step 2 simulates a "perfectly answering" model and checks that our extractor
+ equivalence backend can parse the gold token back out of a realistic
generation envelope. This catches both gold-format gaps (step 1) and
extractor / equivalence gaps (step 2).

This script does NOT call any LLM and does NOT trust the existing GSM8K
extractor as the source of truth — the gold token itself is the reference.

Run from `public/eval/`:

    python scorers/validate_scorer_coverage.py

Expected result:  gsm8k 1319 / math500 500 / jeebench 515  all PASS.
Downloads the three HF test splits (gsm8k / MATH-500 / jeebench) on first run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# This script lives in public/eval/scorers/; add public/eval/ to sys.path so the
# sibling `scorers` package is importable as `from scorers import get_scorer`.
EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from datasets import load_dataset  # noqa: E402

from scorers import get_scorer  # noqa: E402


DATASETS = {
    "gsm8k": {
        # canonical namespaced id (legacy bare "gsm8k" is rejected by current huggingface_hub)
        "hf": ("openai/gsm8k", "main"),
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "meta_fields": [],
    },
    "math500": {
        "hf": ("HuggingFaceH4/MATH-500", None),
        "split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "meta_fields": ["subject", "level"],
    },
    "jeebench": {
        "hf": ("daman1209arora/jeebench", None),
        "split": "test",
        "question_field": "question",
        "answer_field": "gold",
        "meta_fields": ["type", "subject", "description", "index"],
    },
}


# Vendored benchmark splits (public/data/benchmarks/) — prefer these so the
# coverage gate runs fully offline; the Hub id in DATASETS is the fallback.
_LOCAL_BENCH = Path(__file__).resolve().parents[2] / "data" / "benchmarks"
_LOCAL_FILE = {
    "gsm8k": "gsm8k_test.jsonl.gz",
    "math500": "math500_test.jsonl.gz",
    "jeebench": "jeebench_test.jsonl.gz",
}


def _iter_rows(name: str, limit: int | None) -> Iterable[dict[str, Any]]:
    cfg = DATASETS[name]
    local = _LOCAL_BENCH / _LOCAL_FILE[name]
    if local.exists():
        rows: Iterable[dict[str, Any]] = _iter_local_gz(local, limit)
    else:
        hf_name, hf_config = cfg["hf"]
        ds = load_dataset(hf_name, hf_config, split=cfg["split"])
        rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    for row in rows:
        yield row


def _iter_local_gz(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def validate(name: str, limit: int | None) -> dict[str, Any]:
    cfg = DATASETS[name]
    scorer = get_scorer(name)
    n_total = 0
    n_extract_gold_none = 0
    n_self_score_zero = 0
    type_counter: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for row in _iter_rows(name, limit):
        n_total += 1
        gold_text = row[cfg["answer_field"]]
        meta = {k: row[k] for k in cfg["meta_fields"] if k in row}
        if name == "jeebench":
            type_counter[meta.get("type", "?")] += 1

        gold_token = (
            scorer.extract_gold(gold_text, meta=meta)
            if name == "jeebench"
            else scorer.extract_gold(gold_text)
        )
        if gold_token is None:
            n_extract_gold_none += 1
            if len(failures) < 10:
                failures.append({
                    "kind": "extract_gold_none",
                    "row_index": n_total - 1,
                    "gold_text": str(gold_text)[:160],
                    "meta": meta,
                })
            continue

        synthetic_gen = f"After working through the problem, the final answer is \\boxed{{{gold_token}}}."
        result = scorer.score(synthetic_gen, gold_text, meta=meta)
        if result.answer_match != 1.0:
            n_self_score_zero += 1
            if len(failures) < 10:
                failures.append({
                    "kind": "self_score_not_one",
                    "row_index": n_total - 1,
                    "gold_text": str(gold_text)[:160],
                    "synthetic_gen": synthetic_gen[:160],
                    "extracted_pred": result.extracted_pred,
                    "extracted_gold": result.extracted_gold,
                    "answer_match": result.answer_match,
                    "answer_match_partial": result.answer_match_partial,
                    "meta": meta,
                })

    return {
        "dataset": name,
        "total": n_total,
        "extract_gold_none": n_extract_gold_none,
        "self_score_zero": n_self_score_zero,
        "type_distribution": dict(type_counter) if type_counter else None,
        "failure_examples": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate scorer coverage on test splits.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on rows per dataset (debugging).")
    args = parser.parse_args()

    overall_ok = True
    for name in args.datasets:
        report = validate(name, args.limit)
        ok = report["extract_gold_none"] == 0 and report["self_score_zero"] == 0
        overall_ok = overall_ok and ok

        print(f"\n=== {name} ===")
        print(f"  total rows:           {report['total']}")
        print(f"  extract_gold None:    {report['extract_gold_none']}")
        print(f"  self_score < 1.0:     {report['self_score_zero']}")
        if report["type_distribution"]:
            print(f"  type distribution:    {report['type_distribution']}")
        if report["failure_examples"]:
            print("  first failures:")
            for fail in report["failure_examples"]:
                print(f"    - {fail}")
        print(f"  status:               {'PASS' if ok else 'FAIL'}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
