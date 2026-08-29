"""REP distillation corpus — optional top-up question sample.

The primary 10k query set (vendored ``data/openthoughts_10k/questions.jsonl.gz``,
OpenThoughts-114k ``source_index`` 20000..29999) yields ~5.4k *clean* rows on
Qwen3-14B. To reach a 10k clean corpus the published pipeline harvested a
second, disjoint sample of OpenThoughts math questions and topped up with its
clean rows (``assemble_corpus.py --topup``). This script reproduces that
sample.

Universe: ``open-thoughts/OpenThoughts-114k`` (config ``metadata``) rows with
``domain == "math"``, excluding

* ``source_index`` in the primary 10k slice (20000..29999), and
* the 500 victim test rows (``meta.dataset_row_idx`` of the vendored test set).

Output rows: ``{prompt_id, source_index, question, gold_solution, gold_boxed}``
with ``prompt_id = openthoughts_<source_index:06d>`` (same scheme as the 10k
set). ``gold_boxed`` = last ``\\boxed{}`` of ``ground_truth_solution``, falling
back to ``deepseek_solution``.

Run::

    python sample_questions.py                       # 25 000 rows, seed 42
    python sample_questions.py --n 5000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXP_DIR = _HERE.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(_HERE.parents[2] / "experiments"))

import _common as C  # noqa: E402
from corpus_lib import extract_gold_boxed  # noqa: E402

N_SAMPLE = 25_000
SAMPLE_SEED = 42
EXCLUDE_RANGE = range(20_000, 30_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--out", type=Path, default=EXP_DIR / "prompts" / "queries" / "topup_25k.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset  # lazy
    print("[sample] loading OpenThoughts-114k metadata …", flush=True)
    ot = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")

    test_idxs = {int(r["meta"]["dataset_row_idx"]) for r in C.load_ot500_test()}
    print(f"[sample] excluding {len(test_idxs)} test-500 rows + primary slice "
          f"[{EXCLUDE_RANGE.start},{EXCLUDE_RANGE.stop})", flush=True)

    domains = ot["domain"]
    candidates = [i for i, d in enumerate(domains)
                  if d == "math" and i not in EXCLUDE_RANGE and i not in test_idxs]
    print(f"[sample] candidate math rows: {len(candidates)}", flush=True)
    if len(candidates) < args.n:
        raise SystemExit(f"need {args.n}, only have {len(candidates)}")

    sampled = sorted(random.Random(args.seed).sample(candidates, args.n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_no_gold = 0
    with args.out.open("w") as f:
        for si in sampled:
            r = ot[si]
            sol_gt = r.get("ground_truth_solution") or ""
            sol_ds = r.get("deepseek_solution") or ""
            boxed = extract_gold_boxed(sol_gt) or extract_gold_boxed(sol_ds)
            n_no_gold += not boxed
            f.write(json.dumps({
                "prompt_id": f"openthoughts_{si:06d}",
                "source_index": int(si),
                "question": r.get("problem") or "",
                "gold_solution": sol_gt or sol_ds,
                "gold_boxed": boxed,
            }, ensure_ascii=False) + "\n")
    print(f"[sample] wrote {args.out}  rows={len(sampled)}  no_boxed_gold={n_no_gold}", flush=True)


if __name__ == "__main__":
    main()
