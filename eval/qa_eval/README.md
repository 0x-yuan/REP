# eval/qa_eval — downstream utility on three new reasoning categories

Extends the paper's downstream-utility study to **StrategyQA** (commonsense),
**ProntoQA** (symbolic) and **HotpotQA** (multi-hop, open-book distractor).
A Qwen2.5-7B-Instruct student fine-tuned on REP-exposed traces (or answer-only
/ oracle controls) stolen from Qwen3-14B / 32B is evaluated here.
Results: `results/supp_qa_three_tasks.csv`.

```
scoring.py          \boxed{} extraction, binary EM, SQuAD EM/F1
build_eval_sets.py  rebuilds the 3 held-out eval sets (disjoint from the train seed)
run_qa_eval.py      score a directory of generations, or generate with vLLM
modal_run.py        optional Modal GPU wrapper
data/               <bench>_eval.jsonl
```

Eval sets: StrategyQA test (687), ProntoQA (300), HotpotQA distractor
validation (300); all disjoint from the training seed
(`data/qa_seed/reasoning_seed_6k.jsonl.gz`). The per-supervision training data
of the QA study is not part of this release.

## Run

```bash
uv run --with datasets python build_eval_sets.py --out data --check-disjoint
uv run --with vllm python run_qa_eval.py --model Qwen/Qwen2.5-7B-Instruct --eval-dir data --out runs/base --label base
uv run python run_qa_eval.py --generations runs/base --out runs/base --per-row     # score only
```

Output: `runs/<label>/qa.json` with per-bench `acc` (StrategyQA, ProntoQA) and `em` / `f1` (HotpotQA).

## Results (Qwen2.5-7B-Instruct student, %; base: StrategyQA 70.3, ProntoQA 99.7, HotpotQA-EM 56.7)

| Dataset | 14B Ans | 14B REP | 14B Oracle | 32B Ans | 32B REP | 32B Oracle |
|---|---:|---:|---:|---:|---:|---:|
| StrategyQA | 71.6 | 72.9 | 72.9 | 72.3 | 77.0 | 75.0 |
| ProntoQA | 100.0 | 100.0 | 99.7 | 100.0 | 100.0 | 100.0 |
| HotpotQA (EM) | 63.0 | 67.3 | 62.7 | 64.7 | 67.3 | 63.3 |

Tests: `uv run --with pytest python -m pytest eval/tests/test_qa_eval.py -q` (from `public/`).
