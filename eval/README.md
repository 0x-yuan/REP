# eval — evaluation library

Two concerns: **leakage metrics** (how much of the victim's hidden trace an
attack pushes into the visible response) and **answer scoring** for GSM8K /
MATH500 / JEEBench. Everything runs offline on CPU.

```
trace_metrics/score.py          Struct%, R01 / R02 / R12, answer-EM        (Tables 1, 6, 7)
trace_metrics/paper_metrics.py  full-trace ROUGE-1/2/L, LEN, BLEU panel     (Tables 3, 4)
scorers/                        gsm8k / math500 / jeebench / openthoughts, get_scorer(name)
benchmarks/                     per-checkpoint MATH500 / AIME / JEE / LCB protocol (pointer)
qa_eval/                        StrategyQA / ProntoQA / HotpotQA utility study
defenses/                       black-box prompt defenses vs REP
tests/
```

| Metric | Definition |
|---|---|
| **Struct%** | outputs that parse into `<think>r1</think> r2` |
| **R01** | ROUGE-L(r0, r1): attacked internal trace vs clean trace |
| **R02** | ROUGE-L(r0, r2): **primary leakage** — clean trace vs visible body |
| **R12** | ROUGE-L(r1, r2): does the model echo its own `<think>`? |
| **answer-match** | normalized exact match of the final answer vs gold |

## Run

```bash
uv run --with rouge_score python trace_metrics/score.py harvested.jsonl --victim qwen3-32b --out-dir out/
uv run --with rouge_score --with sacrebleu python trace_metrics/paper_metrics.py traces.jsonl --group-by victim_model --out-csv t4.csv
```

Input rows need `generation`; optional `reference_trace` (= r0), `gold_answer`,
`victim` (selects the per-family reassembler: `qwen3-*`, `gpt-oss-20b`,
`gemma-4-31b`, `qwen3p6-27b`). Outputs: `out/scored.jsonl` + `out/summary.json`.

Answer scoring: `get_scorer("math500").score(generation, gold)` returns
`answer_match` / `answer_match_partial` (JEEBench MCQ(multiple) partial credit);
`scorers/validate_scorer_coverage.py` round-trips every gold answer (1319 / 500 / 515 PASS).

Tests: `uv run --with pytest --with rouge_score --with math-verify python -m pytest eval/tests -q` (from `public/`).
