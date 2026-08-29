# Per-checkpoint benchmark protocol

Every distilled student checkpoint is evaluated on five benchmarks; the
reported table cell is the checkpoint with the best **Δ-sum** (sum of
per-benchmark deltas vs the untuned Qwen2.5-7B-Instruct base scored under the
same protocol).

| Benchmark | Dataset | n / problem | T | max_new_tokens | Metric |
|---|---|---|---|---|---|
| MATH500 | `HuggingFaceH4/MATH-500` (500) | 3 | 0.5 | 32 768 | accuracy, `math-verify` on last `\boxed{}` |
| AIME24 | `simplescaling/aime24_nofigures` (30) | 3 | 0.5 | 32 768 | accuracy |
| AIME25 | `TIGER-Lab/AIME25` (30) | 3 | 0.5 | 32 768 | accuracy |
| JEE-Math | `daman1209arora/jeebench`, `subject == "math"` (236) | 6 | 0.5 | 32 768 | strict + partial-credit `answer_match` |
| LCB | LiveCodeBench `release_v5`, 2024-08-01 … 2025-02-01 (167) | 3 | 0.5 | 32 768 | pass@1 |

Sampling uses vLLM with a fixed seed. The runner lives with the training
engine (`downstream-finetune/engine/_common/multibench_runner.py`) because it
shares its Modal image and orchestration; this folder holds no code. It
reuses the `math-verify` backend of `../scorers/math500.py` and the
partial-credit logic of `../scorers/jeebench.py`.
