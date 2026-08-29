# data_builders — teacher data for the control / oracle rows

Offline scripts that produce the datasets consumed by
`configs/{oracle,summary,answer_only}_{14b,32b}.py`. All derive from the same
10k questions as the REP corpus (`steal-method/experiments/distill_corpus/`)
and a Qwen3-14B / Qwen3-32B teacher.

| Config row | Builder | Target the student sees |
|---|---|---|
| `oracle_*` | `build_oracle_corpus.py` | teacher's own internal trace (no attack) + `\boxed{}` |
| `summary_*` | `build_summary_dataset.py` → `build_variant_datasets.py --kind summary_answer` | Qwen2.5-7B-Instruct compression of that trace + `\boxed{}` |
| `answer_only_*` | `build_variant_datasets.py --kind answer_only` | `\boxed{gold}` only |

## Run

```bash
python build_oracle_corpus.py --victim qwen3_14b prompts                     # -> inference-farm inbox; harvest
python build_oracle_corpus.py --victim qwen3_14b assemble --results <farm result dir> --out oracle_q3_14b.jsonl
python build_variant_datasets.py --kind answer_only --source oracle_q3_14b.jsonl --out ans_q3_14b.jsonl
```

Every script has `--help`; `--source` accepts a local JSONL(.gz) (default: the vendored 10k questions),
`--push-to` uploads. Row contracts live in `_builders.py`; all outputs are
raw-harvest-shaped so the engine's loader (`../engine/_common/dataset_prep.py`) reads
them unchanged.

Tests: `uv run --with pytest --with math-verify python -m pytest downstream-finetune/tests/test_data_builders.py -q` (from `public/`).
