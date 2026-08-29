# downstream-finetune

Distillation (student fine-tuning) code for the paper's Tables 2 and 5. Every
row is the same engine driven by a different `configs/<row>.py` and teacher
dataset: full-parameter SFT of Qwen2.5-Instruct students (7B for the paper
rows; 14B / 32B for the student-scale rows) on 10k teacher traces, one checkpoint
per epoch, each evaluated on MATH500 / AIME24 / AIME25 / JEE-Math / LCB, best
epoch selected by Δ-sum vs the base model. Training runs on Modal GPUs.

```
engine/          canonical distillation engine (train.py, eval_multi.py, launch.sh, _common/, s1/)
configs/         one config per paper row (below)
data_builders/   teacher data for the oracle / summary / answer-only rows;  tests/  offline schema tests
```

| Config | Supervision | Teacher dataset |
|---|---|---|
| `rep_{14b,32b}_{clean,orig}` | REP exposed trace + answer | `Chia-Mu-Lab/REP-datasets` config `distill_q3_{14b,32b}_{clean,original}` |
| `oracle_{14b,32b}` | victim internal trace + answer | build with `data_builders/build_oracle_corpus.py`, push, set `SOURCE_REPO` |
| `answer_only_{14b,32b}` | answer only (control) | built by `data_builders/` from the oracle corpora |
| `summary_{14b,32b}` | trace summary + answer (control) | built by `data_builders/` from the oracle corpora |
| `rep_q3_14b_to_qwen25_{14b,32b}` | REP trace, larger students | `REP-datasets` `distill_q3_14b_clean` flattened to `{problem, solution}` (see config comment) |

Results: `results/table{2,5}_*.csv`, `results/supp_student_scale.csv`; released students: `models/README.md`.

## Run one row

```bash
cd downstream-finetune/engine && DISTILL_CONFIG=../configs/<row>.py ./launch.sh
```

`launch.sh` reads `DISTILL_CONFIG` (via `engine/_config_loader.py`; set
`MODAL_PROFILE` to the config's `PROFILE`), deploys the Modal apps, spawns
training, and starts the per-checkpoint eval orchestrator; results land in `final_results.json` on the
run's Modal volume. Requires the Modal CLI and a `huggingface` Modal secret
holding `HF_TOKEN`. Evaluation protocol: `../eval/benchmarks/README.md`. The
TIA rows of Table 2 come from the external Trace Inversion Attack pipeline
(arXiv:2603.07267), not this engine.

Tests: `uv run --with pytest --with math-verify python -m pytest downstream-finetune/tests -q` (from `public/`).
