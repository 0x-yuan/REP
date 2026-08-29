# Released checkpoints

All released students live in one Hub model repo, **`Chia-Mu-Lab/REP-models`**,
one sub-folder per final model reported in the paper. Only checkpoints that
beat their base model on MATH500 (n=3, T=0.5) are released. Full numbers:
`results/table2_distill_main.csv`, `results/supp_student_scale.csv`.

## Students distilled from REP-exposed traces

| Student (base) | Victim / trace source | `REP-datasets` config | `REP-models` sub-folder | MATH500 | Base |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Qwen3-14B, answer-clean | `distill_q3_14b_clean` | `qwen25-7b-rep-q3_14b-clean` | **75.8** | 71.0 |
| Qwen2.5-7B-Instruct | Qwen3-14B, structural | `distill_q3_14b_original` | `qwen25-7b-rep-q3_14b-original` | 72.4 | 71.0 |
| Qwen2.5-7B-Instruct | Qwen3-32B, answer-clean | `distill_q3_32b_clean` | `qwen25-7b-rep-q3_32b-clean` | 72.8 | 71.0 |
| Qwen2.5-7B-Instruct | Qwen3-32B, structural | `distill_q3_32b_original` | `qwen25-7b-rep-q3_32b-original` | 73.9 | 71.0 |

## Reference (oracle) students — trained on the victim's true internal trace

| Student | Trace source | `REP-models` sub-folder | MATH500 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | Qwen3-14B internal | `qwen25-7b-oracle-q3_14b` | 70.3 |
| Qwen2.5-7B-Instruct | Qwen3-32B internal | `qwen25-7b-oracle-q3_32b` | 70.0 |

Not part of this release: the answer-only / summary control students (numbers
in `results/`) and the Qwen2.5-14B / 32B student-scale runs (re-train with
`downstream-finetune/configs/rep_q3_14b_to_qwen25_{14b,32b}.py`).

## Use

```python
AutoModelForCausalLM.from_pretrained("Chia-Mu-Lab/REP-models", subfolder="qwen25-7b-rep-q3_14b-clean")
```

```bash
cd downstream-finetune/engine && DISTILL_CONFIG=../configs/rep_14b_clean.py \
    modal run eval_multi.py::score_one --hub-repo Chia-Mu-Lab/REP-models --subfolder qwen25-7b-rep-q3_14b-clean --ckpt-label rep_14b_clean
```

`score_one` downloads the sub-folder in the container and scores it on a Modal
B200 (requires the `huggingface` Modal secret); results land on the config's
results volume.

Protocol (benchmarks, n, T, max_new_tokens): `eval/benchmarks/README.md`.
