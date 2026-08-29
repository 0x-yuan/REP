# Data

Small files are vendored here so prompt building and scoring run offline;
the training corpora live on the Hub in **`Chia-Mu-Lab/REP-datasets`**
(`load_dataset("Chia-Mu-Lab/REP-datasets", <config>)`).

## Vendored

| Path | Rows | Used by |
|---|---|---|
| `openthoughts_test_500/ri_qwen3_{14b,32b}.jsonl.gz` | 500 each | victim test set with the benign internal trace `r0` (one file per shadow model) |
| `shot_pool/qwen3_14b.jsonl.gz` | 50 | OpenThoughts shadow-demo pool (seed-7 draw; builders slice `[:k]`) |
| `shot_pool/{math500,gsm8k,jeebench}_qwen3_14b.jsonl.gz` | 50 each | off-domain demo pools for the cross-dataset table (Table 3) |
| `openthoughts_10k/questions.jsonl.gz` | 10 000 | distillation questions (`prompt_id, source_index, question`; OpenThoughts-114k 20000..29999) for `distill_corpus/` and the oracle builder |
| `qa_seed/reasoning_seed_6k.jsonl.gz` | 6 000 | QA-study training seed (`id, source, question, answer, answer_type, meta_json`); `eval/qa_eval/build_eval_sets.py --check-disjoint` |
| `benchmarks/{gsm8k,math500,jeebench}_test.jsonl.gz` | 1319 / 500 / 515 | answer scorers |

## `REP-datasets` — the training corpora

| Config | Rows | Victim | Filter | Trains (`REP-models`) |
|---|---|---|---|---|
| `distill_q3_14b_clean` | 10 000 | Qwen3-14B | structural + answer-correct | `qwen25-7b-rep-q3_14b-clean` |
| `distill_q3_14b_original` | 8 046 | Qwen3-14B | structural (no answer check) | `qwen25-7b-rep-q3_14b-original` |
| `distill_q3_32b_clean` | 10 000 | Qwen3-32B | structural + answer-correct | `qwen25-7b-rep-q3_32b-clean` |
| `distill_q3_32b_original` | 6 291 | Qwen3-32B | structural (no answer check) | `qwen25-7b-rep-q3_32b-original` |

Columns: `question, r1, r2, answer, completion`. Build recipe: `steal-method/experiments/distill_corpus/`.

Oracle, answer-only / summary control corpora, the QA-study training data and
intermediate harvests are not part of this release; see `ETHICS.md`.
