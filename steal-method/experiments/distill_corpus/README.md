# distill_corpus — the 10k REP distillation corpus (Tables 2, 5)

Builds the REP-exposed teacher corpus that every distillation row in
`downstream-finetune/` trains on: 10 000 OpenThoughts-114k math questions
attacked with the paper-default REP configuration (V3 `markdown_fence`, k=3,
shadow Qwen3-14B) against a Qwen3-14B or Qwen3-32B victim, then filtered.

| Split | Rule | Hub config (`Chia-Mu-Lab/REP-datasets`) |
|---|---|---|
| **orig** | victim output parses into `<think>…</think>` + body | `distill_q3_{14b,32b}_original` |
| **clean** | orig **and** the `\boxed{}` answer is math-verify-equivalent to gold | `distill_q3_{14b,32b}_clean` |

## Run

```bash
python build_prompts.py                                   # render V3 k=3 prefix, index the 10k questions
python assemble_inbox.py                                  # -> ../../inference-farm/inbox/qwen3-14b__distill10k.jsonl
# harvest with ../../inference-farm/ (see its README)
python assemble_corpus.py --results ../../inference-farm/result --filter original --out results/original.jsonl
python assemble_corpus.py --results ../../inference-farm/result --filter clean --gold hub --out results/clean.jsonl
```

Every script has `--help`. For the 32B victim pass `--victim-key qwen3-32b`
to `assemble_inbox.py` and `--victim qwen3_32b` to `assemble_corpus.py`.
Questions are the vendored `data/openthoughts_10k/questions.jsonl.gz`; shots
are the vendored seed-7 draw in `data/shot_pool/qwen3_14b.jsonl.gz`.

Output rows (`corpus_lib.build_corpus_row`): `question, r1, r2, answer,
completion, completion_tokens, finish_reason, structural, victim`. The Hub
configs are this schema projected to `question, r1, r2, answer, completion`.

Tests: `python -m pytest tests/test_distill_corpus.py -q` (from `public/`).
