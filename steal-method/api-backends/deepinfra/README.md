# REP on a third-party host (DeepInfra)

Runs the paper's V3 / k=3 attack (Table 1 / Table 4 cells, Qwen3-14B and
Qwen3-32B) against DeepInfra's OpenAI-compatible `/completions` endpoint with
the same rendered prompt the SGLang farm receives, then scores it with the
same ROUGE-L code. Leakage reproduces within ~0.01 of our farm; the only gap
is DeepInfra's 40960-token context, which forces a shorter demo draw and some
output truncation.

Setup: `DEEPINFRA_API_KEY` in the environment, `uv sync --extra api-backends`.

```bash
uv run python build_prompts.py --out prompts/qwen3-14b.jsonl
uv run python make_ref.py --config ri_qwen3_14b --out ref/ri_qwen3_14b.json
uv run python run_deepinfra.py --model Qwen/Qwen3-14B --prompts prompts/qwen3-14b.jsonl --out outputs/deepinfra_14b.jsonl
uv run python score_deepinfra.py --outputs outputs/deepinfra_14b.jsonl --ref ref/ri_qwen3_14b.json --save scored/di_14b.csv
```

`rescore_ours.py` re-scores our-side (SGLang) per-row tables with the identical code;
`master_table.py --ci` builds the side-by-side table with a paired bootstrap
CI. Results: `results/supp_deepinfra_fidelity.csv`.

| cell (OT-500, V3-K3, both-structural rows) | our farm | DeepInfra |
|---|---|---|
| Qwen3-14B fidelity | 0.333 | 0.330 |
| Qwen3-32B fidelity | 0.295 | 0.287 |

```
build_prompts.py    V3-K3 render through the Qwen3 chat template, per-row max_tokens
make_ref.py         reference hidden traces {idx: {ri, gold, answer, question}}
run_deepinfra.py    greedy runner, checkpoint / resume / retry
fast_score.py       ROUGE-L(r2, ri) + structural split
score_deepinfra.py  score a run, optional paired comparison vs our side
rescore_ours.py     re-score our-side rows with the same code
master_table.py     side-by-side table + bootstrap CI
```
