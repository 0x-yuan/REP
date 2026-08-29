# eval/defenses — black-box prompt defenses vs REP

Can published black-box, prompt-level defenses (written to protect a system
prompt) be retargeted to protect the hidden `<think>` trace and stop REP?
Setup: REP V3-K3 on OpenThoughts-500, victims Qwen3-14B / 32B on DeepInfra,
fidelity = R02 = ROUGE-L(r0, r2); defenses are generic and non-adaptive.
Results: `results/supp_defenses.csv`.

```
build_agarwal.py     Agarwal et al. prompt-hardening stack -> defended prompt rows
run_victim.py        greedy completions on an OpenAI-compatible endpoint
prob_gate.py         ProB proxy-repeat gate (Llama-3.1-8B proxy)
kad_gate.py          Known-Answer Detection canary (Llama-3.3-70B detector)
ppl_gate.py          GPT-2 windowed perplexity filter
score_defense.py     score outputs, paired analysis, gate -> effective fidelity
```

| Defense | Type | Outcome |
|---|---|---|
| ProB (EMNLP'25) | repeat gate | unusable — blocks every long prompt (100% long-benign FPR) |
| Agarwal et al. (arXiv 2404.16251) | prompt hardening | fails — fidelity unchanged (|Δ| ≤ 0.01), structure up |
| Known-Answer Detection (USENIX'24) | canary detection | works — 100% detect, 0% FPR |
| Perplexity filter | detection | fails — 1% of REP prompts flagged |

## Run

```bash
export DEEPINFRA_API_KEY=...
REP=../../steal-method/inference-farm/inbox/qwen3-14b__ot_V3_K3.jsonl
REF=../../data/openthoughts_test_500/ri_qwen3_14b.jsonl.gz
uv run python run_victim.py --model Qwen/Qwen3-14B --prompts $REP --out outputs/nodef_14b.jsonl
uv run --with transformers python build_agarwal.py --prompts $REP --out prompts/defB.jsonl
uv run python run_victim.py --model Qwen/Qwen3-14B --prompts prompts/defB.jsonl --out outputs/defB_14b.jsonl
uv run --with rouge_score python score_defense.py score --outputs outputs/defB_14b.jsonl --ref $REF --victim qwen3-14b --save scored/defB_14b.csv
```

Gates (`prob_gate.py`, `kad_gate.py`, `ppl_gate.py`) take `--prompts` / `--out` (`--mode benign_long`
for the false-positive control); feed their output to `score_defense.py gate`. All scripts have `--help`.

Tests: `uv run --with pytest --with rouge_score python -m pytest eval/tests/test_defenses.py -q` (from `public/`).
