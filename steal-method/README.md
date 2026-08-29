# steal-method — Reasoning Exposure Prompting (REP)

REP builds a `k`-shot prefix of shadow-model (Qwen3-14B) demonstrations, wraps
each in a code- or tool-like format `T`, prepends it to the question and
elicits `M_v(REP(q)) -> (r1, r2, a)`; `r2` is the exposed trace. Paper default:
wrapper **V3 `markdown_fence`, k = 3**; shots are `random.Random(7).sample(pool, 50)[:k]`.

```
rep_core/              wrappers V0–V5 (variants.py), defender system prompt, shot pool
experiments/
  config_sweep/        wrapper × k ablation, victim Qwen3-14B      (Tables 1, 6)
  cross_dataset/       demo source OT / MATH500 / GSM8K / JEEBench (Table 3)
  cross_victim/        victims 32B / 27B / 235B / gpt-oss / Gemma-4 (Table 4)
  code_paradigm/       full code / bare cmd / no code              (Table 7)
  distill_corpus/      10k REP corpus for distillation             (Tables 2, 5)
api-backends/deepinfra/  same attack on a third-party host
inference-farm/        Modal SGLang batch farm (inbox -> result)
```

| ID | Wrapper | Reveal format |
|---|---|---|
| V0 | `baseline_plain` | plain echo |
| V1 | `shell_cat` | `$ cat reasoning_trace.txt` |
| V2 | `python_repl` | `>>> print(open('reasoning_trace.txt').read())` |
| **V3** | **`markdown_fence`** | fenced `$ cat reasoning_trace.txt` (default) |
| V4 | `jupyter_cell` | `In [1]: !cat reasoning_trace.txt` |
| V5 | `agent_tool` | `<tool_call>…</tool_call> <tool_result>…` |

## Run

```bash
python experiments/config_sweep/build_prompts.py          # -> inference-farm/inbox/qwen3-14b__*.jsonl
cd inference-farm && PROFILE=<modal-profile> MODEL=qwen3-14b bash master/run_pipeline.sh
python ../eval/trace_metrics/score.py inference-farm/result/<batch>.jsonl
```

Every builder takes `--help`. Inputs are vendored under `../data/` (test set
with `r0`, shot pools), so builders run offline. API victims (Gemma-4) use
`experiments/cross_victim/openrouter_runner.py`. Tests: `python -m pytest ../tests/test_rep_prompts.py -q`.
