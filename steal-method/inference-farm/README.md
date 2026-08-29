# inference-farm

Modal-hosted **SGLang** batch-inference farm for the open-weight victims.
Drop `<model>__<batch>.jsonl` files into `inbox/`; the master deploys one
SGLang app per model, streams the rows through with checkpoint + resume, and
writes `result/<model>__<batch>.jsonl`. Every experiment builder in
`../experiments/` targets this inbox.

```
inbox/        input batches            (builders write here)
result/       output batches           (one per inbox file)
processed/    inbox files after success
slave/        Modal app: SGLang server (app.py), per-model ServerArgs (registry.py), weight prefetch
master/       run_pipeline.sh (deploy + run), queue_runner.py, multi_replica_runner.py, queue_cli.py
tests/        smoke fixtures + unit tests
```

## Run

```bash
cp experiment.env.example experiment.env        # set EXP_ID to namespace Modal resources
MODEL_KEY=qwen3-14b MODAL_PROFILE=<profile> uv run modal run slave/prefetch.py   # once per model
PROFILE=<profile> MODEL=qwen3-14b bash master/run_pipeline.sh
```

The runner exits when the inbox is drained; `WATCH=1` keeps it alive,
`REPLICAS=N` fans one model out over N replica apps with work-stealing, and
`SHARD_FILES_MIN_ROWS=N` pre-shards large inbox files. `master/queue_cli.py`
lists / adds / removes queued files and shows live progress.

## Formats

Input rows (`slave/batch_format.py`): `id` plus `prompt` or `messages`;
optional `max_tokens`, `temperature`, `top_p`, `stop`, `n`, `seed`,
`enable_thinking`. Output rows:

```json
{"id": "...", "model": "qwen3-14b", "prompt_tokens": 15352,
 "outputs": [{"text": "<think>...</think>...", "finish_reason": "stop", "completion_tokens": 1197}]}
```

## Models

| model_key | hf_id | GPU | TP | context |
|---|---|---|---|---|
| qwen3-1p7b / 4b / 8b / 14b | Qwen/Qwen3-{1.7B,4B,8B,14B} | H200 | 1 | 131072 (YaRN) |
| qwen3-32b | Qwen/Qwen3-32B | H200×4 | 4 | 131072 (YaRN) |

Per-model knobs (`default_max_tokens`, `context_length`, GPU, …) can be
overridden in `experiment.toml` without editing `slave/registry.py`.
Copying this folder per experiment with a distinct `EXP_ID` lets several
farms run side by side; the HF weight cache volume is always shared.

Tests: `uv run --with pytest python -m pytest tests -q`.
