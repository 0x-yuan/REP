"""Optional Modal wrapper around ``run_qa_eval.py --model`` (one GPU, vLLM).

Mirrors the setup used for the paper's QA numbers (vLLM, bf16, greedy,
max_model_len 8192, 2048 new tokens). Mount your checkpoint volume and HF
cache as you like; nothing else in ``qa_eval`` depends on Modal.

    modal run modal_run.py --model-path Qwen/Qwen2.5-7B-Instruct --label base
    modal run modal_run.py --model-path /ckpts/<run>/checkpoint-N --label rep-ep5

Set ``QA_EVAL_GPU`` (default ``H100``) to pick the GPU type and
``QA_EVAL_CKPT_VOLUME`` to mount a Modal volume at ``/ckpts``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
GPU = os.environ.get("QA_EVAL_GPU", "H100")
CKPT_VOL = os.environ.get("QA_EVAL_CKPT_VOLUME")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.46", "vllm>=0.10", "huggingface_hub", "datasets")
    .add_local_dir(str(_HERE), "/qa_eval", copy=True)
)
app = modal.App("rep-qa-eval", image=image)
volumes = {"/ckpts": modal.Volume.from_name(CKPT_VOL)} if CKPT_VOL else {}


@app.function(gpu=GPU, timeout=60 * 60, volumes=volumes)
def score(model_path: str, label: str) -> dict:
    import subprocess
    out = Path(f"/tmp/qa/{label}")
    subprocess.run(["python", "/qa_eval/run_qa_eval.py", "--model", model_path,
                    "--eval-dir", "/qa_eval/data", "--out", str(out), "--label", label],
                   check=True)
    return json.loads((out / "qa.json").read_text())


@app.local_entrypoint()
def main(model_path: str, label: str):
    print(json.dumps(score.remote(model_path, label), indent=2))
