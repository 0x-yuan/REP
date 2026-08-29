"""Modal app: multi-benchmark eval (AIME24 + AIME25 + MATH500 + JEE-math + LCB)
on B200:1 for one ckpt of one experiment.

Each .spawn() runs the configured `EVAL_BENCH_SUBSET` inside its own container.
Math benches share ONE vLLM boot via `_common/multibench_runner.run_all`; LCB
boots its own vLLM via `lcb_runner`.

Deploy (one base + N replicas in lockstep — see deploy_eval_pool.sh):
    modal deploy eval_multi.py
    EVAL_APP_SUFFIX=-r1 modal deploy eval_multi.py
    ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _config_loader import cfg  # noqa: E402

# Blackwell B200 (SM 100) requires torch built with CUDA 12.8.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl", "ca-certificates")
    .pip_install(
        "transformers==4.46.3",
        "datasets==3.2.0",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "tqdm",
        "math-verify[antlr4_13_2]",
        "openai>=1.40",
        "antlr4-python3-runtime==4.13.2",
    )
    .run_commands(
        "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
        "--index-url https://download.pytorch.org/whl/cu128",
        "pip install vllm==0.10.0",
    )
)

# The lm-evaluation-harness is ONLY needed for the aime24/aime25/gpqa lm-eval
# block (run_lm_eval_block). math500 / jeebench / lcb do not use it. Vendor it
# into `s1/eval/lm-evaluation-harness` to enable those benches; if it is absent
# the image still builds and math500-only evals run fine.
_LM_EVAL_HARNESS = HERE / "s1" / "eval" / "lm-evaluation-harness"
if _LM_EVAL_HARNESS.exists():
    image = (
        image.add_local_dir(str(_LM_EVAL_HARNESS), "/workspace/lm-evaluation-harness", copy=True)
        .run_commands("cd /workspace/lm-evaluation-harness && pip install -e .[vllm,math]")
    )

image = (
    image.run_commands(
        "mkdir -p /workspace",
        "cd /workspace && git clone --depth=1 https://github.com/LiveCodeBench/LiveCodeBench.git",
        "cd /workspace/LiveCodeBench && pip install -e .",
    )
    .add_local_dir(
        str(HERE / "_common"),
        "/workspace/_common",
        copy=True,
    )
    .add_local_file(str(HERE / "config.py"), "/root/config.py", copy=True)
    .add_local_file(str(HERE / "_config_loader.py"), "/root/_config_loader.py", copy=True)
)

# Per-ckpt eval is parallelized by deploying N replicas of THIS app under
# distinct suffix names (-r1, -r2, ...). The base name (cfg.EVAL_APP) handles
# the base model; suffixed ones each handle one ckpt. See deploy_eval_pool.sh +
# orchestrate.py.
_APP_SUFFIX = os.environ.get("EVAL_APP_SUFFIX", "")
app = modal.App(cfg.EVAL_APP + _APP_SUFFIX, image=image)

ckpts_vol = modal.Volume.from_name(cfg.CKPTS_VOL, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(cfg.HF_CACHE_VOL, create_if_missing=True)
results_vol = modal.Volume.from_name(cfg.RESULTS_VOL, create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=60 * 60 * 14,
    volumes={
        "/ckpts": ckpts_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def score_one(
    ckpt_path: str = "",
    ckpt_label: str = "",
    run_id: str = cfg.RUN_ID,
    bench_subset: list[str] | None = None,
    jee_n_samples: int = cfg.JEE_N_SAMPLES,
    jee_math_only: bool = cfg.JEE_MATH_ONLY,
    aime_n_samples: int = cfg.AIME_N_SAMPLES,
    math500_n_samples: int = cfg.MATH500_N_SAMPLES,
    math500_max_tokens: int = 32768,
    model_family: str | None = None,
    hub_repo: str = "",
    subfolder: str = "",
):
    """Score one checkpoint. `ckpt_path` is a path on the ckpts volume, or pass
    `hub_repo` (+ optional `subfolder`) to fetch a released student from the Hub:

        modal run eval_multi.py::score_one --hub-repo Chia-Mu-Lab/REP-models \
            --subfolder qwen25-7b-rep-q3_14b-clean --ckpt-label rep_14b_clean
    """
    sys.path.insert(0, "/workspace")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if hub_repo:
        from huggingface_hub import snapshot_download

        local = snapshot_download(hub_repo, allow_patterns=[f"{subfolder}/*"] if subfolder else None)
        ckpt_path = str(Path(local) / subfolder) if subfolder else local
        ckpt_label = ckpt_label or (subfolder or hub_repo.split("/")[-1])
    if not ckpt_path or not ckpt_label:
        raise ValueError("score_one: pass ckpt_path + ckpt_label, or hub_repo (+ subfolder)")

    try:
        ckpts_vol.reload()
    except Exception as exc:  # noqa: BLE001
        print(f"[score_one] ckpts_vol.reload() warn: {exc}", flush=True)

    from _common.multibench_runner import run_all  # type: ignore

    if bench_subset is None:
        bench_subset = list(cfg.EVAL_BENCH_SUBSET)

    return run_all(
        ckpt_path=ckpt_path,
        ckpt_label=ckpt_label,
        run_id=run_id,
        model_family=model_family or cfg.MODEL_FAMILY,
        results_root=Path("/results"),
        commit_fn=results_vol.commit,
        bench_subset=bench_subset,
        jee_n_samples=jee_n_samples,
        jee_math_only=jee_math_only,
        aime_n_samples=aime_n_samples,
        math500_n_samples=math500_n_samples,
        math500_max_tokens=math500_max_tokens,
    )


@app.function(
    timeout=60 * 5,
    volumes={"/results": results_vol},
)
def list_progress(run_id: str = cfg.RUN_ID):
    import json
    try:
        results_vol.reload()
    except Exception:
        pass
    out_dir = Path(f"/results/{run_id}")
    if not out_dir.exists():
        return []
    out = []
    for ckpt_dir in sorted(out_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        prog = ckpt_dir / "progress.json"
        if prog.exists():
            with prog.open() as f:
                out.append(json.load(f))
    return out
