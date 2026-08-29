"""Modal app: distillation training (full-parameter SFT) for ONE experiment.

Reads `config.py` (or $DISTILL_CONFIG) for every per-experiment knob; the
training body itself lives in `_common/sft_runner.py` and is recipe-frozen.

Deploy:
    modal deploy train.py

Spawn:
    uv run --with modal python spawn_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# In-container (Modal): _common is added at /workspace/_common by the image.
# Locally (deploy): _common sits next to this file.
for _common_candidate in (HERE / "_common", Path("/workspace/_common")):
    if (_common_candidate / "sft_runner.py").exists():
        sys.path.insert(0, str(_common_candidate))
        break

from _config_loader import cfg  # noqa: E402
from sft_runner import build_image, run_sft, list_ckpts_impl  # type: ignore  # noqa: E402

# `build_image` does add_local_dir(repo_root / "s1") and (repo_root / "_common");
# The engine directory is itself the repo root.
image = build_image(HERE).add_local_file(
    str(HERE / "config.py"), "/root/config.py", copy=True
).add_local_file(
    str(HERE / "_config_loader.py"), "/root/_config_loader.py", copy=True
)

app = modal.App(cfg.TRAIN_APP, image=image)

ckpts_vol = modal.Volume.from_name(cfg.CKPTS_VOL, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(cfg.HF_CACHE_VOL, create_if_missing=True)


@app.function(
    gpu=f"{cfg.GPU_TYPE}:{cfg.GPU_COUNT}",
    timeout=60 * 60 * 23,  # 23h hard cap — Modal max for one FC
    volumes={
        "/ckpts": ckpts_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(
    run_id: str = cfg.RUN_ID,
    source_repo: str = cfg.SOURCE_REPO,
    source_config: str | None = getattr(cfg, "SOURCE_CONFIG", None),
    dataset_subdir: str = cfg.DATASET_SUBDIR,
    model_name: str = cfg.MODEL_NAME,
    epochs: int = cfg.EPOCHS,
    save_strategy: str = cfg.SAVE_STRATEGY,
    block_size: int = cfg.BLOCK_SIZE,
    grad_accum: int = cfg.GRAD_ACCUM,
    micro_batch: int = cfg.MICRO_BATCH,
    learning_rate: float = cfg.LEARNING_RATE,
):
    return run_sft(
        volume=ckpts_vol,
        run_id=run_id,
        source_repo=source_repo,
        source_config=source_config,
        dataset_subdir=dataset_subdir,
        model_name=model_name,
        epochs=epochs,
        save_strategy=save_strategy,
        block_size=block_size,
        grad_accum=grad_accum,
        micro_batch=micro_batch,
        learning_rate=learning_rate,
        gpu_count=cfg.GPU_COUNT,
        use_fsdp=cfg.USE_FSDP,
    )


@app.function(
    timeout=60 * 5,
    volumes={"/ckpts": ckpts_vol},
)
def list_ckpts(run_id: str = cfg.RUN_ID):
    return list_ckpts_impl(ckpts_vol, run_id)
