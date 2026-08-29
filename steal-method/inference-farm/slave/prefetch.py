"""One-shot HF weight prefetch into the shared `sglang-hf-cache` volume.

Run once per (model, profile) before deploying the slave; the slave's
first cold-start then loads from the volume instead of HF.

# Usage
    MODEL_KEY=qwen3-1p7b uv run modal run inference-farm/slave/prefetch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from naming import (  # noqa: E402
    hf_cache_volume_name,
    prefetch_app_name,
    resolve_exp_id,
)
from registry import get_config, list_models  # noqa: E402

_MODEL_KEY = os.environ.get("MODEL_KEY", "").strip()
if not _MODEL_KEY:
    raise RuntimeError(
        "MODEL_KEY env var must be set.\n"
        f"Known models: {', '.join(list_models())}"
    )

_EXP_ID = resolve_exp_id()
CFG = get_config(_MODEL_KEY)
APP_NAME = prefetch_app_name(_MODEL_KEY, _EXP_ID)

app = modal.App(APP_NAME)
hf_cache_vol = modal.Volume.from_name(
    hf_cache_volume_name(_EXP_ID), create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.23,<2.0",
        "hf_transfer>=0.1.6",
    )
    .env(
        {
            "MODEL_KEY": _MODEL_KEY,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_python_source("registry", "observability", "naming")
)


@app.function(
    image=image,
    volumes={"/mnt/hf-cache": hf_cache_vol},
    cpu=4,
    timeout=2 * 3600,
    secrets=[
        modal.Secret.from_dict(
            {k: v for k, v in os.environ.items() if k in {"HF_TOKEN", "HUGGINGFACE_TOKEN"}}
        )
    ],
)
def prefetch() -> dict:
    from observability import emit, phase  # noqa: E402

    os.environ["HF_HOME"] = "/mnt/hf-cache/hf-home"
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/hf-cache/hf-home/transformers"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    cfg = get_config(_MODEL_KEY)
    emit("prefetch.start", hf_id=cfg.hf_id, cache_dir=os.environ["HF_HOME"])

    with phase("snapshot_download", hf_id=cfg.hf_id):
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=cfg.hf_id,
            cache_dir=os.environ["HF_HOME"],
        )

    with phase("volume_commit"):
        hf_cache_vol.commit()

    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, fname))
                file_count += 1
            except OSError:
                pass

    result = {
        "ok": True,
        "model_key": cfg.key,
        "hf_id": cfg.hf_id,
        "local_dir": local_dir,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1024**3, 2),
        "file_count": file_count,
    }
    emit("prefetch.done", **result)
    return result


@app.local_entrypoint()
def run() -> None:
    print(f"[client] prefetch {APP_NAME} → {hf_cache_volume_name(_EXP_ID)} volume ...")
    info = prefetch.remote()
    print(f"[client] {info}")
