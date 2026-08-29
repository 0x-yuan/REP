"""Modal app: push checkpoints from the train volume to a Hugging Face repo.

Two FCs:
  * upload_ckpt(volume_subpath, hf_repo, hf_subdir) — push one ckpt subdir.
  * upload_readme(hf_repo, body)                   — push a README.md to root.

Both are driven by run_uploads.py (which also generates the README from
final_results.json + config.py).

Deploy:
    modal deploy upload_ckpts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _config_loader import cfg  # noqa: E402

APP_NAME = f"distill-upload-{cfg.CKPTS_VOL}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.24",
        "tqdm",
    )
)

app = modal.App(APP_NAME, image=image)
ckpts_vol = modal.Volume.from_name(cfg.CKPTS_VOL)


@app.function(
    timeout=60 * 60,
    volumes={"/ckpts": ckpts_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=2,
    memory=4096,
)
def upload_ckpt(volume_subpath: str, hf_repo: str, hf_subdir: str) -> dict:
    """Upload `/ckpts/<volume_subpath>/` → `<hf_repo>:<hf_subdir>/`.

    Creates the repo if missing. Idempotent on re-upload (HfApi diffs vs the
    current repo state). Skips files that already match.
    """
    import os as _os
    from huggingface_hub import HfApi

    try:
        ckpts_vol.reload()
    except Exception as exc:  # noqa: BLE001
        print(f"[upload_ckpt] volume reload warn: {exc}", flush=True)

    src = Path("/ckpts") / volume_subpath
    if not src.exists():
        raise FileNotFoundError(f"missing {src} on the volume")

    api = HfApi(token=_os.environ["HF_TOKEN"])
    api.create_repo(
        repo_id=hf_repo, repo_type="model", exist_ok=True, private=False
    )
    print(f"[upload_ckpt] uploading {src} → {hf_repo}/{hf_subdir}", flush=True)

    api.upload_folder(
        folder_path=str(src),
        repo_id=hf_repo,
        repo_type="model",
        path_in_repo=hf_subdir,
        commit_message=f"upload {hf_subdir}",
        ignore_patterns=["*.log", "fsdp_config_qwen.json", "rng_state*.pth"],
    )
    return {"src": str(src), "repo": hf_repo, "subdir": hf_subdir, "ok": True}


@app.function(
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=1,
    memory=2048,
)
def upload_readme(hf_repo: str, body: str) -> dict:
    """Upload a README.md to the repo root."""
    import os as _os
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=_os.environ["HF_TOKEN"])
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True, private=False)

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name

    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="model",
        commit_message="update README.md",
    )
    return {"repo": hf_repo, "ok": True}
