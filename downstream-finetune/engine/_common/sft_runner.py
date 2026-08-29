"""Shared SFT runner for distillation experiments.

Each per-experiment train.py is a thin Modal app shell that:
  * picks per-profile APP_NAME, VOLUME_NAME, DEFAULT_SOURCE_REPO, RUN_ID
  * calls `build_image()` once
  * declares @app.function() bindings that delegate to `run_sft(...)`
    and `list_ckpts(...)` here

The training body itself (dataset prep + torchrun launch + log streaming +
volume commits) is identical across all experiments — only the dataset
source and run_id differ.

Public API:
  build_image(repo_root: Path) -> modal.Image
  run_sft(volume, run_id, source_repo, dataset_subdir, model_name=..., ...) -> dict
  list_ckpts(volume, run_id) -> list[dict]
"""
from __future__ import annotations

import importlib.util
import json as _json
import os
import subprocess
import sys
from pathlib import Path

import modal

# Blackwell B200 (SM 100) requires CUDA-12.8-built PyTorch wheels.
# flash-attn 2.7.4.post1 ships a cu12+torch2.7 wheel (cxx11abiFALSE/cp310) that
# links cleanly against the cu128 PyTorch wheel. Keeping abiFALSE (matches the
# default torch wheel's _GLIBCXX_USE_CXX11_ABI=0 build).
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)


def build_image(repo_root: Path) -> modal.Image:
    """Return the B200-ready Modal image used by every training app.

    Key differences vs the H200 distillation image:
      * torch 2.7.1 + torchvision 0.22.1 + torchaudio 2.7.1 from PyTorch's
        cu128 wheel index (sm_100 kernels for Blackwell B200).
      * transformers 4.46.3 (matches the eval image), datasets 3.2.0.
      * flash-attn 2.7.4.post1 cu12+torch2.7 wheel.
    """
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
        )
        .apt_install("git", "build-essential", "ninja-build")
        .run_commands(
            "pip install --upgrade pip",
            "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
            "--index-url https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "transformers==4.46.3",
            "datasets==3.2.0",
            "accelerate==1.0.1",
            "trl==0.12.0",
            "wandb==0.17.3",
            "huggingface_hub",
            "sentencepiece",
            "protobuf",
            "packaging",
            "ninja",
        )
        .pip_install(FLASH_ATTN_WHEEL)
        .add_local_dir(repo_root / "s1", remote_path="/workspace/s1", copy=True)
        .add_local_dir(repo_root / "_common", remote_path="/workspace/_common", copy=True)
    )


# ---------------------------------------------------------------------------
# Patches applied INSIDE the container the first time training starts.
# ---------------------------------------------------------------------------


def _patch_sft_py() -> None:
    sft_py = Path("/workspace/s1/train/sft.py")
    src = sft_py.read_text()

    if "LOCAL_PATH_LOAD" not in src:
        patched = src.replace(
            "    dataset = load_dataset(config.train_file_path)\n",
            (
                "    if config.train_file_path.startswith('/'):\n"
                "        from datasets import load_from_disk  # LOCAL_PATH_LOAD\n"
                "        dataset = load_from_disk(config.train_file_path)\n"
                "    else:\n"
                "        dataset = load_dataset(config.train_file_path)\n"
            ),
        )
        if patched != src:
            sft_py.write_text(patched)
            src = patched
            print("[sft_runner] patched sft.py: load_from_disk for local paths", flush=True)

    # Force flash_attention_2 for ALL Qwen/Llama models, not just 70B.
    # Without it, sdpa on a 32768-token sequence pre-allocates O(S²) memory
    # per attention layer, which under DDP (no param sharding) blows past
    # B200's 178 GiB usable HBM and OOMs early in epoch 1. With
    # flash_attention_2 attention memory drops to O(S·d) ≈ 0.5 GiB per layer.
    if "MODAL_FLASH_ATTN_NON70B" not in src:
        patched = src.replace(
            "        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)\n",
            (
                "        # MODAL_FLASH_ATTN_NON70B: force flash-attn for non-70B too\n"
                "        kwargs = {\"attn_implementation\": \"flash_attention_2\", \"torch_dtype\": \"auto\", \"use_cache\": False}\n"
                "        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)\n"
            ),
        )
        if patched != src:
            sft_py.write_text(patched)
            src = patched
            print("[sft_runner] patched sft.py: flash_attention_2 for non-70B", flush=True)

    if "MODAL_QWEN_DEFAULT" not in src:
        patched = src.replace(
            '    elif "Qwen" in config.model_name:\n'
            '        instruction_template = "<|im_start|>user"\n'
            '        response_template = "<|im_start|>assistant\\n"\n'
            '        # Use a token that is never used\n'
            '        tokenizer.pad_token = "<|fim_pad|>"\n',
            '    elif "Qwen" in config.model_name:\n'
            '        instruction_template = "<|im_start|>user"\n'
            '        response_template = "<|im_start|>assistant\\n"\n'
            '        # Use a token that is never used\n'
            '        tokenizer.pad_token = "<|fim_pad|>"\n'
            '    else: # MODAL_QWEN_DEFAULT\n'
            '        instruction_template = "<|im_start|>user"\n'
            '        response_template = "<|im_start|>assistant\\n"\n'
            '        tokenizer.pad_token = "<|fim_pad|>"\n',
        )
        if patched != src:
            sft_py.write_text(patched)
            print("[sft_runner] patched sft.py: MODAL_QWEN_DEFAULT fallback", flush=True)


def _build_dataset(volume, dataset_dir: Path, source_repo: str, tokenizer_name: str, source_config: str | None = None) -> None:
    try:
        volume.reload()
    except Exception:
        pass
    marker = dataset_dir / "train" / "dataset_info.json"
    if marker.exists():
        print(f"[sft_runner] dataset already prepared at {dataset_dir}", flush=True)
        return
    print(f"[sft_runner] building dataset at {dataset_dir} from {source_repo} (config={source_config}) ...", flush=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "dataset_prep", "/workspace/_common/dataset_prep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(str(dataset_dir), tokenizer_name, source_repo=source_repo, config_name=source_config)
    volume.commit()


def run_sft(
    *,
    volume,
    run_id: str,
    source_repo: str,
    dataset_subdir: str,
    source_config: str | None = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 5,
    save_strategy: str = "epoch",
    save_steps: int = 250,
    block_size: int = 32768,
    grad_accum: int = 4,
    micro_batch: int = 1,
    learning_rate: float = 1e-5,
    gpu_count: int = 4,
    use_fsdp: bool = True,
) -> dict:
    """Run one s1 sft.py training.

    Idempotent dataset prep + torchrun launch. Streams logs + commits volume
    every time an HF Trainer checkpoint is written. Returns when torchrun
    exits with code 0 (raises on non-zero).
    """
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["WANDB_MODE"] = "disabled"

    sys.path.insert(0, "/workspace/_common")

    out_dir = Path("/ckpts") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    dataset_dir = Path(f"/ckpts/datasets/{dataset_subdir}")
    # For continuation runs from /ckpts/<prev_run>, use the base-family tokenizer.
    tokenizer_name = model_name if not model_name.startswith("/") else "Qwen/Qwen2.5-7B-Instruct"
    _build_dataset(volume, dataset_dir, source_repo, tokenizer_name, source_config)

    # FSDP config — wraps Qwen2DecoderLayer, gathers FULL_STATE_DICT for vLLM-friendly ckpts.
    fsdp_cfg = {
        "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
        "state_dict_type": "FULL_STATE_DICT",
    }
    fsdp_cfg_path = out_dir / "fsdp_config_qwen.json"
    with fsdp_cfg_path.open("w") as f:
        _json.dump(fsdp_cfg, f)

    _patch_sft_py()

    # When `use_fsdp=False`, fall back to plain DDP (full param replication).
    # Workaround for the torch 2.7.1 + FSDP `_group_tensors_by_device_and_dtype`
    # device-mismatch bug that crashes at the first optimizer step AFTER an
    # end-of-epoch ckpt save (reproduced on real 7B runs even with
    # `--optim=adamw_torch_fused`). 7B + AdamW on B200 (192GB) fits in
    # plain DDP with ~70-100GB per rank; no FSDP needed.
    fsdp_args = (
        ["--fsdp=full_shard auto_wrap", f"--fsdp_config={fsdp_cfg_path}"]
        if use_fsdp else []
    )

    cmd = [
        "torchrun",
        f"--nproc-per-node={gpu_count}",
        "--master_port=12345",
        "/workspace/s1/train/sft.py",
        f"--block_size={block_size}",
        f"--per_device_train_batch_size={micro_batch}",
        f"--per_device_eval_batch_size={micro_batch}",
        f"--gradient_accumulation_steps={grad_accum}",
        f"--num_train_epochs={epochs}",
        f"--train_file_path={dataset_dir}",
        f"--model_name={model_name}",
        "--warmup_ratio=0.05",
        *fsdp_args,
        "--bf16=True",
        "--eval_strategy=no",
        "--logging_steps=1",
        f"--save_strategy={save_strategy}",
        f"--save_steps={save_steps}",
        "--lr_scheduler_type=cosine",
        f"--learning_rate={learning_rate}",
        "--weight_decay=1e-4",
        "--adam_beta1=0.9",
        "--adam_beta2=0.95",
        f"--output_dir={out_dir}",
        "--push_to_hub=False",
        "--save_only_model=True",
        "--gradient_checkpointing=True",
        "--report_to=none",
        # torch 2.7.1 + FSDP + default adamw_torch hit a "Tensors of the same
        # index must be on the same device" RuntimeError on the FIRST optimizer
        # step AFTER an end-of-epoch checkpoint save. The fused C++
        # implementation routes around the broken _group_tensors path.
        "--optim=adamw_torch_fused",
    ]

    print("=" * 80, flush=True)
    print(f"[sft_runner] {' '.join(cmd)}", flush=True)
    print("=" * 80, flush=True)

    with log_path.open("a", buffering=1) as logf:
        logf.write("### CMD\n" + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd="/workspace/s1",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=os.environ,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                if "Saving model checkpoint to" in line or "checkpoint-" in line.lower():
                    try:
                        volume.commit()
                    except Exception as e:  # noqa: BLE001
                        print(f"[sft_runner] volume.commit() warn: {e}", flush=True)
        finally:
            ret = proc.wait()
        logf.write(f"\n### exit code = {ret}\n")

    try:
        volume.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[sft_runner] final volume.commit() warn: {e}", flush=True)

    print(f"[sft_runner] torchrun exit={ret}", flush=True)
    if ret != 0:
        raise RuntimeError(f"torchrun failed with exit code {ret}")
    return {"run_id": run_id, "out_dir": str(out_dir), "exit_code": ret}


def list_ckpts_impl(volume, run_id: str) -> list[dict]:
    try:
        volume.reload()
    except Exception:
        pass
    out_dir = Path("/ckpts") / run_id
    if not out_dir.exists():
        return []
    ckpts = []
    for p in sorted(out_dir.glob("checkpoint-*")):
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        files = sorted(f.name for f in p.iterdir())
        ckpts.append({"step": step, "path": str(p), "files": files})
    return ckpts
