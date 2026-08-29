"""Drive the per-ckpt HF fanout + push a generated README.

Reads the experiment config + final_results.json (produced by cron_tick.py
after all evals complete) and:
  1. Spawns one upload FC per ckpt step that exists on the train volume.
  2. Waits for them.
  3. Renders a README.md and pushes it to the repo root.

Run AFTER training + eval are done:

    uv run --with modal python run_uploads.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _config_loader import cfg  # noqa: E402


def _final_results() -> dict | None:
    p = HERE / "final_results.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _list_ckpt_steps_on_volume() -> list[int]:
    """List checkpoint steps on the ckpts volume via `modal volume ls`."""
    import subprocess

    proc = subprocess.run(
        ["modal", "volume", "ls", cfg.CKPTS_VOL, f"{cfg.RUN_ID}/"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    steps: list[int] = []
    if proc.returncode != 0:
        return steps
    for ln in proc.stdout.splitlines():
        if "checkpoint-" in ln:
            try:
                s = ln.split("checkpoint-")[-1].rstrip("/").split("/")[0]
                steps.append(int(s))
            except Exception:
                pass
    return sorted(set(steps))


def _build_readme(results: dict | None, ckpt_steps: list[int]) -> str:
    """Render a README.md from config + results. Falls back gracefully if
    final_results.json is missing.
    """
    sharding = (
        "FSDP (full_shard auto_wrap, Qwen2DecoderLayer, FULL_STATE_DICT)"
        if cfg.USE_FSDP else "plain DDP (no FSDP)"
    )

    lines = []
    lines.append("---")
    lines.append("license: apache-2.0")
    lines.append(f"base_model: {cfg.MODEL_NAME}")
    lines.append("datasets:")
    lines.append(f"  - {cfg.SOURCE_REPO}")
    lines.append("language:")
    lines.append("  - en")
    lines.append("library_name: transformers")
    lines.append("pipeline_tag: text-generation")
    lines.append("tags:")
    lines.append("  - distillation")
    lines.append("  - reasoning")
    lines.append("---\n")

    title = cfg.HF_REPO.split("/")[-1] if cfg.HF_REPO else cfg.RUN_ID
    lines.append(f"# {title}\n")
    lines.append(
        f"Distilled checkpoints from full-parameter SFT of "
        f"`{cfg.MODEL_NAME}` on [`{cfg.SOURCE_REPO}`](https://huggingface.co/datasets/{cfg.SOURCE_REPO}). "
        f"{cfg.EPOCHS} epoch ckpts, {cfg.GPU_COUNT}×{cfg.GPU_TYPE}, "
        f"eff_batch {cfg.MICRO_BATCH * cfg.GRAD_ACCUM * cfg.GPU_COUNT}, "
        f"lr {cfg.LEARNING_RATE} cosine warmup 0.05.\n"
    )
    if cfg.HF_VARIANT_NOTE:
        lines.append(f"{cfg.HF_VARIANT_NOTE}\n")

    lines.append("## Training recipe\n")
    lines.append(
        "| field | value |\n"
        "|---|---|\n"
        f"| Student | `{cfg.MODEL_NAME}` |\n"
        f"| Dataset | [`{cfg.SOURCE_REPO}`](https://huggingface.co/datasets/{cfg.SOURCE_REPO}) |\n"
        f"| Hardware | {cfg.GPU_COUNT}×{cfg.GPU_TYPE} (Modal) |\n"
        f"| Epochs | {cfg.EPOCHS} (one ckpt per epoch) |\n"
        f"| Block size | {cfg.BLOCK_SIZE} |\n"
        f"| Micro / Grad-accum / Effective batch | "
        f"{cfg.MICRO_BATCH} / {cfg.GRAD_ACCUM} / "
        f"{cfg.MICRO_BATCH * cfg.GRAD_ACCUM * cfg.GPU_COUNT} |\n"
        f"| Learning rate | {cfg.LEARNING_RATE} (cosine, warmup 0.05) |\n"
        f"| Optimizer | AdamW (β=0.9/0.95, wd=1e-4) |\n"
        f"| Sharding | {sharding} |\n"
        f"| Attention | flash_attention_2 |\n"
        f"| Precision | bf16 |\n"
    )

    if results:
        base = results.get("base", {})
        sorted_labels = sorted(
            (l for l in results if l.startswith("step-")),
            key=lambda l: int(l.split("-")[1]),
        )

        def _fmt(v, base_v=None):
            if v is None:
                return "—"
            pct = v * 100
            if base_v is None:
                return f"{pct:.2f}"
            delta = (v - base_v) * 100
            sign = "+" if delta >= 0 else ""
            return f"{pct:.2f} ({sign}{delta:.1f})"

        lines.append("\n## Evaluation\n")
        lines.append(
            "All numbers are % accuracy; `(±N.N)` is the delta vs the base "
            f"`{cfg.MODEL_NAME}` evaluated under the same protocol.\n"
        )
        lines.append(
            "| ckpt | epoch | AIME24 | AIME25 | MATH500 | JEE-math | LCB-v5 |\n"
            "|---|---|---|---|---|---|---|"
        )
        lines.append(
            f"| base | — | {_fmt(base.get('aime24'))} | {_fmt(base.get('aime25'))} | "
            f"{_fmt(base.get('math500'))} | {_fmt(base.get('jeebench'))} | "
            f"{_fmt(base.get('lcb_v5'))} |"
        )
        for idx, label in enumerate(sorted_labels, 1):
            r = results[label]
            lines.append(
                f"| `{label}` | ep{idx} | "
                f"{_fmt(r.get('aime24'), base.get('aime24'))} | "
                f"{_fmt(r.get('aime25'), base.get('aime25'))} | "
                f"{_fmt(r.get('math500'), base.get('math500'))} | "
                f"{_fmt(r.get('jeebench'), base.get('jeebench'))} | "
                f"{_fmt(r.get('lcb_v5'), base.get('lcb_v5'))} |"
            )

    if ckpt_steps:
        lines.append("\n## Checkpoints layout\n")
        lines.append(
            "Each epoch ckpt lives in its own subdirectory inside this repo. "
            "To load a specific epoch with 🤗 Transformers:\n"
        )
        lines.append("```python")
        lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
        lines.append(f'repo = "{cfg.HF_REPO}"')
        mid = ckpt_steps[len(ckpt_steps) // 2]
        lines.append(
            f'sub  = "checkpoint-{mid}"  # one of: '
            + ", ".join(f"checkpoint-{s}" for s in ckpt_steps)
        )
        lines.append(
            'model = AutoModelForCausalLM.from_pretrained(repo, subfolder=sub, torch_dtype="bfloat16")'
        )
        lines.append('tok   = AutoTokenizer.from_pretrained(repo, subfolder=sub)')
        lines.append("```")

    lines.append("\n## Caveats\n")
    lines.append(
        "* Research artifact. Not intended for production use."
    )
    lines.append(
        "* Evaluation uses a single seed (T=0.5, seed=7 for vLLM); "
        "per-ckpt variance is ±1-2 pp."
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    if not cfg.HF_REPO:
        print("[run_uploads] cfg.HF_REPO is empty — set it in config.py to enable uploads. Aborting.",
              file=sys.stderr)
        return 2

    app_name = f"distill-upload-{cfg.CKPTS_VOL}"
    upload_ckpt_fn = modal.Function.from_name(app_name, "upload_ckpt")
    upload_readme_fn = modal.Function.from_name(app_name, "upload_readme")

    ckpt_steps = _list_ckpt_steps_on_volume()
    if not ckpt_steps:
        print(f"[run_uploads] no checkpoints found on {cfg.CKPTS_VOL}:{cfg.RUN_ID}/. Did training finish?",
              file=sys.stderr)
        return 1

    print(f"[run_uploads] spawning {len(ckpt_steps)} ckpt uploads → {cfg.HF_REPO}", flush=True)
    fcs = []
    for step in ckpt_steps:
        sub = f"checkpoint-{step}"
        fc = upload_ckpt_fn.spawn(
            volume_subpath=f"{cfg.RUN_ID}/{sub}",
            hf_repo=cfg.HF_REPO,
            hf_subdir=sub,
        )
        print(f"  spawned {sub} fc={fc.object_id}", flush=True)
        fcs.append((sub, fc))

    print("\n[run_uploads] waiting for ckpt uploads ...", flush=True)
    for sub, fc in fcs:
        try:
            fc.get()
            print(f"  OK  {sub} → {cfg.HF_REPO}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {sub} → {cfg.HF_REPO}: {exc}", flush=True)

    body = _build_readme(_final_results(), ckpt_steps)
    (HERE / "README.HF.md").write_text(body)
    print(f"\n[run_uploads] pushing README → {cfg.HF_REPO}", flush=True)
    upload_readme_fn.remote(hf_repo=cfg.HF_REPO, body=body)
    print("[run_uploads] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
