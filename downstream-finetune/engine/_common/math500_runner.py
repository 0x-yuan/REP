"""Standalone MATH-500 runner (n=3, T=0.5, last_boxed + math_verify).

Reuses helpers from multibench_runner so the extraction / scoring / IO
contracts stay identical to AIME24/AIME25. Writes `math500.summary.json`
into the same `<results_root>/<run_id>/<ckpt_label>/` directory next to the
existing main-lane summaries; the runner is deterministic (seed=7) so a
re-run is bit-identical and the idempotent skip-if-summary-exists guard
applies.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional


class _NoopProgress:
    """Minimal progress shim that satisfies _run_math_task_nk's contract
    (start/finish/fail) without touching the shared progress.json file —
    we don't want to clobber state from the main-lane container that may
    be running concurrently on the same ckpt directory."""
    def __init__(self) -> None:
        self.events: list[tuple[str, str, float | None]] = []

    def start(self, bench: str) -> None:
        self.events.append((bench, "start", None))
        print(f"[math500/progress] start {bench}", flush=True)

    def finish(self, bench: str, metric: float | None) -> None:
        self.events.append((bench, "finish", metric))
        print(f"[math500/progress] finish {bench} metric={metric}", flush=True)

    def fail(self, bench: str, err: str) -> None:
        self.events.append((bench, "fail", None))
        print(f"[math500/progress] fail {bench}: {err}", flush=True)


def run_math500(
    *,
    ckpt_path: str,
    ckpt_label: str,
    run_id: str,
    model_family: str,
    results_root: Path,
    commit_fn: Optional[Callable[[], None]] = None,
    n_samples: int = 3,
    temperature: float = 0.5,
    max_new_tokens: int = 32768,
    gpu_memory_utilization: float = 0.85,
) -> dict[str, Any]:
    """Run HuggingFaceH4/MATH-500 split=test under TIA-style n-averaged protocol."""
    for p in ("/workspace", str(Path(__file__).resolve().parent.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from _common.multibench_runner import (
        _build_aime_prompt,
        _build_math_vllm,
        _math_verify_equiv,
        _last_boxed_only_string,
        _remove_boxed,
        _run_math_task_nk,
        _summary_path,
    )
    from datasets import load_dataset

    out_dir = results_root / run_id / ckpt_label
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = "math500"
    summary_path = _summary_path(out_dir, bench)
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text())
            v = prev.get("accuracy")
            if isinstance(v, (int, float)):
                print(f"[{bench}] SKIP — existing summary accuracy={v:.4f}", flush=True)
                return prev
        except Exception:
            pass

    print(f"[math500] booting vLLM ckpt={ckpt_path}", flush=True)
    llm = _build_math_vllm(ckpt_path, gpu_memory_utilization=gpu_memory_utilization)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = []
    rows_meta = []
    for row in ds:
        q = row.get("problem") or row.get("question")
        a = row.get("answer")
        if q is None or a is None:
            continue
        prompts.append(_build_aime_prompt(str(q), model_family))
        rows_meta.append({
            "gold": str(a),
            "subject": row.get("subject"),
            "level": row.get("level"),
            "unique_id": row.get("unique_id"),
        })

    print(f"[math500] dataset n={len(prompts)}", flush=True)

    def _score(text: str, meta: dict) -> tuple[Optional[str], int]:
        extracted = _remove_boxed(_last_boxed_only_string(text))
        if extracted is None:
            return None, 0
        return extracted, int(_math_verify_equiv(meta["gold"], extracted))

    progress = _NoopProgress()
    summary = _run_math_task_nk(
        llm=llm, bench=bench, out_dir=out_dir, progress=progress,
        ckpt_path=ckpt_path, prompts=prompts, rows_meta=rows_meta,
        extract_and_score_fn=_score, n_samples=n_samples, temperature=temperature,
        max_new_tokens=max_new_tokens, label="math500_test",
    )

    if commit_fn is not None:
        try:
            commit_fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[math500] commit warn: {exc}", flush=True)

    return summary
