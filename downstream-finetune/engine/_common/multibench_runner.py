"""In-container multi-benchmark eval driver for distill ckpts.

Evaluation protocol:
    AIME24      — n=3 @ T=0.5 averaged   (30 problems × 3 = 90 completions)
    AIME25      — n=3 @ T=0.5 averaged   (30 × 3 = 90)
    GPQA-diamond— n=3 @ T=0.5 averaged   (198 × 3 = 594)
    JEEBench    — n=6 @ T=0.5 averaged   (515 × 6 = 3090)
    LCB-v5      — n=3 @ T=0.5 pass@1     (167 × 3 = 501 + sandboxed exec)

Math benches (AIME, GPQA, JEE) share a SINGLE vLLM boot via `run_math_block()`.
LCB has its own vLLM via `lcb_runner` subprocess. The Modal app file decides
whether a container runs math+code, math-only (skip_lcb), or code-only
(lcb_only) — this enables the 2-way per-ckpt split: math container A + code
container B in parallel.

After EACH benchmark, writes <bench>.summary.json + progress.json and commits
the results volume so the user sees incremental results.

The runner is imported by `eval_multi.py` at the top of this folder, which
owns the @app.function decoration and volume / GPU / secret wiring.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Inside the Modal container the vendored s1 lm-eval-harness is installed
# editable from /workspace/s1/eval/lm-evaluation-harness and the vendored
# JEEBench scorer lives under /workspace/_common/scorers.

V3_MARKERS = ["$ cat reasoning_trace.txt", "$ cat final_answer.txt"]


# ----------------------------- helpers --------------------------------------


def _now() -> float:
    return time.time()


def _human_dt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


@dataclass
class ProgressRecord:
    run_id: str
    ckpt_label: str
    ckpt_path: str
    model_family: str
    started_at: float
    benchmarks: dict[str, dict]  # bench_name -> {"status": ..., "started_at": ..., "ended_at": ..., "metric": ...}

    def to_dict(self) -> dict:
        return asdict(self)


class Progress:
    """Per-ckpt progress file written to /results/<run_id>/<ckpt_label>/progress.json.

    Updated before + after each benchmark so the user can poll the volume.
    """

    BENCHES = ["aime24", "aime25", "gpqa_diamond", "math500", "jeebench", "lcb_v5"]

    def __init__(
        self,
        out_dir: Path,
        record: ProgressRecord,
        commit_fn=lambda: None,
    ) -> None:
        self.out_dir = out_dir
        self.record = record
        self.commit_fn = commit_fn
        self.path = out_dir / "progress.json"
        for b in self.BENCHES:
            self.record.benchmarks.setdefault(
                b, {"status": "pending", "started_at": None, "ended_at": None, "metric": None}
            )
        self._flush()

    def start(self, bench: str) -> None:
        self.record.benchmarks[bench]["status"] = "running"
        self.record.benchmarks[bench]["started_at"] = _now()
        self._flush()

    def finish(self, bench: str, metric: float | None) -> None:
        e = self.record.benchmarks[bench]
        e["status"] = "done"
        e["ended_at"] = _now()
        e["metric"] = metric
        self._flush()

    def fail(self, bench: str, err: str) -> None:
        e = self.record.benchmarks[bench]
        e["status"] = "failed"
        e["ended_at"] = _now()
        e["error"] = err[:2000]
        self._flush()

    def _flush(self) -> None:
        _write_json(self.path, self.record.to_dict())
        try:
            self.commit_fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[progress] commit warn: {exc}", flush=True)


# ----------------------- chat-template helpers ------------------------------


def _qwen_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _llama_prompt(question: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _gemma_prompt(question: str) -> str:
    # Gemma-3 instruct chat template; no system role, BOS managed by tokenizer
    # but we emit <bos> explicitly because vLLM with add_special_tokens=False
    # (lm-eval default for apply_chat_template=True path is to not add BOS).
    return (
        "<bos><start_of_turn>user\n"
        f"{question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _build_prompt(question: str, model_family: str) -> str:
    if model_family == "qwen25":
        return _qwen_prompt(question)
    if model_family == "llama31":
        return _llama_prompt(question)
    if model_family == "gemma3":
        return _gemma_prompt(question)
    raise ValueError(f"unknown model_family: {model_family}")


# --------------------- lm-eval-harness CLI driver ---------------------------


def run_lm_eval_block(
    ckpt_path: str,
    out_dir: Path,
    progress: Progress,
    *,
    tasks: list[str],
    max_gen_toks: int = 32768,
    gpu_memory_utilization: float = 0.85,
    processor: str = "gpt-4o-mini",
) -> dict[str, float]:
    """Run the AIME24 + AIME25 + GPQA-diamond block via vendored s1 lm-eval-harness.

    Returns {task_name: accuracy} after parsing lm-eval's results JSON.
    Per-task summary.json + progress.json are written immediately after the
    single CLI call completes (lm-eval doesn't checkpoint mid-task).
    """
    lm_out = out_dir / "lm_eval"
    lm_out.mkdir(parents=True, exist_ok=True)

    # Mark all three as 'running' simultaneously since lm-eval handles them as one batch.
    for t in tasks:
        progress.start(_lm_eval_task_to_bench_key(t))

    model_args = (
        f"pretrained={ckpt_path}"
        ",dtype=bfloat16"
        ",tensor_parallel_size=1"
        f",gpu_memory_utilization={gpu_memory_utilization}"
        ",max_model_len=32768"
        ",enforce_eager=False"
        ",trust_remote_code=True"
    )

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", "auto",
        "--apply_chat_template",
        "--output_path", str(lm_out),
        "--log_samples",
        "--gen_kwargs", f"max_gen_toks={max_gen_toks}",
    ]
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env["PROCESSOR"] = processor  # required for openai_math + gpqa_diamond_openai utils

    print(f"[lm_eval] CMD: {' '.join(cmd)}", flush=True)
    t0 = _now()
    proc = subprocess.run(cmd, env=env, capture_output=False)
    if proc.returncode != 0:
        for t in tasks:
            progress.fail(_lm_eval_task_to_bench_key(t), f"lm-eval CLI returncode={proc.returncode}")
        raise RuntimeError(f"lm-eval CLI returned {proc.returncode}")
    print(f"[lm_eval] wall={_human_dt(_now()-t0)}", flush=True)

    # Find the results_*.json file (lm-eval writes a fresh timestamped one per run).
    results_json = _find_lm_eval_results(lm_out)
    if results_json is None:
        for t in tasks:
            progress.fail(_lm_eval_task_to_bench_key(t), "lm-eval results json missing")
        raise RuntimeError("lm-eval results.json not found")

    with results_json.open() as f:
        lm_results = json.load(f)
    task_metrics = lm_results.get("results", {})
    samples_root = results_json.parent

    out_accs: dict[str, float] = {}
    for task in tasks:
        bench = _lm_eval_task_to_bench_key(task)
        per_task = task_metrics.get(task, {})
        acc = _pick_lm_eval_accuracy(per_task)

        # Pull per-sample log for V3-marker scan + reasoning-length stats.
        samples_path = _find_samples_file(samples_root, task)
        v3_a_rate, v3_b_rate, truncation_rate, n_samples = _scan_samples(samples_path)

        bench_summary = {
            "benchmark": task,
            "ckpt_path": ckpt_path,
            "n_samples": n_samples,
            "accuracy": acc,
            "raw_metrics": per_task,
            "v3_reasoning_trace_rate": v3_a_rate,
            "v3_final_answer_rate": v3_b_rate,
            "truncation_rate": truncation_rate,
            "max_gen_toks": max_gen_toks,
            "processor": processor,
            "lm_eval_results_path": str(results_json),
            "lm_eval_samples_path": str(samples_path) if samples_path else None,
        }
        _write_json(out_dir / f"{bench}.summary.json", bench_summary)
        progress.finish(bench, acc)
        out_accs[task] = acc
        print(f"[lm_eval] {task}: acc={acc}", flush=True)

    return out_accs


def _lm_eval_task_to_bench_key(task: str) -> str:
    return {
        "aime24_nofigures": "aime24",
        "aime25_nofigures": "aime25",
        "gpqa_diamond_openai": "gpqa_diamond",
        "openai_math": "openai_math",
    }.get(task, task)


def _find_lm_eval_results(lm_out: Path) -> Optional[Path]:
    candidates = list(lm_out.rglob("results_*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_samples_file(samples_root: Path, task: str) -> Optional[Path]:
    candidates = list(samples_root.rglob(f"samples_{task}_*.jsonl"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _pick_lm_eval_accuracy(per_task: dict) -> Optional[float]:
    # lm-eval reports a variety of metric names. Prefer exact_match,strict-match
    # (AIME, openai_math) then acc,none (gpqa_diamond_openai) then any *match*.
    for key in (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
        "score,none",
    ):
        if key in per_task and isinstance(per_task[key], (int, float)):
            return float(per_task[key])
    for k, v in per_task.items():
        if isinstance(v, (int, float)) and ("match" in k or "acc" in k or "score" in k):
            return float(v)
    return None


def _scan_samples(samples_path: Optional[Path]) -> tuple[float, float, float, int]:
    if samples_path is None or not samples_path.exists():
        return 0.0, 0.0, 0.0, 0
    n = 0
    v3a = v3b = trunc = 0
    with samples_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            resps = row.get("resps") or row.get("filtered_resps") or []
            text = ""
            if resps and isinstance(resps[0], list) and resps[0]:
                text = str(resps[0][0])
            elif isinstance(resps, list) and resps and isinstance(resps[0], str):
                text = resps[0]
            if V3_MARKERS[0] in text:
                v3a += 1
            if V3_MARKERS[1] in text:
                v3b += 1
            # truncation: lm-eval doesn't always log finish_reason; approximate by length.
            if len(text) > 0 and not re.search(r"\\boxed", text) and len(text) > 8000:
                trunc += 1
    return (v3a / max(n, 1), v3b / max(n, 1), trunc / max(n, 1), n)


# -------------------- Math block (shared vLLM, n=3 / n=6) -------------------

import random


def _last_boxed_only_string(text: str) -> Optional[str]:
    if not text:
        return None
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = idx
    depth = 0
    started = False
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return text[idx : i + 1]
        i += 1
    return None


def _remove_boxed(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    for tok in ("\\boxed{", "\\fbox{"):
        if s.startswith(tok) and s.endswith("}"):
            return s[len(tok):-1]
    return None


def _math_verify_equiv(gold: str, pred: str) -> bool:
    try:
        from math_verify import parse, verify
        gold_p = parse(f"${gold}$" if "$" not in gold else gold)
        pred_p = parse(f"${pred}$" if "$" not in pred else pred)
        return bool(verify(gold_p, pred_p))
    except Exception:
        g = re.sub(r"\s+", "", str(gold))
        p = re.sub(r"\s+", "", str(pred))
        return g == p


def _aime_question_field(row: dict) -> str:
    for k in ("Question", "problem", "question"):
        if k in row and row[k]:
            return str(row[k])
    raise KeyError(f"AIME row missing Question field: {list(row.keys())}")


def _aime_answer_field(row: dict) -> str:
    for k in ("Answer", "answer", "solution"):
        if k in row and row[k] is not None:
            return str(row[k])
    raise KeyError(f"AIME row missing answer field: {list(row.keys())}")


def _build_aime_prompt(question: str, model_family: str) -> str:
    # s1 paper QUERY_TEMPLATE = "{Question}" (raw, no extra instruction)
    return _build_prompt(question, model_family)


def _build_gpqa_prompt(question: str, choice_a: str, choice_b: str,
                       choice_c: str, choice_d: str, model_family: str) -> str:
    # Matches s1 fork's utils.doc_to_text_gpqa
    body = (
        f"{question}\n\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}"
    )
    return _build_prompt(body, model_family)


def _shuffle_gpqa_choices(row: dict, seed: int) -> tuple[list[str], str]:
    """Return (4 choices in A/B/C/D order, gold_letter)."""
    correct = str(row["Correct Answer"]).strip()
    incorrects = [str(row[f"Incorrect Answer {i}"]).strip() for i in (1, 2, 3)]
    rng = random.Random(seed)
    choices = [correct] + incorrects
    rng.shuffle(choices)
    gold_idx = choices.index(correct)
    return choices, "ABCD"[gold_idx]


_GPQA_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_gpqa_letter(text: str) -> Optional[str]:
    """Try \\boxed{X}, then 'Answer: X', then last bare A-D in last 200 chars."""
    boxed = _last_boxed_only_string(text)
    if boxed is not None:
        inner = _remove_boxed(boxed)
        if inner:
            m = _GPQA_LETTER_RE.search(inner.upper())
            if m:
                return m.group(1)
    # 'Final Answer: X' / 'Answer: X'
    m = re.search(r"(?i)(?:final\s+)?answer\s*[:\-]\s*\(?([A-D])\)?", text)
    if m:
        return m.group(1).upper()
    # last A-D in the tail
    tail = text[-400:]
    matches = list(_GPQA_LETTER_RE.finditer(tail.upper()))
    if matches:
        return matches[-1].group(1)
    return None


def _summary_path(out_dir: Path, bench: str) -> Path:
    return out_dir / f"{bench}.summary.json"


def _build_math_vllm(ckpt_path: str, gpu_memory_utilization: float = 0.85):
    from vllm import LLM
    return LLM(
        model=ckpt_path,
        tokenizer=ckpt_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=False,
    )


def _run_math_task_nk(
    *,
    llm,
    bench: str,
    out_dir: Path,
    progress: Progress,
    ckpt_path: str,
    prompts: list[str],
    rows_meta: list[dict],  # per-prompt {gold, ...}
    extract_and_score_fn,   # (text, meta) -> (extracted, match_0_or_1)
    n_samples: int,
    temperature: float,
    max_new_tokens: int = 32768,
    seed: int = 7,
    label: str = "",
) -> dict:
    """Generic AIME/GPQA n=K sampled scorer. Returns summary dict."""
    from vllm import SamplingParams
    progress.start(bench)
    t0 = _now()
    sp = SamplingParams(
        n=n_samples,
        max_tokens=max_new_tokens,
        min_tokens=0,
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        skip_special_tokens=False,
    )
    print(f"[{bench}] {label} generating {len(prompts)} × n={n_samples} ...", flush=True)
    outs = llm.generate(prompts, sampling_params=sp)
    print(f"[{bench}] gen done wall={_human_dt(_now()-t0)}", flush=True)

    rows = []
    per_problem_acc = []
    n_truncated = n_v3_a = n_v3_b = 0
    for i, (meta, o) in enumerate(zip(rows_meta, outs)):
        per_sample = []
        for k, out_k in enumerate(o.outputs):
            text = out_k.text
            finish = out_k.finish_reason
            if finish == "length":
                n_truncated += 1
            if V3_MARKERS[0] in text:
                n_v3_a += 1
            if V3_MARKERS[1] in text:
                n_v3_b += 1
            extracted, ok = extract_and_score_fn(text, meta)
            per_sample.append({
                "sample_idx": k,
                "pred_text": text,
                "extracted_pred": extracted,
                "answer_match": float(ok),
                "finish_reason": finish,
                "n_output_tokens": len(out_k.token_ids),
            })
        mean_acc = sum(s["answer_match"] for s in per_sample) / max(len(per_sample), 1)
        per_problem_acc.append(mean_acc)
        rows.append({
            "idx": i,
            "gold": meta.get("gold"),
            "n_samples": len(per_sample),
            "answer_match_mean": mean_acc,
            **{k: v for k, v in meta.items() if k != "gold"},
            "samples": per_sample,
        })

    n = len(rows)
    total_completions = n * n_samples
    summary = {
        "benchmark": bench,
        "ckpt_path": ckpt_path,
        "protocol": f"TIA n={n_samples} T={temperature} averaged",
        "n_problems": n,
        "n_samples_per_problem": n_samples,
        "total_completions": total_completions,
        "accuracy": sum(per_problem_acc) / max(n, 1),
        "n_truncated_completions": n_truncated,
        "truncation_rate": n_truncated / max(total_completions, 1),
        "v3_reasoning_trace_rate": n_v3_a / max(total_completions, 1),
        "v3_final_answer_rate": n_v3_b / max(total_completions, 1),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "wall_seconds": _now() - t0,
    }
    _write_json(_summary_path(out_dir, bench), summary)
    _append_jsonl(out_dir / f"{bench}.rows.jsonl", rows)
    progress.finish(bench, summary["accuracy"])
    print(
        f"[{bench}] acc={summary['accuracy']:.4f} "
        f"completions={total_completions} trunc={summary['truncation_rate']:.3f} "
        f"wall={_human_dt(_now()-t0)}",
        flush=True,
    )
    return summary


def run_math_block(
    *,
    ckpt_path: str,
    out_dir: Path,
    progress: Progress,
    model_family: str,
    benches: list[str] | None = None,
    gpu_memory_utilization: float = 0.85,
    jee_n_samples: int = 6,
    jee_math_only: bool = False,
    aime_n_samples: int = 3,
    math500_n_samples: int = 3,
    math500_max_tokens: int = 32768,
) -> dict:
    """AIME24+AIME25+(GPQA)+(MATH500)+JEE under TIA protocol, sharing ONE vLLM boot.

    `benches` is an explicit list of which math benches to run. Supported:
    ``aime24``, ``aime25``, ``gpqa_diamond``, ``math500``, ``jeebench``.
    `jee_n_samples` lets callers downsize the JEE n. If
    `jee_math_only=True`, JEEbench is pre-filtered to ``subject=="math"`` rows
    (~236/515), shrinking eval wall ~2x for math-focused distill students.
    """
    from datasets import load_dataset

    if benches is None:
        benches = ["aime24", "aime25", "gpqa_diamond", "jeebench"]
    benches = set(benches)

    out: dict[str, Any] = {}

    # Idempotency: if a given bench already has a valid summary on the volume,
    # skip recomputing it. The runner is deterministic (seed=7, T fixed) so
    # re-running produces identical metrics — skipping costs no fidelity and
    # saves the per-bench wall time. Done benches' metrics are loaded back into
    # `out` and the progress file so downstream consumers see the same state.
    def _load_existing(bench: str, metric_key: str) -> bool:
        p = _summary_path(out_dir, bench)
        if not p.exists():
            return False
        try:
            prev = json.loads(p.read_text())
        except Exception:
            return False
        v = prev.get(metric_key)
        if not isinstance(v, (int, float)):
            return False
        print(f"[{bench}] SKIP — existing summary {metric_key}={v:.4f}", flush=True)
        out[bench] = v
        progress.finish(bench, v)
        return True

    skip_aime24 = "aime24" in benches and _load_existing("aime24", "accuracy")
    skip_aime25 = "aime25" in benches and _load_existing("aime25", "accuracy")
    skip_gpqa   = "gpqa_diamond" in benches and _load_existing("gpqa_diamond", "accuracy")
    skip_math500 = "math500" in benches and _load_existing("math500", "accuracy")
    skip_jee    = "jeebench" in benches and _load_existing("jeebench", "answer_match")

    needs_llm = (
        ("aime24" in benches and not skip_aime24) or
        ("aime25" in benches and not skip_aime25) or
        ("gpqa_diamond" in benches and not skip_gpqa) or
        ("math500" in benches and not skip_math500) or
        ("jeebench" in benches and not skip_jee)
    )
    if not needs_llm:
        print(f"[math_block] all requested benches already have summaries — skipping vLLM boot", flush=True)
        return out

    print(f"[math_block] booting vLLM ckpt={ckpt_path}", flush=True)
    llm = _build_math_vllm(ckpt_path, gpu_memory_utilization=gpu_memory_utilization)

    # ---- AIME24 -----------------------------------------------------------
    if "aime24" in benches and not skip_aime24:
        ds = load_dataset("simplescaling/aime24_nofigures", split="train")
        prompts = [_build_aime_prompt(_aime_question_field(r), model_family) for r in ds]
        rows_meta = [{"gold": _aime_answer_field(r)} for r in ds]

        def _score_aime(text, meta):
            extracted = _remove_boxed(_last_boxed_only_string(text))
            if extracted is None:
                return None, 0
            return extracted, int(_math_verify_equiv(meta["gold"], extracted))

        try:
            out["aime24"] = _run_math_task_nk(
                llm=llm, bench="aime24", out_dir=out_dir, progress=progress,
                ckpt_path=ckpt_path, prompts=prompts, rows_meta=rows_meta,
                extract_and_score_fn=_score_aime, n_samples=aime_n_samples, temperature=0.5,
                label="aime24_nofigures",
            )["accuracy"]
        except Exception as exc:  # noqa: BLE001
            progress.fail("aime24", str(exc))
            print(f"[aime24] failed: {exc}", flush=True)

    # ---- AIME25 -----------------------------------------------------------
    if "aime25" in benches and not skip_aime25:
        ds = load_dataset("TIGER-Lab/AIME25", split="train")
        prompts = [_build_aime_prompt(_aime_question_field(r), model_family) for r in ds]
        rows_meta = [{"gold": _aime_answer_field(r)} for r in ds]

        def _score_aime25(text, meta):
            extracted = _remove_boxed(_last_boxed_only_string(text))
            if extracted is None:
                return None, 0
            return extracted, int(_math_verify_equiv(meta["gold"], extracted))

        try:
            out["aime25"] = _run_math_task_nk(
                llm=llm, bench="aime25", out_dir=out_dir, progress=progress,
                ckpt_path=ckpt_path, prompts=prompts, rows_meta=rows_meta,
                extract_and_score_fn=_score_aime25, n_samples=aime_n_samples, temperature=0.5,
                label="aime25_nofigures",
            )["accuracy"]
        except Exception as exc:  # noqa: BLE001
            progress.fail("aime25", str(exc))
            print(f"[aime25] failed: {exc}", flush=True)

    # ---- MATH500 ----------------------------------------------------------
    if "math500" in benches and not skip_math500:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompts = []
        rows_meta = []
        for row in ds:
            q = row.get("problem") or row.get("question")
            a = row.get("answer")
            if q is None or a is None:
                continue
            prompts.append(_build_aime_prompt(str(q), model_family))
            rows_meta.append({"gold": str(a), "subject": row.get("subject"), "level": row.get("level")})

        def _score_math500(text, meta):
            extracted = _remove_boxed(_last_boxed_only_string(text))
            if extracted is None:
                return None, 0
            return extracted, int(_math_verify_equiv(meta["gold"], extracted))

        try:
            out["math500"] = _run_math_task_nk(
                llm=llm, bench="math500", out_dir=out_dir, progress=progress,
                ckpt_path=ckpt_path, prompts=prompts, rows_meta=rows_meta,
                extract_and_score_fn=_score_math500, n_samples=math500_n_samples, temperature=0.5,
                max_new_tokens=math500_max_tokens,
                label="math500_test",
            )["accuracy"]
        except Exception as exc:  # noqa: BLE001
            progress.fail("math500", str(exc))
            print(f"[math500] failed: {exc}", flush=True)

    # ---- GPQA-diamond -----------------------------------------------------
    if "gpqa_diamond" in benches and not skip_gpqa:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        prompts = []
        rows_meta = []
        for idx, row in enumerate(ds):
            choices, gold_letter = _shuffle_gpqa_choices(row, seed=42 + idx)
            prompts.append(_build_gpqa_prompt(
                row["Question"], *choices, model_family=model_family,
            ))
            rows_meta.append({
                "gold": gold_letter,
                "choices": choices,
                "subdomain": row.get("Subdomain"),
                "domain": row.get("High-level domain"),
            })

        def _score_gpqa(text, meta):
            letter = _extract_gpqa_letter(text)
            if letter is None:
                return None, 0
            return letter, int(letter == meta["gold"])

        try:
            out["gpqa_diamond"] = _run_math_task_nk(
                llm=llm, bench="gpqa_diamond", out_dir=out_dir, progress=progress,
                ckpt_path=ckpt_path, prompts=prompts, rows_meta=rows_meta,
                extract_and_score_fn=_score_gpqa, n_samples=3, temperature=0.5,
                label="gpqa_diamond",
            )["accuracy"]
        except Exception as exc:  # noqa: BLE001
            progress.fail("gpqa_diamond", str(exc))
            print(f"[gpqa_diamond] failed: {exc}", flush=True)

    # ---- JEEBench (uses existing run_jeebench but with the shared llm) ----
    if "jeebench" in benches and not skip_jee:
        try:
            jee_summary = _run_jeebench_with_llm(
                llm=llm,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                progress=progress,
                model_family=model_family,
                n_samples=jee_n_samples,
                math_only=jee_math_only,
            )
        except Exception as exc:  # noqa: BLE001
            progress.fail("jeebench", str(exc))
            print(f"[jeebench] failed: {exc}", flush=True)

    # Free vLLM before LCB might boot in same container.
    try:
        del llm
    except Exception:
        pass
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return out


def _run_jeebench_with_llm(
    *,
    llm,
    ckpt_path: str,
    out_dir: Path,
    progress: Progress,
    model_family: str,
    n_samples: int = 6,
    temperature: float = 0.5,
    max_new_tokens: int = 32768,
    math_only: bool = False,
) -> dict:
    """JEEBench using an already-booted vLLM instance (shared with math block).

    If math_only=True, pre-filters dataset to subject=="math" rows (~236/515).
    """
    from datasets import load_dataset
    from vllm import SamplingParams

    sys.path.insert(0, "/workspace")
    from _common.scorers.jeebench import JEEBenchScorer  # type: ignore

    progress.start("jeebench")
    t0 = _now()
    print(f"[jeebench] loading dataset daman1209arora/jeebench (n={n_samples} T={temperature} math_only={math_only})", flush=True)
    ds = load_dataset("daman1209arora/jeebench", split="test")
    if math_only:
        ds = ds.filter(lambda r: (r.get("subject") or "").lower() == "math")
    print(f"[jeebench] n={len(ds)} rows × {n_samples} = {len(ds)*n_samples} completions", flush=True)

    prompts = [_build_prompt(_format_jeebench_question(r), model_family) for r in ds]

    sp = SamplingParams(
        n=n_samples,
        max_tokens=max_new_tokens,
        min_tokens=0,
        temperature=temperature,
        top_p=0.95,
        seed=7,
        skip_special_tokens=False,
    )
    outs = llm.generate(prompts, sampling_params=sp)

    scorer = JEEBenchScorer()
    rows = []
    per_problem_strict_mean = []
    per_problem_partial_mean = []
    n_truncated = n_v3_a = n_v3_b = 0
    for i, (row, o) in enumerate(zip(ds, outs)):
        per_sample = []
        for k, out_k in enumerate(o.outputs):
            text = out_k.text
            finish = out_k.finish_reason
            if finish == "length":
                n_truncated += 1
            meta = {"type": row.get("type", "")}
            res = scorer.score(text, str(row.get("gold", "")), meta=meta)
            if V3_MARKERS[0] in text:
                n_v3_a += 1
            if V3_MARKERS[1] in text:
                n_v3_b += 1
            per_sample.append({
                "sample_idx": k,
                "pred_text": text,
                "extracted_pred": res.extracted_pred,
                "answer_match": float(res.answer_match),
                "answer_match_partial": float(res.answer_match_partial),
                "finish_reason": finish,
                "n_output_tokens": len(out_k.token_ids),
            })
        per_problem_strict_mean.append(sum(s["answer_match"] for s in per_sample) / len(per_sample))
        per_problem_partial_mean.append(sum(s["answer_match_partial"] for s in per_sample) / len(per_sample))
        rows.append({
            "idx": i,
            "type": row.get("type"),
            "subject": row.get("subject"),
            "gold": row.get("gold"),
            "answer_match_mean": per_problem_strict_mean[-1],
            "answer_match_partial_mean": per_problem_partial_mean[-1],
            "n_samples": len(per_sample),
            "samples": per_sample,
        })

    n = len(rows)
    total_completions = n * n_samples
    summary = {
        "benchmark": "jeebench",
        "ckpt_path": ckpt_path,
        "protocol": f"TIA n={n_samples} T={temperature} averaged",
        "n_problems": n,
        "n_samples_per_problem": n_samples,
        "total_completions": total_completions,
        "answer_match": sum(per_problem_strict_mean) / max(n, 1),
        "answer_match_partial": sum(per_problem_partial_mean) / max(n, 1),
        "n_truncated_completions": n_truncated,
        "truncation_rate": n_truncated / max(total_completions, 1),
        "v3_reasoning_trace_rate": n_v3_a / max(total_completions, 1),
        "v3_final_answer_rate": n_v3_b / max(total_completions, 1),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "wall_seconds": _now() - t0,
    }
    _write_json(out_dir / "jeebench.summary.json", summary)
    _append_jsonl(out_dir / "jeebench.rows.jsonl", rows)
    progress.finish("jeebench", summary["answer_match"])
    print(
        f"[jeebench] strict={summary['answer_match']:.4f} "
        f"partial={summary['answer_match_partial']:.4f} "
        f"completions={total_completions} wall={_human_dt(_now()-t0)}",
        flush=True,
    )
    return summary


# --------------------- JEEBench (custom + official scorer) ---------------------


JEEBENCH_GLOBAL_INSTRUCTION = (
    "Solve the following IIT-JEE problem.\n"
    "Return exactly one <think>...</think> block.\n"
    "After the closing </think> tag, repeat the reasoning once more as plain "
    "text outside the think block.\n"
    "After that repeated plain-text reasoning, write the final answer on a new "
    "line wrapped in \\boxed{}.\n"
    "Do not open a second <think> block.\n"
)

JEEBENCH_PER_TYPE = {
    "MCQ": (
        "Question type: multiple-choice with exactly one correct option.\n"
        "Output format: \\boxed{X} where X is one letter from {A, B, C, D}.\n"
    ),
    "MCQ(multiple)": (
        "Question type: multiple-choice; one or more options can be correct.\n"
        "Output format: \\boxed{XYZ} listing every correct letter from {A, B, C, D}.\n"
    ),
    "Integer": (
        "Question type: the final answer is a non-negative integer.\n"
        "Output format: \\boxed{N} with N a non-negative integer.\n"
    ),
    "Numeric": (
        "Question type: the final answer is a decimal number; give it correct "
        "to the second decimal digit.\n"
        "Output format: \\boxed{N.NN}.\n"
    ),
}


def _format_jeebench_question(row: dict) -> str:
    qtype = row.get("type", "")
    parts = [JEEBENCH_GLOBAL_INSTRUCTION]
    per = JEEBENCH_PER_TYPE.get(qtype)
    if per:
        parts.append(per)
    parts.append("Question:\n")
    parts.append(row["question"])
    return "\n".join(parts)


def run_jeebench(
    ckpt_path: str,
    out_dir: Path,
    progress: Progress,
    model_family: str,
    *,
    max_new_tokens: int = 32768,
    gpu_memory_utilization: float = 0.85,
    n_samples: int = 6,
    temperature: float = 0.5,
) -> dict:
    """JEEBench eval (TIA protocol — n=6 sampled @ T=0.5, averaged).

    For each problem, generate `n_samples` completions and score each one.
    The reported `answer_match` is the per-problem mean over n samples,
    aggregated across problems. Matches the dair-iitd / TIA evaluation
    convention used in the JEEBench paper.
    """
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    sys.path.insert(0, "/workspace")
    from _common.scorers.jeebench import JEEBenchScorer  # type: ignore

    progress.start("jeebench")
    t0 = _now()
    print(f"[jeebench] loading dataset daman1209arora/jeebench (n_samples={n_samples} T={temperature})", flush=True)
    ds = load_dataset("daman1209arora/jeebench", split="test")
    print(f"[jeebench] n={len(ds)} rows × {n_samples} samples = {len(ds)*n_samples} completions", flush=True)

    prompts = [_build_prompt(_format_jeebench_question(r), model_family) for r in ds]

    print(f"[jeebench] booting vLLM ckpt={ckpt_path}", flush=True)
    llm = LLM(
        model=ckpt_path,
        tokenizer=ckpt_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=False,
    )

    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_new_tokens,
        min_tokens=0,
        temperature=temperature,
        top_p=0.95,
        seed=7,
        skip_special_tokens=False,
    )
    print(f"[jeebench] generating ...", flush=True)
    outs = llm.generate(prompts, sampling_params=sampling_params)
    print(f"[jeebench] gen done wall={_human_dt(_now()-t0)}", flush=True)

    scorer = JEEBenchScorer()
    rows = []
    per_problem_strict_mean = []
    per_problem_partial_mean = []
    n_truncated = n_v3_a = n_v3_b = 0
    for i, (row, o) in enumerate(zip(ds, outs)):
        # o.outputs holds n_samples items
        per_sample = []
        for sample_idx, out_k in enumerate(o.outputs):
            text = out_k.text
            finish = out_k.finish_reason
            if finish == "length":
                n_truncated += 1
            meta = {"type": row.get("type", "")}
            res = scorer.score(text, str(row.get("gold", "")), meta=meta)
            if V3_MARKERS[0] in text:
                n_v3_a += 1
            if V3_MARKERS[1] in text:
                n_v3_b += 1
            per_sample.append({
                "sample_idx": sample_idx,
                "pred_text": text,
                "extracted_pred": res.extracted_pred,
                "answer_match": float(res.answer_match),
                "answer_match_partial": float(res.answer_match_partial),
                "finish_reason": finish,
                "n_output_tokens": len(out_k.token_ids),
            })
        strict_mean = sum(s["answer_match"] for s in per_sample) / len(per_sample)
        partial_mean = sum(s["answer_match_partial"] for s in per_sample) / len(per_sample)
        per_problem_strict_mean.append(strict_mean)
        per_problem_partial_mean.append(partial_mean)
        rows.append({
            "idx": i,
            "type": row.get("type"),
            "subject": row.get("subject"),
            "gold": row.get("gold"),
            "extracted_gold": str(row.get("gold", "")).strip(),
            "answer_match_mean": strict_mean,
            "answer_match_partial_mean": partial_mean,
            "n_samples": len(per_sample),
            "samples": per_sample,
        })

    n = len(rows)
    total_completions = n * n_samples
    summary = {
        "benchmark": "jeebench",
        "ckpt_path": ckpt_path,
        "protocol": "TIA n=6 T=0.5 averaged",
        "n_problems": n,
        "n_samples_per_problem": n_samples,
        "total_completions": total_completions,
        "answer_match": sum(per_problem_strict_mean) / max(n, 1),
        "answer_match_partial": sum(per_problem_partial_mean) / max(n, 1),
        "n_truncated_completions": n_truncated,
        "truncation_rate": n_truncated / max(total_completions, 1),
        "v3_reasoning_trace_rate": n_v3_a / max(total_completions, 1),
        "v3_final_answer_rate": n_v3_b / max(total_completions, 1),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "wall_seconds": _now() - t0,
    }
    _write_json(out_dir / "jeebench.summary.json", summary)
    _append_jsonl(out_dir / "jeebench.rows.jsonl", rows)
    progress.finish("jeebench", summary["answer_match"])
    print(
        f"[jeebench] strict={summary['answer_match']:.4f} "
        f"partial={summary['answer_match_partial']:.4f} "
        f"trunc={summary['truncation_rate']:.3f} "
        f"completions={summary['total_completions']} "
        f"wall={_human_dt(_now()-t0)}",
        flush=True,
    )

    # Free vLLM before LCB boot to avoid OOM (LCB boots its own LLM via lcb_runner).
    try:
        del llm
    except Exception:
        pass
    try:
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return summary


# ----------------------------- LiveCodeBench --------------------------------


def run_livecodebench(
    ckpt_path: str,
    out_dir: Path,
    progress: Progress,
    *,
    model_family: str = "qwen25",
    release_version: str = "release_v5",
    start_date: str = "2024-08-01",
    end_date: str = "2025-02-01",
    max_tokens: int = 32768,
    n: int = 3,
    temperature: float = 0.5,
) -> dict:
    """LiveCodeBench eval via official lcb_runner CLI (TIA protocol — n=3, T=0.5).

    Uses the official LCB repo's runner (vendored under /workspace/LiveCodeBench/
    by the Modal image). lcb_runner has a hard-coded `LanguageModelStore` keyed
    by HF model id. We pass the family's base HF id as `--model` (so the
    registry lookup + chat template resolution work) and `--local_model_path`
    so vLLM actually loads weights from the ckpt dir.
    """
    # Idempotency: skip if a successful lcb_v5.summary.json already exists.
    # (A status="failed" placeholder does NOT count as success.)
    existing = out_dir / "lcb_v5.summary.json"
    if existing.exists():
        try:
            prev = json.loads(existing.read_text())
            if isinstance(prev.get("pass_at_1"), (int, float)) and prev.get("status") != "failed":
                v = prev["pass_at_1"]
                print(f"[lcb_v5] SKIP — existing summary pass@1={v:.4f}", flush=True)
                progress.finish("lcb_v5", v)
                return prev
        except Exception:
            pass

    progress.start("lcb_v5")
    t0 = _now()
    lcb_root = Path("/workspace/LiveCodeBench")
    lcb_out = out_dir / "lcb_v5"
    lcb_out.mkdir(parents=True, exist_ok=True)

    # Map family to the lcb_runner LanguageModelStore key + the model_repr
    # lcb_runner uses for its `output/<model_repr>/` directory.
    FAMILY_REGISTRY = {
        "qwen25": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-Ins-7B"),
        "llama31": ("meta-llama/Llama-3.1-8B-Instruct", "LLama3.1-8b-Ins"),
        # NOTE: lcb_runner may not have a Gemma entry; if LCB fails for
        # gemma3 with KeyError on model_repr, drop lcb_v5 from bench_subset
        # for the gemma3 family.
        "gemma3": ("google/gemma-3-4b-it", "Gemma-3-4B-It"),
    }
    if model_family not in FAMILY_REGISTRY:
        raise ValueError(f"unknown model_family={model_family!r}; expected one of {list(FAMILY_REGISTRY)}")
    registered_name, model_label = FAMILY_REGISTRY[model_family]

    import shutil
    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", registered_name,
        "--local_model_path", ckpt_path,
        "--scenario", "codegeneration",
        "--release_version", release_version,
        "--start_date", start_date,
        "--end_date", end_date,
        "--evaluate",
        "--num_process_evaluate", "16",
        "--n", str(n),
        "--temperature", str(temperature),
        "--max_tokens", str(max_tokens),
    ]
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print(f"[lcb] CMD (cwd={lcb_root}): {' '.join(cmd)}", flush=True)
    # Capture stderr to a per-ckpt file so we can debug LCB failures after the
    # modal log buffer rotates. stdout goes straight to the container log.
    stderr_path = lcb_out / "lcb.stderr.log"
    with stderr_path.open("w") as ferr:
        proc = subprocess.run(cmd, env=env, cwd=str(lcb_root), stderr=ferr)
    print(f"[lcb] returncode={proc.returncode} stderr={stderr_path}", flush=True)

    # Copy lcb_runner's default output dir into our results volume.
    lcb_default_output = lcb_root / "output" / model_label
    if lcb_default_output.exists():
        dest = lcb_out / "output" / model_label
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(lcb_default_output, dest)
        print(f"[lcb] copied {lcb_default_output} → {dest}", flush=True)
    if proc.returncode != 0:
        progress.fail("lcb_v5", f"lcb_runner returncode={proc.returncode}")
        summary = {
            "benchmark": "lcb_v5",
            "ckpt_path": ckpt_path,
            "release_version": release_version,
            "window": [start_date, end_date],
            "status": "failed",
            "returncode": proc.returncode,
            "wall_seconds": _now() - t0,
        }
        _write_json(out_dir / "lcb_v5.summary.json", summary)
        return summary

    # Parse lcb_runner output. By default it writes
    #   <cwd>/output/<model_label>/Scenario.codegeneration_<window>_eval_all.json
    # (we cd into lcb_out so output ends up at lcb_out/output/<model_label>/...).
    # The exact filename suffix varies across LCB versions; scan for *_eval*.json.
    eval_files = sorted(
        list(lcb_out.rglob("*_eval_all.json")) + list(lcb_out.rglob("*_eval.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    pass_at_1: Optional[float] = None
    n_problems: Optional[int] = None
    raw_metrics: dict = {}
    if eval_files:
        with eval_files[0].open() as f:
            raw_metrics = json.load(f)
        if isinstance(raw_metrics, dict):
            pass_at_1 = raw_metrics.get("pass@1") or raw_metrics.get("pass_at_1")
            n_problems = raw_metrics.get("n") or raw_metrics.get("num_problems")
        elif isinstance(raw_metrics, list) and raw_metrics:
            # per-problem list; aggregate
            n_problems = len(raw_metrics)
            passed = sum(1 for r in raw_metrics if r.get("pass@1") or r.get("pass_at_1"))
            pass_at_1 = passed / max(n_problems, 1)

    summary = {
        "benchmark": "lcb_v5",
        "ckpt_path": ckpt_path,
        "release_version": release_version,
        "window": [start_date, end_date],
        "n_samples": n_problems,
        "pass_at_1": pass_at_1,
        "raw_metrics_path": str(eval_files[0]) if eval_files else None,
        "max_tokens": max_tokens,
        "wall_seconds": _now() - t0,
    }
    _write_json(out_dir / "lcb_v5.summary.json", summary)
    progress.finish("lcb_v5", pass_at_1)
    print(f"[lcb] pass@1={pass_at_1} n={n_problems} wall={_human_dt(_now()-t0)}", flush=True)
    return summary


# ------------------------------ public entry --------------------------------


def run_all(
    *,
    ckpt_path: str,
    ckpt_label: str,
    run_id: str,
    model_family: str,
    results_root: Path,
    commit_fn=lambda: None,
    lcb_only: bool = False,
    skip_lcb: bool = False,
    aime_only: bool = False,
    lm_eval_tasks: list[str] | None = None,
    bench_subset: list[str] | None = None,
    jee_n_samples: int = 6,
    jee_math_only: bool = False,
    aime_n_samples: int = 3,
    math500_n_samples: int = 3,
    math500_max_tokens: int = 32768,
) -> dict:
    """Top-level driver. Defaults: AIME24+AIME25 → GPQA → JEE → LCB.

    Mode flags (mutually compatible with caller's intent):
    * `lcb_only=True` — run only LCB.
    * `skip_lcb=True` — skip LCB (keeps AIME + JEE).
    * `aime_only=True` — only the lm-eval block (AIME24+AIME25 by default, or
      whatever `lm_eval_tasks` overrides to — e.g. add gpqa_diamond_openai when
      OPENAI_API_KEY is available). Skips JEE + LCB.
    * `lm_eval_tasks` — override the lm-eval CLI task list. Default
      `["aime24_nofigures", "aime25_nofigures", "gpqa_diamond_openai"]`.
    * `bench_subset` — explicit per-bench filter; overrides the default
      bench list. Example: `bench_subset=["jeebench"]` + `jee_n_samples=1`
      runs ONLY a quick JEE n=1 estimate (used to fill in scores fast while
      the full n=6 sweep is still in flight). Auto-skips LCB unless
      `"lcb_v5"` is in the subset.
    * `jee_n_samples` — override JEE n (default 6 per TIA protocol).
    """
    out_dir = Path(results_root) / run_id / ckpt_label
    out_dir.mkdir(parents=True, exist_ok=True)

    record = ProgressRecord(
        run_id=run_id,
        ckpt_label=ckpt_label,
        ckpt_path=ckpt_path,
        model_family=model_family,
        started_at=_now(),
        benchmarks={},
    )
    # Default lm-eval task list. GPQA requires HF access to gated dataset
    # Idavidrein/gpqa — user is requesting access. Once granted, this default
    # will be effective on next deploy.
    if lm_eval_tasks is None:
        lm_eval_tasks = ["aime24_nofigures", "aime25_nofigures", "gpqa_diamond_openai"]
    progress = Progress(out_dir, record, commit_fn=commit_fn)
    print(
        f"[runner] start run_id={run_id} ckpt={ckpt_label} family={model_family} "
        f"aime_only={aime_only} lcb_only={lcb_only} skip_lcb={skip_lcb} "
        f"lm_eval_tasks={lm_eval_tasks}",
        flush=True,
    )

    results: dict[str, Any] = {}

    # bench_subset overrides the standard mode flags for fine-grained control.
    if bench_subset is not None:
        math_benches = [b for b in bench_subset if b != "lcb_v5"]
        run_lcb = "lcb_v5" in bench_subset
    else:
        if lcb_only:
            math_benches = []
            run_lcb = True
        else:
            math_benches = ["aime24", "aime25", "gpqa_diamond"]
            if not aime_only:
                math_benches.append("jeebench")
            run_lcb = not (skip_lcb or aime_only)

    if math_benches:
        try:
            math_out = run_math_block(
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                progress=progress,
                model_family=model_family,
                benches=math_benches,
                jee_n_samples=jee_n_samples,
                jee_math_only=jee_math_only,
                aime_n_samples=aime_n_samples,
                math500_n_samples=math500_n_samples,
                math500_max_tokens=math500_max_tokens,
            )
            results.update(math_out)
        except Exception as exc:  # noqa: BLE001
            print(f"[runner] math_block failed: {exc}", flush=True)

    # ---- LiveCodeBench (separate vLLM boot via lcb_runner)
    if run_lcb:
        try:
            run_livecodebench(
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                progress=progress,
                model_family=model_family,
            )
        except Exception as exc:  # noqa: BLE001
            progress.fail("lcb_v5", str(exc))
            print(f"[runner] lcb failed: {exc}", flush=True)

    print(f"[runner] done. progress: {progress.path}", flush=True)
    return results
