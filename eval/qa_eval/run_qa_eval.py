"""Run the 3-task QA utility eval (StrategyQA / ProntoQA / HotpotQA).

Two input modes:

  (1) --generations DIR   score pre-computed generations. DIR holds
                          ``<bench>.jsonl`` files whose rows carry the eval-set
                          fields plus ``"output"`` (raw model text).
  (2) --model HF_ID_OR_PATH  generate locally with vLLM (greedy, chat template,
                          ``BASE_SYS`` system prompt, max 2048 new tokens) on
                          the eval sets in --eval-dir, then score. Generations
                          are written to --out/<bench>.jsonl so mode (1) can
                          re-score them later.

Modal is optional: ``modal_run.py`` next to this file wraps mode (2) in a
Modal GPU function; nothing here imports modal.

Examples:

    uv run python run_qa_eval.py --generations runs/base --out runs/base
    uv run --with vllm python run_qa_eval.py --model Qwen/Qwen2.5-7B-Instruct \\
        --eval-dir data --out runs/base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from qa_eval.scoring import BASE_SYS, BENCHES, aggregate, score_row  # noqa: E402

MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 2048


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def generate_vllm(model: str, eval_dir: Path, out_dir: Path, benches, tp: int,
                  gpu_mem: float) -> dict[str, list[dict]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=tp, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=gpu_mem)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    out_dir.mkdir(parents=True, exist_ok=True)
    gens: dict[str, list[dict]] = {}
    for bench in benches:
        rows = load_jsonl(eval_dir / f"{bench}_eval.jsonl")
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": BASE_SYS}, {"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True) for r in rows]
        outs = llm.generate(prompts, sp)
        gens[bench] = [{**r, "output": o.outputs[0].text} for r, o in zip(rows, outs)]
        (out_dir / f"{bench}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in gens[bench]) + "\n")
        print(f"[gen] {bench}: {len(rows)} rows -> {out_dir / f'{bench}.jsonl'}", flush=True)
    return gens


def score_all(gens: dict[str, list[dict]], label: str, model: str | None) -> dict:
    summary = {"label": label, "model": model, "benches": {}}
    for bench, rows in gens.items():
        summary["benches"][bench] = aggregate(rows, bench)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--generations", help="dir with <bench>.jsonl generation files")
    src.add_argument("--model", help="HF model id / local path, generated with vLLM")
    ap.add_argument("--eval-dir", default=str(_HERE / "data"),
                    help="dir with <bench>_eval.jsonl (from build_eval_sets.py)")
    ap.add_argument("--out", default=None, help="write generations + qa.json here")
    ap.add_argument("--label", default=None)
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--per-row", action="store_true", help="also write scored_<bench>.jsonl")
    args = ap.parse_args()

    benches = [b for b in args.benches.split(",") if b]
    label = args.label or (args.model or args.generations).rstrip("/").split("/")[-1]

    if args.generations:
        gdir = Path(args.generations)
        gens = {b: load_jsonl(gdir / f"{b}.jsonl") for b in benches if (gdir / f"{b}.jsonl").exists()}
        missing = [b for b in benches if b not in gens]
        if missing:
            print(f"[warn] no generations for: {missing}", file=sys.stderr)
    else:
        out_dir = Path(args.out or f"runs/{label}")
        gens = generate_vllm(args.model, Path(args.eval_dir), out_dir, benches, args.tp, args.gpu_mem)

    summary = score_all(gens, label, args.model)
    print(json.dumps(summary, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "qa.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.per_row:
            for bench, rows in gens.items():
                (out_dir / f"scored_{bench}.jsonl").write_text("\n".join(
                    json.dumps({"id": r.get("id"), "answer": r.get("answer"), **score_row(r)})
                    for r in rows) + "\n")
        print(f"wrote {out_dir / 'qa.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
