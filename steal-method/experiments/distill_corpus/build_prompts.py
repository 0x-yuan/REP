"""REP distillation corpus — step 1: render the V3 k=3 prefix once + index the
10k OpenThoughts distillation questions (paper §5.5, Table 5).

The 10k query set shares ONE ~42k-token REP prefix (3 shadow demonstrations
wrapped in the markdown-fence wrapper V3 under the deployed defender system
prompt). We render that prefix exactly once, split it at the target-question
slot, and store the per-question token counts, so the inbox assembler can
concatenate ``prefix + question + post_marker`` without re-rendering.

Outputs (under ``--out``, default ``./prompts``)::

    cells/<cell_id>.json         prefix_text / post_marker + demo provenance
    queries/<victim>.jsonl       {prompt_id, source_index, question, n_query_tokens}
    preview/preview_row0.txt     full rendered prompt for row 0
    manifest.json                summary stats + token-budget audit

Question source (``--questions``): a JSONL (optionally gzipped) with
``{prompt_id, source_index, question}`` rows. Default = the vendored
``public/data/openthoughts_10k/questions.jsonl.gz`` (10 000 math rows,
OpenThoughts-114k source_index 20000..29999); alternatively the disjoint
top-up sample written by ``sample_questions.py``.

Prerequisite: the vendored shadow pool ``public/data/shot_pool/qwen3_14b.jsonl.gz``
(see ``rep_core/prepare_shot_pool.py``). Shots are the first
``k`` rows of the deterministic seed-7 50-row sample (same rule as every other
REP builder).

Run::

    python build_prompts.py                                   # vendored 10k questions
    python build_prompts.py --questions prompts/queries/topup_25k.jsonl --tag topup
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXP_DIR = _HERE.parent
STEAL_METHOD = _HERE.parents[2]
sys.path.insert(0, str(STEAL_METHOD / "experiments"))

import _common as C  # noqa: E402
from rep_core.variants import render_user_content  # noqa: E402
from rep_core.build_helpers import hint_for  # noqa: E402

# --- Pinned configuration (paper default: V3 markdown_fence, k=3) ------------
VICTIM = "qwen3_14b"
SHOT_POOL_VICTIM = "qwen3_14b"        # shadow model = Qwen3-14B for every victim
CELL_WRAP = "V3"
CELL_K = 3
QUESTIONS_DEFAULT = _HERE.parents[3] / "data" / "openthoughts_10k" / "questions.jsonl.gz"
MAX_INPUT_TOKENS = 125_000            # YaRN-131k context minus output slack

# Unique marker at the target-question slot so prefix / post-marker can be
# split once without per-row re-tokenisation.
MARKER = "\x00DISTILL_QUERY_MARKER\x00"

# Prefix sanity markers: every rendered prompt must contain these.
COMMON_MARKERS = [
    "<|im_start|>system", "/think", "You are a careful math reasoning assistant.",
    "<|im_start|>user", "Response: <think>",
    "Now solve the following question using the same format",
    "Output format: write the final answer wrapped in \\boxed{}",
    "Question:\n", "<|im_start|>assistant",
]
WRAP_MARKERS = {
    "V3": ["```bash", "$ cat reasoning_trace.txt", "$ cat final_answer.txt"],
}


def cell_id_for(victim: str, k: int, wrap: str) -> str:
    return f"distill_C_{victim}_{SHOT_POOL_VICTIM}_K{k}_{wrap}"


def load_questions(src: str) -> list[dict]:
    """Local JSONL (or .jsonl.gz) with {prompt_id, source_index, question} rows."""
    p = Path(src)
    rows = []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        lines = f.read().splitlines()
    for ln in lines:
        if ln.strip():
            r = json.loads(ln)
            rows.append({"prompt_id": r["prompt_id"], "source_index": int(r["source_index"]),
                         "question": r["question"]})
    print(f"[build] loaded {len(rows)} questions from {p}", flush=True)
    return rows


def build_prefix(demos: list[dict], tok, *, victim: str, k: int, wrap: str) -> dict:
    """Render the REP body with the marker as the target question, wrap it in
    the chat template, then split into prefix / post-marker halves."""
    if len(demos) < k:
        raise SystemExit(f"need {k} demos, only have {len(demos)}")
    used = demos[:k]
    body = render_user_content(
        demos=used, target_q=MARKER, bench="math500", wrap=wrap,
        target_meta=None, extra_hint=hint_for("math500", None),
    )
    full = C.render_chat(tok, C.build_defender_system(), body,
                         enable_thinking=True, add_generation_prompt=True)
    if full.count(MARKER) != 1:
        raise SystemExit(f"marker present {full.count(MARKER)} times; expected 1")
    prefix_text, post_marker = full.split(MARKER)
    return {
        "cell_id": cell_id_for(victim, k, wrap),
        "wrap": wrap,
        "K": k,
        "prefix_text": prefix_text,
        "post_marker": post_marker,
        "n_prefix_tokens": len(tok(prefix_text, add_special_tokens=False).input_ids),
        "n_post_marker_tokens": len(tok(post_marker, add_special_tokens=False).input_ids),
        "demos": [{"slot": i, "shot_id": d.get("src_idx"), "trace_chars": len(d["r"]),
                   "answer_repr": str(d["a"])[:120]} for i, d in enumerate(used)],
        "shot_pool_victim": SHOT_POOL_VICTIM,
        "shot_pool_seed": C.SHOT_POOL_SEED,
        "shot_pool_size": C.SHOT_POOL_SIZE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(QUESTIONS_DEFAULT),
                    help="JSONL(.gz) of {prompt_id, source_index, question} "
                         "(default: vendored data/openthoughts_10k/questions.jsonl.gz)")
    ap.add_argument("--tag", default=VICTIM,
                    help="basename of the queries file (default: %(default)s)")
    ap.add_argument("--victim", default=VICTIM, help="victim label stamped into cell_id")
    ap.add_argument("--k", type=int, default=CELL_K, help="shots (default: %(default)s)")
    ap.add_argument("--wrap", default=CELL_WRAP, help="wrapper id (default: %(default)s)")
    ap.add_argument("--shot-pool", type=Path, default=C.SHOT_POOL_LOCAL,
                    help="vendored shadow pool jsonl.gz")
    ap.add_argument("--out", type=Path, default=EXP_DIR / "prompts")
    ap.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS)
    args = ap.parse_args()

    cells, queries, preview = args.out / "cells", args.out / "queries", args.out / "preview"
    for d in (cells, queries, preview):
        d.mkdir(parents=True, exist_ok=True)

    tok = C.make_tokenizer()
    demos = C.load_ot_shot_pool(args.shot_pool)
    print(f"[build] demos[:{args.k}] of the seed-{C.SHOT_POOL_SEED} sample:", flush=True)
    for i, d in enumerate(demos[:args.k]):
        print(f"  demo[{i}] id={d.get('src_idx')} trace_chars={len(d['r'])}", flush=True)

    cell = build_prefix(demos, tok, victim=args.victim, k=args.k, wrap=args.wrap)
    cell_path = cells / f"{cell['cell_id']}.json"
    cell_path.write_text(json.dumps(cell, indent=2, ensure_ascii=False) + "\n")
    print(f"[build] cell prefix -> {cell_path}  prefix_tok={cell['n_prefix_tokens']} "
          f"post_marker_tok={cell['n_post_marker_tokens']}", flush=True)

    rows = load_questions(args.questions)
    for r in rows:
        r["n_query_tokens"] = len(tok(r["question"], add_special_tokens=False).input_ids)
    q_path = queries / f"{args.tag}.jsonl"
    with q_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    max_q = max(r["n_query_tokens"] for r in rows)
    n_pref, n_post = cell["n_prefix_tokens"], cell["n_post_marker_tokens"]
    over = sum(1 for r in rows if n_pref + r["n_query_tokens"] + n_post > args.max_input_tokens)
    print(f"[build] queries -> {q_path} rows={len(rows)} max_query_tok={max_q} "
          f"over_budget_rows={over}", flush=True)

    rendered = cell["prefix_text"] + rows[0]["question"] + cell["post_marker"]
    (preview / "preview_row0.txt").write_text(rendered)
    missing = [m for m in COMMON_MARKERS + WRAP_MARKERS.get(args.wrap, []) if m not in rendered]
    if missing:
        raise SystemExit(f"[build] PREFIX SANITY FAILED — missing: {missing}")
    print("[build] prefix sanity OK", flush=True)

    manifest = {
        "cell_id": cell["cell_id"], "victim": args.victim, "K": args.k, "wrap": args.wrap,
        "questions": args.questions, "n_rows": len(rows),
        "n_prefix_tokens": n_pref, "n_post_marker_tokens": n_post,
        "max_query_tokens": max_q, "max_total_tokens": n_pref + max_q + n_post,
        "max_input_tokens_budget": args.max_input_tokens, "n_over_budget_rows": over,
        "demos": cell["demos"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[build] manifest -> {args.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
