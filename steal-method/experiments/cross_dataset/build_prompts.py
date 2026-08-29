"""Cross-dataset transfer — does the demo pool need to match the target dataset?

Reproduces the paper "Cross-dataset transfer" table. Fixes the default REP
configuration (V3 markdown_fence, K=3, victim Qwen3-14B) and varies only the
SHADOW DEMONSTRATION SOURCE: OpenThoughts (in-domain control), MATH500, GSM8K,
and JEEBench. Targets are always the 500-example OpenThoughts-114k subset, so
any lift over the no-trigger baseline that survives an off-domain demo pool
shows REP is not pure in-domain memorization.

Data: all four shadow demo pools are vendored under ``public/data/shot_pool/``
and load offline:

  * ``qwen3_14b.jsonl.gz``            — OpenThoughts (in-domain control)
  * ``math500_qwen3_14b.jsonl.gz``    — MATH500 shadow demos
  * ``gsm8k_qwen3_14b.jsonl.gz``      — GSM8K shadow demos
  * ``jeebench_qwen3_14b.jsonl.gz``   — JEEBench shadow demos

Each off-domain file is the *resolved* 50-row pool used for the paper: a
Qwen3-14B shadow baseline harvested on that dataset, then
``random.Random(7).sample(valid_rows, 50)`` applied once; rows are stored in
sample order as ``{q, r, a, src_idx}``, so the builder slices ``[:k]`` directly
(re-sampling would change the demos). A raw harvested baseline
(``{question, ri, answer, idx, valid}``) is also accepted and sampled on the fly
with the same seed.

Run::

    python build_prompts.py --help
    python build_prompts.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXP_DIR = _HERE.parent
STEAL_METHOD = _HERE.parents[2]
sys.path.insert(0, str(STEAL_METHOD / "experiments"))

import _common as C  # noqa: E402
from rep_core.variants import render_user_content as render_c  # noqa: E402
from rep_core.baseline import baseline_r_instruction  # noqa: E402
from rep_core.build_helpers import hint_for, RENDER_MODEL_ID  # noqa: E402

INBOX_DIR = STEAL_METHOD / "inference-farm" / "inbox"
DATA_DIR = EXP_DIR / "data"
SHOT_DIR = C.DATA_ROOT / "shot_pool"   # vendored under public/data/shot_pool/

WRAP = "V3"
K = 3
VICTIM = "qwen3-14b"
SHOT_TEACHER = "qwen3_14b"
MAX_NEW_TOKENS = 80000
MAX_INPUT_TOKENS = 128_000

# Off-domain shadow pools (harvested Qwen3-14B baselines); the OpenThoughts pool
# is the pre-sampled in-domain control shared with config_sweep.
OT_SHOT_POOL_PATH = C.SHOT_POOL_LOCAL   # vendored under public/data/shot_pool/
CROSS_SHOT_PATHS = {
    "math500":  SHOT_DIR / "math500_qwen3_14b.jsonl.gz",
    "gsm8k":    SHOT_DIR / "gsm8k_qwen3_14b.jsonl.gz",
    "jeebench": SHOT_DIR / "jeebench_qwen3_14b.jsonl.gz",
}
CELL = {
    "openthoughts": "OT-V3-K3", "math500": "MATH-V3-K3",
    "gsm8k": "GSM-V3-K3", "jeebench": "JEE-V3-K3",
}
BATCH = {
    "openthoughts": "ot_V3_K3", "math500": "math_V3_K3",
    "gsm8k": "gsm_V3_K3", "jeebench": "jee_V3_K3",
}


def load_shots(source: str, path: Path | None = None) -> list[dict]:
    """Deterministic 50-row shadow demo pool as {q,r,a} triples.

    Two on-disk schemas are accepted for the off-domain pools:

    * **resolved pool** (vendored default): rows ``{q, r, a, src_idx}`` already
      in seed-7 sample order -> returned as-is (first 50 rows).
    * **raw harvested baseline**: rows ``{question, ri, answer, idx, valid}`` ->
      filtered to valid/non-empty rows and sampled with
      ``random.Random(7).sample(rows, 50)`` (the original recipe).
    """
    if source == "openthoughts":
        return C.load_ot_shot_pool(path or OT_SHOT_POOL_PATH)
    rows = C.load_jsonl_gz(path or CROSS_SHOT_PATHS[source])
    if rows and "q" in rows[0] and "r" in rows[0]:
        pool = [r for r in rows if r.get("q") and r.get("r") and r.get("a")]
        if len(pool) < C.SHOT_POOL_SIZE:
            raise SystemExit(f"[build] {source}: only {len(pool)} valid rows (<{C.SHOT_POOL_SIZE})")
        return [{"q": r["q"], "r": r["r"], "a": r["a"], "src_idx": r.get("src_idx")}
                for r in pool[:C.SHOT_POOL_SIZE]]
    rows = [r for r in rows if r.get("valid", True)]
    rows = [r for r in rows if r.get("ri") and r.get("answer") and r.get("question")]
    if len(rows) < C.SHOT_POOL_SIZE:
        raise SystemExit(f"[build] {source}: only {len(rows)} valid rows (<{C.SHOT_POOL_SIZE})")
    sampled = random.Random(C.SHOT_POOL_SEED).sample(rows, C.SHOT_POOL_SIZE)
    return [{"q": r["question"], "r": r["ri"], "a": r["answer"], "src_idx": r.get("idx")}
            for r in sampled]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inbox-dir", type=Path, default=INBOX_DIR,
                    help="where to write <victim>__b_<batch>.jsonl (default: inference-farm/inbox)")
    ap.add_argument("--sources", default="openthoughts,math500,gsm8k,jeebench",
                    help="comma-separated demo sources to build (default: all four)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the no-trigger Baseline-R cell")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    inbox_dir = args.inbox_dir
    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    unknown = [x for x in sources if x not in CELL]
    if unknown:
        raise SystemExit(f"unknown --sources {unknown}; choose from {sorted(CELL)}")
    inbox_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build] tokenizer {RENDER_MODEL_ID} …", flush=True)
    tok = C.make_tokenizer()
    defender_system = C.build_defender_system()

    print(f"[build] loading OT500 test rows: {C.OT_CONFIG} …", flush=True)
    test_rows = C.load_ot500_test()
    print(f"[build] OT500 test rows: {len(test_rows)}", flush=True)

    def meta_of(tr):
        return tr.get("meta") if isinstance(tr.get("meta"), dict) else None

    def make_body_fn(source):
        demos = load_shots(source)[:K]
        def body_fn(tr):
            meta = meta_of(tr)
            return render_c(demos=demos, target_q=tr["question"],
                            bench="openthoughts500", wrap=WRAP, target_meta=meta,
                            extra_hint=hint_for("openthoughts500", meta))
        return body_fn

    def extra_meta(source):
        def fn(tr):
            return {"meta_wrap": WRAP, "meta_K": K, "meta_Mc": SHOT_TEACHER,
                    "meta_demo_source": source}
        return fn

    stats = []
    for source in sources:
        out = inbox_dir / f"{VICTIM}__b_{BATCH[source]}.jsonl"
        stats.append(C.build_inbox(
            cell_id=CELL[source], batch_name=BATCH[source], test_rows=test_rows,
            tok=tok, defender_system=defender_system, body_fn=make_body_fn(source),
            out_path=out, victim_key=VICTIM, max_new_tokens=MAX_NEW_TOKENS,
            max_input_tokens=MAX_INPUT_TOKENS, extra_meta_fn=extra_meta(source),
        ))

    # No-trigger baseline (same body as config_sweep's baseline).
    def base_body(tr):
        meta = meta_of(tr)
        instr = baseline_r_instruction("openthoughts500", meta)
        return f"{instr}\n\nQuestion:\n{tr['question'].strip()}"
    if not args.no_baseline:
        stats.append(C.build_inbox(
            cell_id="NoTrigger-14b", batch_name="baseline", test_rows=test_rows,
            tok=tok, defender_system=defender_system, body_fn=base_body,
            out_path=inbox_dir / f"{VICTIM}__b_baseline.jsonl", victim_key=VICTIM,
            max_new_tokens=MAX_NEW_TOKENS, max_input_tokens=MAX_INPUT_TOKENS,
            extra_meta_fn=lambda tr: {"meta_wrap": "", "meta_K": 0},
        ))

    (DATA_DIR / "build_manifest.json").write_text(json.dumps({
        "wrap": WRAP, "K": K, "victim": VICTIM, "shot_teacher": SHOT_TEACHER,
        "ot_config": C.OT_CONFIG,
        "render_model_id": RENDER_MODEL_ID, "cells": stats,
    }, indent=2) + "\n")
    print(f"[build] DONE {len(stats)} cells", flush=True)


if __name__ == "__main__":
    main()
