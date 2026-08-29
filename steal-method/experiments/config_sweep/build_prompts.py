"""REP configuration sweep — wrapper x shot-count ablation.

Reproduces the wrapper/shot selection experiment (paper Table "Wrapper
comparison at fixed k=3" and the full Appendix sweep): all six wrappers V0..V5
crossed with K=1..4 (24 attack cells) plus one no-trigger baseline, on the
500-example OpenThoughts-114k subset with victim Qwen3-14B. The default
configuration selected from this sweep is V3 (markdown_fence) with K=3.

Writes SGLang inbox JSONL files into ``../../inference-farm/inbox/`` for the
farm to harvest; score the harvested traces with ``public/eval/trace_metrics``.

Prerequisite: the vendored shadow demo pool ``data/shot_pool/qwen3_14b.jsonl.gz``
(ships with the release; ``rep_core/prepare_shot_pool.py`` rebuilds it from a
local pool JSONL).

Baseline rows (paper Table 1 rows 1-2, Appendix "Baseline Trigger Prompts"):

* **Baseline R** (no-trigger, always built): after ``</think>`` repeat the
  reasoning as plain text (``rep_core.baseline.baseline_r_instruction``).
* **Baseline C** (simple CoT; ``--include-baseline-c``): after ``</think>``
  "let's think step by step" (``rep_core.baseline.baseline_c_instruction``).
  Byte-identical to the original CoT-baseline experiment's body.

Run::

    python build_prompts.py --help
    python build_prompts.py                         # 24 REP cells + Baseline R
    python build_prompts.py --include-baseline-c    # + Baseline C cell
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXP_DIR = _HERE.parent
STEAL_METHOD = _HERE.parents[2]
sys.path.insert(0, str(STEAL_METHOD / "experiments"))

import _common as C  # noqa: E402
from rep_core.variants import (  # noqa: E402
    VARIANT_IDS, VARIANT_NAMES, render_user_content as render_c,
)
from rep_core.baseline import baseline_r_instruction, baseline_c_instruction  # noqa: E402
from rep_core.build_helpers import hint_for, RENDER_MODEL_ID  # noqa: E402

# --- Pinned configuration --------------------------------------------------
INBOX_DIR = STEAL_METHOD / "inference-farm" / "inbox"
DATA_DIR = EXP_DIR / "data"
SHOT_POOL_PATH = C.SHOT_POOL_LOCAL   # vendored under public/data/shot_pool/

VICTIM = "qwen3-14b"          # victim model key (inference-farm/slave/registry.py)
SHOT_TEACHER = "qwen3_14b"    # shadow model that generated the demos (in-domain)
K_SWEEP = (1, 2, 3, 4)
MAX_NEW_TOKENS = 32000
MAX_INPUT_TOKENS = 128_000    # 131072 ctx minus headroom


BASELINE_C = "CoT"            # sentinel `wrap` value for the Baseline C cell


def cell_id(wrap, K):
    if wrap is None:
        return "NoTrigger-14b"
    if wrap == BASELINE_C:
        return "CoT-14b"
    return f"OT-{wrap}-K{K}"


def batch_name(wrap, K):
    if wrap is None:
        return "baseline"
    if wrap == BASELINE_C:
        return "cot_baseline"
    return f"ot_{wrap}_K{K}"


def _meta_of(tr):
    return tr.get("meta") if isinstance(tr.get("meta"), dict) else None


def baseline_r_body(tr: dict) -> str:
    """Baseline R user-message body for one OT-500 test row."""
    instr = baseline_r_instruction("openthoughts500", _meta_of(tr))
    return f"{instr}\n\nQuestion:\n{tr['question'].strip()}"


def baseline_c_body(tr: dict) -> str:
    """Baseline C (simple CoT) user-message body for one OT-500 test row."""
    instr = baseline_c_instruction("openthoughts500", _meta_of(tr))
    return f"{instr}\n\nQuestion:\n{tr['question'].strip()}"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inbox-dir", type=Path, default=INBOX_DIR,
                    help="where to write <victim>__<batch>.jsonl (default: inference-farm/inbox)")
    ap.add_argument("--wraps", default=",".join(VARIANT_IDS),
                    help="comma-separated wrapper ids to build (default: V0..V5)")
    ap.add_argument("--k", default=",".join(str(k) for k in K_SWEEP),
                    help="comma-separated shot counts (default: 1,2,3,4)")
    ap.add_argument("--include-baseline-c", action="store_true",
                    help="also emit the Baseline C (simple CoT) cell, batch `cot_baseline`")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the Baseline R (no-trigger) cell")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    inbox_dir = args.inbox_dir
    wraps = [w.strip() for w in args.wraps.split(",") if w.strip()]
    bad = [w for w in wraps if w not in VARIANT_IDS]
    if bad:
        raise SystemExit(f"unknown --wraps {bad}; choose from {VARIANT_IDS}")
    ks = [int(k) for k in args.k.split(",") if k.strip()]
    inbox_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build] tokenizer {RENDER_MODEL_ID} …", flush=True)
    tok = C.make_tokenizer()
    defender_system = C.build_defender_system()

    print(f"[build] loading OT500 test rows: {C.OT_CONFIG} …", flush=True)
    test_rows = C.load_ot500_test()
    print(f"[build] OT500 test rows: {len(test_rows)}", flush=True)

    demos_full = C.load_ot_shot_pool(SHOT_POOL_PATH)
    print(f"[build] shot pool: {len(demos_full)} demos", flush=True)

    def make_body_fn(wrap, K):
        if wrap is None:
            return baseline_r_body
        if wrap == BASELINE_C:
            return baseline_c_body
        demos = demos_full[:K]
        def body_fn(tr):
            meta = _meta_of(tr)
            return render_c(
                demos=demos, target_q=tr["question"], bench="openthoughts500",
                wrap=wrap, target_meta=meta,
                extra_hint=hint_for("openthoughts500", meta),
            )
        return body_fn

    def extra_meta(wrap, K):
        is_rep = wrap is not None and wrap != BASELINE_C
        def fn(tr):
            return {
                "meta_wrap": wrap if is_rep else "",
                "meta_wrap_name": VARIANT_NAMES[wrap] if is_rep else "",
                "meta_K": K if is_rep else 0,
                "meta_Mc": SHOT_TEACHER if is_rep else "",
                "meta_mode": "rep" if is_rep else ("cot" if wrap == BASELINE_C else "notrigger"),
            }
        return fn

    cells = [(w, k) for w in wraps for k in ks]
    if not args.no_baseline:
        cells.append((None, None))
    if args.include_baseline_c:
        cells.append((BASELINE_C, None))
    stats = []
    for wrap, K in cells:
        out = inbox_dir / f"{VICTIM}__{batch_name(wrap, K)}.jsonl"
        stats.append(C.build_inbox(
            cell_id=cell_id(wrap, K), batch_name=batch_name(wrap, K),
            test_rows=test_rows, tok=tok, defender_system=defender_system,
            body_fn=make_body_fn(wrap, K), out_path=out, victim_key=VICTIM,
            max_new_tokens=MAX_NEW_TOKENS, max_input_tokens=MAX_INPUT_TOKENS,
            extra_meta_fn=extra_meta(wrap, K),
        ))

    (DATA_DIR / "build_manifest.json").write_text(json.dumps({
        "victim": VICTIM, "shot_teacher": SHOT_TEACHER, "K_sweep": ks,
        "variant_ids": wraps, "variant_names": VARIANT_NAMES,
        "include_baseline_c": bool(args.include_baseline_c),
        "baseline_r_instruction": baseline_r_instruction("openthoughts500", None),
        "baseline_c_instruction": baseline_c_instruction("openthoughts500", None),
        "ot_config": C.OT_CONFIG,
        "render_model_id": RENDER_MODEL_ID, "cells": stats,
    }, indent=2) + "\n")
    total = sum(s["n_rows"] for s in stats)
    print(f"[build] DONE {len(stats)} cells: total_rows={total}", flush=True)


if __name__ == "__main__":
    main()
