"""Code-paradigm ablation — how code-like must the reveal framing be? (paper Table 7)

Fixes the default REP configuration (k=3, OpenThoughts demos, victim
Qwen3-14B, deployed defender system prompt) and varies ONLY the post-``</think>``
reveal framing of the demonstrations along a code-likeness gradient:

  ========  ================  =====================================================
  cell      wrap              reveal framing
  ========  ================  =====================================================
  V3anchor  V3 markdown_fence full code:  ```bash / $ cat reasoning_trace.txt / ```
  BareCat   bare_cat          bare command: `cat reasoning_trace.txt` (no fence, no $)
  LabeledP  labeled_prose     no code:      "Reasoning:" / "Answer:"
  ProseRev  prose_reveal      extra: prose sentence naming the file (not in paper)
  ========  ================  =====================================================

The V3 anchor is rendered by ``rep_core.variants`` and is byte-identical to the
config-sweep ``OT-V3-K3`` cell; the non-code wrappers live in
``variants_noncode.py`` next to this file. Demos, shot pool, system prompt,
output-format hint, chat template and k are held constant across the four cells.

Writes SGLang inbox JSONL files (``qwen3-14b__cp_<wrap>.jsonl``) for the
inference farm; score the harvested traces with ``public/eval/trace_metrics``.

Run::

    python build_prompts.py --help
    python build_prompts.py                       # all four cells
    python build_prompts.py --cells v3,bare_cat,labeled_prose
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
sys.path.insert(0, str(EXP_DIR))

import _common as C  # noqa: E402
from rep_core.variants import render_user_content as render_v3  # noqa: E402
from rep_core.build_helpers import hint_for, RENDER_MODEL_ID  # noqa: E402
from variants_noncode import render_user_content as render_noncode  # noqa: E402

INBOX_DIR = STEAL_METHOD / "inference-farm" / "inbox"
DATA_DIR = EXP_DIR / "data"
SHOT_POOL_PATH = C.SHOT_POOL_LOCAL   # vendored under public/data/shot_pool/

VICTIM = "qwen3-14b"
SHOT_TEACHER = "qwen3_14b"
K = 3
MAX_NEW_TOKENS = 32000
MAX_INPUT_TOKENS = 128_000

# key -> (cell_id, batch_name, kind, wrap)
CELLS = {
    "v3":            ("V3anchor-14b",     "cp_v3_anchor",     "v3",      "V3"),
    "prose_reveal":  ("ProseReveal-14b",  "cp_prose_reveal",  "noncode", "prose_reveal"),
    "labeled_prose": ("LabeledProse-14b", "cp_labeled_prose", "noncode", "labeled_prose"),
    "bare_cat":      ("BareCat-14b",      "cp_bare_cat",      "noncode", "bare_cat"),
}


def _meta_of(tr: dict):
    return tr.get("meta") if isinstance(tr.get("meta"), dict) else None


def render_body(kind: str, wrap: str, demos: list[dict], tr: dict) -> str:
    """User-message body for one OT-500 test row under one cell."""
    meta = _meta_of(tr)
    hint = hint_for("openthoughts500", meta)
    if kind == "v3":
        return render_v3(demos=demos, target_q=tr["question"], bench="openthoughts500",
                         wrap="V3", target_meta=meta, extra_hint=hint)
    return render_noncode(demos=demos, target_q=tr["question"], wrap=wrap, extra_hint=hint)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inbox-dir", type=Path, default=INBOX_DIR,
                    help="where to write <victim>__<batch>.jsonl (default: inference-farm/inbox)")
    ap.add_argument("--cells", default=",".join(CELLS),
                    help=f"comma-separated subset of {list(CELLS)} (default: all)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    keys = [k.strip() for k in args.cells.split(",") if k.strip()]
    bad = [k for k in keys if k not in CELLS]
    if bad:
        raise SystemExit(f"unknown --cells {bad}; choose from {list(CELLS)}")
    inbox_dir = args.inbox_dir
    inbox_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[build] tokenizer {RENDER_MODEL_ID} …", flush=True)
    tok = C.make_tokenizer()
    defender_system = C.build_defender_system()

    print(f"[build] loading OT500 test rows: {C.OT_CONFIG} …", flush=True)
    test_rows = C.load_ot500_test()
    print(f"[build] OT500 test rows: {len(test_rows)}", flush=True)

    demos_full = C.load_ot_shot_pool(SHOT_POOL_PATH)
    demos = demos_full[:K]
    print(f"[build] using first K={K} of {len(demos_full)} shots", flush=True)

    stats = []
    for key in keys:
        cell_id, batch, kind, wrap = CELLS[key]
        stats.append(C.build_inbox(
            cell_id=cell_id, batch_name=batch, test_rows=test_rows, tok=tok,
            defender_system=defender_system,
            body_fn=lambda tr, kind=kind, wrap=wrap: render_body(kind, wrap, demos, tr),
            out_path=inbox_dir / f"{VICTIM}__{batch}.jsonl", victim_key=VICTIM,
            max_new_tokens=MAX_NEW_TOKENS, max_input_tokens=MAX_INPUT_TOKENS,
            extra_meta_fn=lambda tr, wrap=wrap: {"meta_wrap": wrap, "meta_K": K,
                                                 "meta_Mc": SHOT_TEACHER},
        ))

    (DATA_DIR / "build_manifest.json").write_text(json.dumps({
        "victim": VICTIM, "shot_teacher": SHOT_TEACHER, "K": K,
        "ot_config": C.OT_CONFIG,
        "render_model_id": RENDER_MODEL_ID, "cells": stats,
    }, indent=2) + "\n")
    print(f"[build] DONE {len(stats)} cells, total_rows={sum(s['n_rows'] for s in stats)}",
          flush=True)


if __name__ == "__main__":
    main()
