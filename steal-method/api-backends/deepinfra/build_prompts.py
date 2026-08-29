"""Render V3-K3 REP attack prompts for a Qwen3 victim hosted on DeepInfra.

Byte-faithful to the paper's V3 (`markdown_fence`) render with the deployed
defender system prompt, passed through the Qwen3 chat template (enable_thinking)
so the request can be sent RAW to `/v1/openai/completions` -- identical to
what the self-hosted SGLang farm receives.

DeepInfra rule 1 -- 40960 context, no YaRN. DeepInfra serves Qwen3-14B/32B at
max_total_tokens = 40960 (our farm runs YaRN-131K). Per-row
``max_tokens = ctx - prompt_tokens - MARGIN`` (the runner auto-shrinks on 400).

DeepInfra rule 2 -- the 0.9x short demo draw. The paper's seed-7 K=3 draw is
~42K tokens on 14B and cannot be submitted at all. Default here is the short
draw: shot-pool rows idx 15, 21, 33 (total think 15967 chars vs 17755 for the
paper draw, ~0.9x) -> prompt ~32.5K, ~8.5K output budget. Pass --paper-draw to
render the original draw (only fits if the victim has a larger context).

Output rows: {"idx", "prompt", "ptok", "max_tokens"}.

Usage:
  python build_prompts.py --out prompts/qwen3-14b.jsonl
  python build_prompts.py --out prompts/qwen3-32b.jsonl --ctx 40960 --demo-idx 15,21,33
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_STEAL_METHOD = Path(__file__).resolve().parents[2]
if str(_STEAL_METHOD) not in sys.path:
    sys.path.insert(0, str(_STEAL_METHOD))
from rep_core import build_defender_system, render_user_content  # noqa: E402
from rep_core.build_helpers import RENDER_MODEL_ID  # noqa: E402
from experiments._common import load_jsonl_gz, load_ot500_test, render_chat  # noqa: E402

DATA_ROOT = _STEAL_METHOD.parent / "data"
SHOT_POOL = DATA_ROOT / "shot_pool" / "qwen3_14b.jsonl.gz"
EXTRA_HINT = r"Output format: write the final answer wrapped in \boxed{}."
SHORT_DRAW = (15, 21, 33)
MARGIN = 256
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def pick_demos(pool: list[dict], demo_idx: tuple[int, ...] | None, k: int) -> list[dict]:
    if demo_idx is None:
        picked = random.Random(7).sample(pool, 50)[:k]        # the paper draw
    else:
        picked = [pool[i] for i in demo_idx]                   # the short draw
    return [{"q": d["question"], "r": _THINK_TAG.sub("", d["think"]).strip(), "a": d["answer"]}
            for d in picked]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=40960, help="victim context (DeepInfra Qwen3: 40960)")
    ap.add_argument("--demo-idx", default=",".join(map(str, SHORT_DRAW)), help="shot-pool row indices")
    ap.add_argument("--paper-draw", action="store_true", help="use random.Random(7).sample(pool,50)[:k] instead")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cap-out", type=int, default=24000, help="upper cap on per-row max_tokens")
    ap.add_argument("--shot-pool", default=str(SHOT_POOL))
    ap.add_argument("--render-model", default=RENDER_MODEL_ID, help="tokenizer used for the chat template")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pool = load_jsonl_gz(Path(args.shot_pool))
    demo_idx = None if args.paper_draw else tuple(int(x) for x in args.demo_idx.split(","))
    demos = pick_demos(pool, demo_idx, args.k)
    print(f"demos: {'paper draw' if demo_idx is None else demo_idx}  total think chars={sum(len(d['r']) for d in demos)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.render_model)
    system = build_defender_system()
    test = load_ot500_test()
    if args.limit:
        test = test[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = []
    with out.open("w") as f:
        for row in test:
            body = render_user_content(demos, row["question"], "openthoughts", "V3", extra_hint=EXTRA_HINT)
            prompt = render_chat(tok, system, body, enable_thinking=True)
            ptok = len(tok(prompt).input_ids)
            budget = args.ctx - ptok - MARGIN
            f.write(json.dumps({"idx": row["idx"], "prompt": prompt, "ptok": ptok,
                                "max_tokens": max(256, min(args.cap_out, budget))}) + "\n")
            lens.append(ptok)
    lens.sort()
    n = len(lens)
    print(f"wrote {n} rows -> {out}")
    print(f"  prompt tokens: min={lens[0]} p50={lens[n // 2]} max={lens[-1]}")
    over = sum(1 for L in lens if L + MARGIN + 256 > args.ctx)
    print(f"  rows with <256 output budget (ctx={args.ctx}): {over}")


if __name__ == "__main__":
    main()
