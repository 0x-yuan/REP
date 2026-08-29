"""Write the reference file {idx: {ri, gold, answer, question}} used by the
scorers. ``ri`` = the victim's benign hidden trace (recorded in the
OpenThoughts-500 test set; one config per victim: ri_qwen3_14b / ri_qwen3_32b).

Both configs are vendored under public/data/openthoughts_test_500/.
Alternatively build the reference from a baseline (no-attack) run file with
--baseline: ri = content of the first <think>...</think> block of each output.

Usage:
  python make_ref.py --config ri_qwen3_14b --out ref/ri_qwen3_14b.json
  python make_ref.py --baseline outputs/baseline_32b.jsonl --out ref/ri_qwen3_32b.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_STEAL_METHOD = Path(__file__).resolve().parents[2]
if str(_STEAL_METHOD) not in sys.path:
    sys.path.insert(0, str(_STEAL_METHOD))
from experiments._common import load_ot500_test  # noqa: E402

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_ri(text: str) -> str:
    m = _THINK_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").replace("<think>", "").replace("</think>", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="ri_qwen3_14b", help="test-set config (victim whose ri to use)")
    ap.add_argument("--baseline", default=None, help="baseline run jsonl {idx,text}; overrides the recorded ri")
    args = ap.parse_args()

    rows = load_ot500_test(args.config)
    ref = {r["idx"]: {"ri": r.get("ri") or "", "gold": r.get("gold"),
                      "answer": r.get("answer"), "question": r["question"]} for r in rows}
    if args.baseline:
        seen = set()
        for line in Path(args.baseline).open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or r.get("text") is None or r["idx"] not in ref:
                continue
            ref[r["idx"]]["ri"] = extract_ri(r["text"])
            seen.add(r["idx"])
        ref = {k: v for k, v in ref.items() if k in seen}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ref, out.open("w"), ensure_ascii=False)
    lens = sorted(len(v["ri"]) for v in ref.values())
    print(f"wrote {len(ref)} ri rows -> {out}  (empty ri: {sum(1 for L in lens if L == 0)})")
    if lens:
        print(f"  ri chars: min={lens[0]} p50={lens[len(lens) // 2]} max={lens[-1]}")


if __name__ == "__main__":
    main()
