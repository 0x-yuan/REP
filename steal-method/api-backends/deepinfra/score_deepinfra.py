"""Score DeepInfra outputs with the paper's ROUGE logic and (optionally) pair
them against an our-side (SGLang) per-row table on the identical idx set.

Per cell:
  - DeepInfra fidelity = mean ROUGE-L(r2, ri) over rows
  - our-side fidelity  = `rir2_rouge_l` from a per-row CSV (rescore_ours.py)
  - reports structural rate, truncation rate, completion length, paired delta.

Usage:
  python score_deepinfra.py --outputs outputs/deepinfra_14b.jsonl --ref ref/ri_qwen3_14b.json \\
      --label "Qwen3-14B V3-K3" --save scored/di_14b_short.csv [--ours scored/our_14b.csv]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fast_score import score_fidelity

def load_outputs(path):
    out = {}
    for line in Path(path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("text") is not None and r.get("error") is None:
            out[r["idx"]] = r
    return out


def score_outputs(outputs, ref):
    rows = []
    t0 = time.time()
    for i, (idx, o) in enumerate(outputs.items()):
        g = ref.get(idx)
        if g is None:
            continue
        s = score_fidelity(o["text"], g["ri"])
        s["idx"] = idx
        s["finish_reason"] = o.get("finish_reason")
        s["truncated"] = 1.0 if o.get("finish_reason") == "length" else 0.0
        s["completion_tokens"] = o.get("completion_tokens")
        rows.append(s)
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(outputs)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--save", required=True, help="per-row CSV of DeepInfra scores")
    ap.add_argument("--ours", default=None, help="our-side per-row CSV (rescore_ours.py)")
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    outs = load_outputs(args.outputs)
    print(f"[{args.label}] DeepInfra outputs loaded: {len(outs)}", flush=True)
    di = score_outputs(outs, ref)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    di.to_csv(args.save, index=False)

    print(f"\n================ {args.label} ================", flush=True)
    print(f"-- DeepInfra (all {len(di)} scored) --")
    print(f"  fidelity ROUGE-L(r2,ri) : {di.rir2_rouge_l.mean():.4f}")
    print(f"  r1r2_rouge_l            : {di.r1r2_rouge_l.mean():.4f}")
    print(f"  structural_success      : {di.structural_success.mean():.4f}")
    print(f"  truncated (finish=len)  : {di.truncated.mean():.4f}")
    print(f"  mean completion tokens  : {di.completion_tokens.dropna().mean():.0f}")

    if not args.ours:
        return
    ours = pd.read_csv(args.ours)
    common = set(di.idx) & set(ours.idx)
    di_c = di[di.idx.isin(common)].set_index("idx")
    ours_c = ours[ours.idx.isin(common)].set_index("idx")
    print(f"\n-- paired on {len(common)} common idx --")
    print(f"  OUR  fidelity ROUGE-L(r2,ri): {ours_c.rir2_rouge_l.mean():.4f}")
    print(f"  DI   fidelity ROUGE-L(r2,ri): {di_c.rir2_rouge_l.mean():.4f}")
    print(f"  delta (DI - OUR)            : {di_c.rir2_rouge_l.mean() - ours_c.rir2_rouge_l.mean():+.4f}")
    print(f"  OUR  structural            : {ours_c.structural_success.mean():.4f}")
    print(f"  DI   structural            : {di_c.structural_success.mean():.4f}")


if __name__ == "__main__":
    main()
