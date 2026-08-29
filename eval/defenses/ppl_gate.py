"""Defense C-2 = perplexity filter (Alon & Kamfonas 2023 / Jain et al. 2023;
cataloged in Liu et al., USENIX Security 2024).

Adversarial-suffix attacks (GCG) inject high-perplexity gibberish, so a small
LM's perplexity flags them. REP injects fluent natural language + light
markdown, so its perplexity sits in the benign range and the filter cannot
separate it. We score windowed perplexity with GPT-2 (the canonical filter LM)
and report mean and max-window PPL for REP vs the long benign control. The
FPR-0 threshold is the max benign ``max_ppl``; REP rows above it are flagged.

Needs torch + transformers (CPU is fine, ~1s/row).

    uv run --with torch --with transformers python ppl_gate.py --prompts <rep rows> \\
        --out gate_out/ppl_rep_n100.jsonl --limit 100
    uv run ... python ppl_gate.py ... --mode benign_long --out gate_out/ppl_benign_long_n100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from defenses.prompt_lib import benign_text, parse, row_idx  # noqa: E402

WIN = 1024
MAXW = 12  # sampled windows per input (evenly spaced, always incl. first + last)
FILTER_LM = "gpt2"


def window_starts(n_tokens: int, win: int = WIN, maxw: int = MAXW) -> list[int]:
    """Start offsets of the scored windows: every ``win`` tokens, subsampled to
    at most ``maxw`` evenly spaced starts that always keep the first and last."""
    starts = list(range(0, n_tokens, win))
    if len(starts) > maxw:
        idxs = sorted(set([0, len(starts) - 1] +
                          [round(i * (len(starts) - 1) / (maxw - 1)) for i in range(maxw)]))
        starts = [starts[i] for i in idxs]
    return starts


def flag_threshold(benign_max_ppl: list[float]) -> float:
    """FPR-0 threshold: the largest benign max-window perplexity."""
    return max(benign_max_ppl)


class PPLScorer:
    def __init__(self, model_id: str = FILTER_LM):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).eval()

    def _window_ppl(self, ids) -> float:
        with self.torch.no_grad():
            out = self.model(ids.unsqueeze(0), labels=ids.unsqueeze(0))
        return float(self.torch.exp(out.loss))

    def ppl_of(self, text: str) -> dict:
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        n = int(ids.shape[0])
        ppls = [self._window_ppl(ids[s:s + WIN]) for s in window_starts(n) if ids[s:s + WIN].shape[0] >= 2]
        return {"n_tokens": n, "n_windows": len(ppls),
                "mean_ppl": sum(ppls) / len(ppls), "max_ppl": max(ppls)}


def summarize(rows: list[dict], mode: str) -> str:
    mp = sorted(r["mean_ppl"] for r in rows)
    xp = sorted(r["max_ppl"] for r in rows)
    return (f"[{mode}] N={len(rows)} mean_ppl median={mp[len(mp)//2]:.1f} "
            f"max_ppl median={xp[len(xp)//2]:.1f} max_ppl p95={xp[max(0, int(0.95*len(xp))-1)]:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rep", "benign_long"], default="rep")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--lm", default=FILTER_LM)
    ap.add_argument("--benign", default=None,
                    help="benign_long decisions jsonl; if given, report how many REP rows "
                         "exceed the FPR-0 threshold")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.prompts).read_text().splitlines() if ln.strip()][: args.limit]
    scorer = PPLScorer(args.lm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    with open(args.out, "w") as fh:
        for i, r in enumerate(rows, 1):
            _s, u = parse(r["prompt"])
            res = scorer.ppl_of(benign_text(u, args.mode))
            res["idx"] = row_idx(r)
            out_rows.append(res)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(rows)} mean_ppl={res['mean_ppl']:.1f} max_ppl={res['max_ppl']:.1f}", flush=True)
    print(summarize(out_rows, args.mode))
    if args.benign:
        ben = [json.loads(ln) for ln in Path(args.benign).read_text().splitlines() if ln.strip()]
        thr = flag_threshold([b["max_ppl"] for b in ben])
        flagged = sum(r["max_ppl"] > thr for r in out_rows)
        print(f"[{args.mode}] FPR-0 threshold={thr:.1f}: flagged {flagged}/{len(out_rows)}")


if __name__ == "__main__":
    main()
