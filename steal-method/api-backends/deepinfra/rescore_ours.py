"""Re-score our-side (SGLang farm) generations with fast_score so the two
sides use one identical ROUGE implementation.

Input is either
  * a per-row jsonl / parquet with `idx` and `full_response` (raw generation), or
  * one with `idx`, `r1`, `r2` (reconstructed as <think>r1</think>\\nr2; --recon),
e.g. a per-experiment result table (table1.parquet: filter wrapper/k;
table4.parquet: filter victim_model).

Output: per-row CSV with idx, rir2_rouge_l, rir1_rouge_l, r1r2_rouge_l,
structural_success, answer_match (consumed by score_deepinfra.py --ours and
master_table.py).

Usage:
  python rescore_ours.py --input table1.parquet --ref ref/ri_qwen3_14b.json \\
      --filter wrapper=markdown_fence --filter k=3 --out scored/our_14b.csv
  python rescore_ours.py --input table4.parquet --ref ref/ri_qwen3_32b.json \\
      --filter victim_model=qwen3-32b --recon --out scored/our_32b.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fast_score import score_fidelity


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def apply_filters(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    for f in filters:
        col, val = f.split("=", 1)
        series = df[col]
        df = df[series.astype(str) == val]
    return df


def rescore(df: pd.DataFrame, ref: dict, label: str, recon: bool) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        idx = r["idx"]
        g = ref.get(idx)
        if g is None:
            continue
        text = (f"<think>{r.get('r1') or ''}</think>\n{r.get('r2') or ''}") if recon else (r.get("full_response") or "")
        s = score_fidelity(text, g["ri"])
        s["idx"] = idx
        s["answer_match"] = float(r.get("answer_match") or 0.0)
        rows.append(s)
        if (i + 1) % 50 == 0:
            print(f"  {label} {i + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
    o = pd.DataFrame(rows)
    print(f"=== {label} (n={len(o)}) ===")
    for k in ["rir2_rouge_l", "rir1_rouge_l", "r1r2_rouge_l", "structural_success", "answer_match"]:
        print(f"  {k:22s}: {o[k].mean():.4f}")
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="parquet / csv / jsonl per-row table")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", action="append", default=[], help="col=value (repeatable)")
    ap.add_argument("--recon", action="store_true", help="rebuild text from r1/r2 columns instead of full_response")
    ap.add_argument("--label", default="OURS")
    a = ap.parse_args()
    df = apply_filters(load_table(Path(a.input)), a.filter)
    ref = json.load(open(a.ref))
    out = rescore(df, ref, a.label, a.recon)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
