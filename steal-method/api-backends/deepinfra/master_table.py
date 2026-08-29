"""Aggregate our-side (SGLang) vs DeepInfra fidelity into one comparison
table: overall, on DeepInfra non-truncated rows, on common idx, and on the
both-structural common subset (the cleanest paired number). Optionally a
bootstrap 95% CI on the mean paired delta (DI - OUR) over that subset.

Fidelity = ROUGE-L(r2, ri) = source_reasoning_rouge_l_r2.

Usage:
  python master_table.py --cell "Qwen3-14B V3-K3 (0.9x short demo)" \\
      --di scored/di_14b_short.csv --ours scored/our_14b.csv --ci
"""
from __future__ import annotations

import argparse
import random

import pandas as pd


def paired_ci(delta: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(delta)
    means = []
    for _ in range(n_boot):
        s = [delta[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def block(label: str, di_csv: str, ours_csv: str, ci: bool) -> None:
    df = pd.read_csv(di_csv)
    ours = pd.read_csv(ours_csv)
    nt = df[df.truncated == 0]
    common = set(df.idx) & set(ours.idx)
    di_c = df[df.idx.isin(common)]
    ours_c = ours[ours.idx.isin(common)]
    cs = common & set(df[df.structural_success == 1].idx) & set(ours[ours.structural_success == 1].idx)
    print(f"\n================ {label} ================")
    print(f"DeepInfra rows scored: {len(df)}   our-side rows: {len(ours)}   common idx: {len(common)}")
    print(f"{'metric':<34}{'OUR (SGLang)':>16}{'DeepInfra':>16}")
    print(f"{'fidelity ROUGE-L(r2,ri) [all]':<34}{ours.rir2_rouge_l.mean():>16.4f}{df.rir2_rouge_l.mean():>16.4f}")
    print(f"{'fidelity [DI non-truncated]':<34}{'':>16}{nt.rir2_rouge_l.mean():>16.4f}  (n={len(nt)})")
    print(f"{'fidelity [paired common idx]':<34}{ours_c.rir2_rouge_l.mean():>16.4f}{di_c.rir2_rouge_l.mean():>16.4f}  (n={len(common)})")
    if cs:
        o = ours[ours.idx.isin(cs)].set_index("idx").rir2_rouge_l
        d = df[df.idx.isin(cs)].set_index("idx").rir2_rouge_l
        print(f"{'fidelity [both-structural common]':<34}{o.mean():>16.4f}{d.mean():>16.4f}  (n={len(cs)})")
        if ci:
            delta = [float(d[i] - o[i]) for i in cs]
            lo, hi = paired_ci(delta)
            print(f"{'  paired delta DI-OUR, 95% CI':<34}{sum(delta) / len(delta):>+16.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"{'structural_success':<34}{ours.structural_success.mean():>16.4f}{df.structural_success.mean():>16.4f}")
    print(f"{'r1r2_rouge_l (self-echo)':<34}{ours.r1r2_rouge_l.mean():>16.4f}{df.r1r2_rouge_l.mean():>16.4f}")
    print(f"{'DI truncation rate':<34}{'':>16}{df.truncated.mean():>16.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, help="label (repeatable, paired with --di/--ours order)")
    ap.add_argument("--di", action="append", required=True, help="DeepInfra per-row CSV (score_deepinfra.py --save)")
    ap.add_argument("--ours", action="append", required=True, help="our-side per-row CSV (rescore_ours.py)")
    ap.add_argument("--ci", action="store_true", help="bootstrap 95%% CI on the both-structural paired delta")
    a = ap.parse_args()
    if not (len(a.cell) == len(a.di) == len(a.ours)):
        raise SystemExit("--cell/--di/--ours must be given the same number of times")
    for label, di, ours in zip(a.cell, a.di, a.ours):
        block(label, di, ours, a.ci)


if __name__ == "__main__":
    main()
