"""REP distillation corpus — step 3: harvested outputs -> corpus rows -> the
Table-5 *orig* / *clean* splits. The ``Chia-Mu-Lab/REP-datasets`` configs
``distill_q3_{14b,32b}_{original,clean}`` are these rows projected to
``question, r1, r2, answer, completion``.

Pipeline per output row (``corpus_lib``):

1. map ``id = "b2::<prompt_id>"`` back to the query (question, source_index)
2. ``parse_think`` -> r1 (inside ``<think>``), r2 (after ``</think>``),
   ``structural`` (well-ordered pair present)
3. ``answer`` = last brace-matched ``\\boxed{}`` of r2 (or of the whole output
   when non-structural); the ``\\boxed{}`` is stripped off r2
4. dedupe by prompt_id (last write wins — result files are read in sorted
   name order, so name re-runs so they sort later), sort by source_index

Filters (``--filter``):

* ``corpus``     — every harvested row (10 000 rows incl. ``structural=False``)
* ``original`` — paper *orig* split: ``structural == True`` (the Hub
                 ``*_original`` configs: 8 046 rows for 14B, 6 291 for 32B)
* ``clean``    — paper *clean* split: structural AND ``answer`` math-verify-
                 equivalent to the OpenThoughts gold; rows are promoted to the
                 15-column clean schema (adds gold_boxed / answer_match /
                 source). Gold comes from ``--gold hub`` (OpenThoughts-114k
                 metadata: union of the ``ground_truth_solution`` and
                 ``deepseek_solution`` boxed values) or a local JSONL with
                 ``{source_index, gold_boxed}`` (single gold, e.g. the file
                 written by ``sample_questions.py``).

Clean top-up: ``--topup-results`` + ``--topup-queries`` score a second
harvest the same way and union it (primary wins), capped at ``--target``
rows — the selection rule behind the published 10k clean corpus.

Run::

    python assemble_corpus.py --results ../../inference-farm/result --filter corpus \\
        --out results/corpus.jsonl
    python assemble_corpus.py --results ../../inference-farm/result --filter clean \\
        --gold hub --topup-results result_topup/ \\
        --topup-queries prompts/queries/topup_25k.jsonl --out results/clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXP_DIR = _HERE.parent
sys.path.insert(0, str(EXP_DIR))

import corpus_lib as L  # noqa: E402

VICTIM = "qwen3_14b"
PRIMARY_RANGE = range(20_000, 30_000)     # OpenThoughts-114k rows of the 10k set


def iter_result_rows(paths: list[Path]) -> list[dict]:
    """All JSONL rows under the given files/dirs (skips ``*.meta.jsonl`` /
    ``*.preshard.jsonl``), files in sorted name order."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(fp for fp in p.rglob("*.jsonl")
                            if not fp.name.endswith((".meta.jsonl", ".preshard.jsonl")))
        elif p.exists():
            files.append(p)
    rows: list[dict] = []
    for fp in files:
        with fp.open() as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    print(f"[collect] {len(rows)} rows from {len(files)} file(s)", flush=True)
    return rows


def load_queries(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["prompt_id"]] = r
    return out


def corpus_rows(result_paths: list[Path], q_lookup: dict[str, dict], *,
              victim: str, cell_id: str) -> list[dict]:
    by_pid: dict[str, dict] = {}
    n_skip = 0
    for raw in iter_result_rows(result_paths):
        row = L.build_corpus_row(raw, q_lookup, victim=victim, cell_id=cell_id)
        if row is None:
            n_skip += 1
            continue
        by_pid[row["prompt_id"]] = row
    rows = sorted(by_pid.values(), key=lambda r: r["source_index"])
    n_struct = sum(r["structural"] for r in rows)
    print(f"[corpus] {len(rows)} unique rows (skip={n_skip})  structural={n_struct} "
          f"({100 * n_struct / max(1, len(rows)):.1f}%)", flush=True)
    return rows


def gold_lookup_hub(indices: range | list[int]) -> dict[int, tuple[str, str]]:
    """source_index -> (gold_from_ground_truth, gold_from_deepseek)."""
    from datasets import load_dataset  # lazy
    print("[gold] loading OpenThoughts-114k metadata …", flush=True)
    ot = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
    out: dict[int, tuple[str, str]] = {}
    for si in indices:
        r = ot[int(si)]
        out[int(si)] = (L.extract_gold_boxed(r.get("ground_truth_solution") or ""),
                        L.extract_gold_boxed(r.get("deepseek_solution") or ""))
    return out


def gold_lookup_jsonl(path: Path) -> dict[int, tuple[str]]:
    out: dict[int, tuple[str]] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[int(r["source_index"])] = (r.get("gold_boxed") or "",)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="+", required=True,
                    help="farm result files/dirs of the primary harvest")
    ap.add_argument("--queries", type=Path, default=EXP_DIR / "prompts" / "queries" / "qwen3_14b.jsonl")
    ap.add_argument("--filter", choices=["corpus", "original", "clean"], default="corpus")
    ap.add_argument("--gold", default="hub",
                    help="clean only: 'hub' (OpenThoughts-114k metadata, union gold) "
                         "or a JSONL with {source_index, gold_boxed}")
    ap.add_argument("--topup-results", type=Path, nargs="*", default=[],
                    help="clean only: result files/dirs of a top-up harvest")
    ap.add_argument("--topup-queries", type=Path, default=None,
                    help="clean only: queries JSONL of the top-up harvest (carries gold_boxed)")
    ap.add_argument("--target", type=int, default=10_000, help="clean cap (default: %(default)s)")
    ap.add_argument("--source-tag", default="existing_10k")
    ap.add_argument("--topup-source-tag", default="new_25k")
    ap.add_argument("--victim", default=VICTIM)
    ap.add_argument("--cell-id", default=None, help="default: from the cell file in prompts/cells/")
    ap.add_argument("--expect", type=int, default=10_000,
                    help="abort if fewer unique harvested rows (0 disables)")
    ap.add_argument("--out", type=Path, required=True, help="output JSONL")
    ap.add_argument("--push-to", default=None, help="optional HF dataset repo id")
    args = ap.parse_args()

    cell_id = args.cell_id
    if cell_id is None:
        cands = sorted((EXP_DIR / "prompts" / "cells").glob("*.json"))
        cell_id = json.loads(cands[0].read_text())["cell_id"] if len(cands) == 1 else \
            f"distill_C_{args.victim}_qwen3_14b_K3_V3"

    q_lookup = load_queries(args.queries)
    rows = corpus_rows(args.results, q_lookup, victim=args.victim, cell_id=cell_id)
    if args.expect and len(rows) < args.expect:
        raise SystemExit(f"[assemble] have {len(rows)} rows < expected {args.expect}; "
                         f"pass --expect 0 to proceed")

    if args.filter == "corpus":
        final = rows
    elif args.filter == "original":
        final = L.filter_original(rows)
    else:
        idx = [r["source_index"] for r in rows]
        gold = gold_lookup_hub(idx) if args.gold == "hub" else gold_lookup_jsonl(Path(args.gold))
        final = L.filter_clean(rows, gold, source=args.source_tag,
                               victim=args.victim, cell_id=cell_id)
        print(f"[clean] primary clean rows: {len(final)} / {len(rows)}", flush=True)
        if args.topup_results:
            if args.topup_queries is None:
                raise SystemExit("--topup-results needs --topup-queries")
            tq = load_queries(args.topup_queries)
            t_rows = corpus_rows(args.topup_results, tq, victim=args.victim, cell_id=cell_id)
            t_gold = gold_lookup_jsonl(args.topup_queries)
            topup = L.filter_clean(t_rows, t_gold, source=args.topup_source_tag,
                                   victim=args.victim, cell_id=cell_id)
            print(f"[clean] top-up clean rows: {len(topup)} / {len(t_rows)}", flush=True)
            final = L.merge_clean(final, topup, args.target)
        else:
            final = final[: args.target]
        if len(final) < args.target:
            print(f"[clean] WARN: {len(final)} clean rows < target {args.target}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[assemble] filter={args.filter} rows={len(final)} -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.push_to:
        from datasets import Dataset  # lazy
        Dataset.from_list(final).push_to_hub(args.push_to, private=False)
        print(f"[push] https://huggingface.co/datasets/{args.push_to}", flush=True)


if __name__ == "__main__":
    main()
