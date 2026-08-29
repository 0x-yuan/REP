"""Build the three held-out QA eval sets, disjoint from the 6k training seed.

The distillation seed (vendored ``data/qa_seed/reasoning_seed_6k.jsonl.gz``,
6000 rows, 2000 per domain) was drawn index-wise with
``random.Random(7).sample(range(n), 2000)`` from each public source. The eval
sets are the *complement*:

  strategyqa : the ChilleD/StrategyQA ``test`` split (687 rows). The seed is
               built from ``train`` + ``test``; any seed row that overlaps the
               test split was dropped from the distillation data instead
               (contamination fix), so the full test split is a clean eval set.
  prontoqa   : longface/prontoqa-train complement of the 2000 seed, sample 300
               (``random.Random(123)``).
  hotpotqa   : hotpotqa/hotpot_qa ``distractor`` validation complement of the
               2000 seed, sample 300 (all ``level == hard``), open-book: the
               10 distractor paragraphs are prepended to the question.

Each output row: ``{"id", "question", "answer", "source", "answer_type"}``.

``--check-disjoint`` additionally reads the vendored seed and asserts, by
normalised question text, that no eval row is in the seed.

Run (downloads the three public source datasets on first use):

    uv run --with datasets python build_eval_sets.py --out data/ --check-disjoint
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from pathlib import Path

SEED, N_SEED, EVAL_N = 7, 2000, 300
SEED_LOCAL = Path(__file__).resolve().parents[2] / "data" / "qa_seed" / "reasoning_seed_6k.jsonl.gz"


def load_seed(path: Path = SEED_LOCAL) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s: str) -> str:
    return " ".join((s or "").split())


def held_out(n: int) -> list[int]:
    """Indices NOT selected by the seed draw (index-based, matches the seed builder)."""
    sel = set(random.Random(SEED).sample(range(n), N_SEED))
    return [i for i in range(n) if i not in sel]


def build_strategyqa() -> list[dict]:
    from datasets import load_dataset
    sqa = load_dataset("ChilleD/StrategyQA")["test"]
    return [{"id": f"sqa-test-{r['qid']}", "question": norm(r["question"]),
             "answer": "Yes" if r["answer"] else "No",
             "source": "strategyqa", "answer_type": "yes_no"} for r in sqa]


_PQ_ANS = re.compile(r"###The answer is:\s*(True|False)", re.I)


def parse_prontoqa(prompt: str):
    left = prompt.split("###Response:")[0]
    q = norm(left.replace("###Context:", "").strip())
    m = _PQ_ANS.search(prompt)
    return (q, m.group(1).capitalize()) if (m and q) else None


def build_prontoqa() -> list[dict]:
    from datasets import load_dataset
    pq = load_dataset("longface/prontoqa-train", split="train")
    parsed = [x for x in (parse_prontoqa(r["prompt"]) for r in pq) if x]
    pick = random.Random(123).sample(held_out(len(parsed)), EVAL_N)
    return [{"id": f"prontoqa-src-{i:05d}", "question": parsed[i][0], "answer": parsed[i][1],
             "source": "prontoqa", "answer_type": "true_false"} for i in pick]


def hotpot_context(r: dict) -> str:
    paras = "\n\n".join(f"{t}: {norm(''.join(s))}"
                        for t, s in zip(r["context"]["title"], r["context"]["sentences"]))
    return f"{paras}\n\nQuestion: {norm(r['question'])}"


def build_hotpotqa() -> list[dict]:
    from datasets import load_dataset
    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    pick = random.Random(123).sample(held_out(len(hp)), EVAL_N)
    return [{"id": f"hotpotqa-{hp[i]['id']}", "question": hotpot_context(hp[i]),
             "answer": norm(hp[i]["answer"]), "source": "hotpotqa",
             "answer_type": "span_or_yesno"} for i in pick]


BUILDERS = {"strategyqa": build_strategyqa, "prontoqa": build_prontoqa, "hotpotqa": build_hotpotqa}


def check_disjoint(sets: dict[str, list[dict]]) -> None:
    seed = load_seed()
    seed_q = {}
    for r in seed:
        seed_q.setdefault(r["source"], set()).add(norm(r["question"]))
    for bench, rows in sets.items():
        overlap = [r["id"] for r in rows if norm(r["question"]) in seed_q.get(bench, set())]
        status = "OK" if not overlap else f"OVERLAP {len(overlap)} e.g. {overlap[:3]}"
        print(f"[disjoint] {bench:10s} eval={len(rows):4d} seed={len(seed_q.get(bench, ())):4d} {status}")
        if overlap:
            raise SystemExit(f"{bench}: {len(overlap)} eval rows overlap the training seed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data", help="output dir for <bench>_eval.jsonl")
    ap.add_argument("--benches", default=",".join(BUILDERS), help="comma list subset")
    ap.add_argument("--check-disjoint", action="store_true",
                    help="read the vendored seed and assert no question overlap")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for bench in args.benches.split(","):
        rows = BUILDERS[bench]()
        (out / f"{bench}_eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"{bench}: {len(rows)} rows -> {out / f'{bench}_eval.jsonl'}")
        sets[bench] = rows
    if args.check_disjoint:
        check_disjoint(sets)


if __name__ == "__main__":
    main()
