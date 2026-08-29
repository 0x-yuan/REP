"""Pure-python scorer for the 3-task QA utility eval (no network, no torch).

Protocol (paper "downstream utility on new reasoning categories"):
  * the student is prompted with ``BASE_SYS`` + the question, reasons step by
    step and puts its final answer in ``\\boxed{}``;
  * prediction = last brace-balanced ``\\boxed{...}`` (fallback: last non-empty
    line);
  * StrategyQA (yes/no) and ProntoQA (true/false) -> exact match: the gold
    token must appear among the alphabetic tokens of the prediction;
  * HotpotQA (open-book distractor setting) -> SQuAD-style EM and token F1
    after SQuAD normalisation (lowercase, strip punctuation + articles).

Row schema (one JSON object per line):
  {"question": str, "answer": str, "source": "strategyqa"|"prontoqa"|"hotpotqa",
   "answer_type": "yes_no"|"true_false"|"span_or_yesno", "output": str}
``output`` is the raw model generation. Extra keys are ignored.
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable

BASE_SYS = ("You are a helpful assistant. Reason step by step, then give your "
            "final answer within \\boxed{}.")

BENCHES = ("strategyqa", "prontoqa", "hotpotqa")
_BINARY_TYPES = {"yes_no", "true_false"}
_PUNCT = set(string.punctuation)


# ----------------------------------------------------------------- extraction
def extract_boxed(text: str) -> str:
    """Content of the LAST ``\\boxed{...}`` (brace-balanced). If there is no
    ``\\boxed``, fall back to the last non-empty line of the text."""
    text = text or ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    i = idx + len("\\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


# -------------------------------------------------------------- normalisation
def norm_squad(s: str) -> str:
    """SQuAD answer normalisation: lowercase, drop punctuation, drop articles,
    collapse whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


# ------------------------------------------------------------------- metrics
def em_binary(gold: str, pred: str) -> float:
    """Yes/No and True/False EM: gold word appears among the alphabetic tokens
    of the prediction (so ``\\boxed{\\text{Yes}}`` and ``Yes.`` both count)."""
    g = (gold or "").strip().lower()
    toks = re.findall(r"[a-z]+", (pred or "").lower())
    return 1.0 if g and g in toks else 0.0


def hotpot_scores(gold: str, pred: str) -> tuple[float, float]:
    """SQuAD/HotpotQA EM and token-F1 on normalised strings."""
    g, p = norm_squad(gold), norm_squad(pred)
    em = 1.0 if g == p else 0.0
    gt, pt = g.split(), p.split()
    if not gt or not pt:
        return em, (1.0 if gt == pt else 0.0)
    common = Counter(gt) & Counter(pt)
    overlap = sum(common.values())
    if overlap == 0:
        return em, 0.0
    prec, rec = overlap / len(pt), overlap / len(gt)
    return em, 2 * prec * rec / (prec + rec)


def score_row(row: dict) -> dict:
    """Score one generation row. Returns the row's metrics:
    ``{"pred", "em"}`` for binary tasks, ``{"pred", "em", "f1"}`` for HotpotQA."""
    pred = extract_boxed(row.get("output") or "")
    atype = row.get("answer_type") or ("span_or_yesno" if row.get("source") == "hotpotqa" else "yes_no")
    if atype in _BINARY_TYPES:
        return {"pred": pred, "em": em_binary(row["answer"], pred)}
    em, f1 = hotpot_scores(row["answer"], pred)
    return {"pred": pred, "em": em, "f1": f1}


def aggregate(rows: Iterable[dict], bench: str | None = None) -> dict:
    """Mean metrics over rows. Binary benches report ``acc``; HotpotQA reports
    ``em`` and ``f1``. ``bench`` defaults to the rows' ``source``."""
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"n": 0}
    bench = bench or rows[0].get("source") or "unknown"
    scored = [score_row(r) for r in rows]
    if bench == "hotpotqa" or "f1" in scored[0]:
        return {"n": n,
                "em": sum(s["em"] for s in scored) / n,
                "f1": sum(s.get("f1", 0.0) for s in scored) / n}
    return {"n": n, "acc": sum(s["em"] for s in scored) / n}


def score_generations_file(path: Path, bench: str | None = None) -> tuple[list[dict], dict]:
    """Score a jsonl of generation rows. Returns (per-row scored, aggregate)."""
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    per_row = [{**{k: r.get(k) for k in ("id", "source", "answer")}, **score_row(r)} for r in rows]
    return per_row, aggregate(rows, bench)
