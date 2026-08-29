"""Fidelity scorer: ROUGE-L(r2, ri) with the paper's structural split.

Byte-identical ROUGE / structural logic to the paper's scorer (rouge_score,
use_stemmer=False), without the slow math-verify answer match.

  fidelity = rir2_rouge_l = ROUGE-L(r2, ri)   (source_reasoning_rouge_l_r2)

Structure: r1 = first <think>...</think> block; the trailing text after the last
block is split into r2 (everything but the last non-empty line) and the answer
(the last non-empty line). structural_success = r1, r2 and answer all present.

ROUGE-L is O(|pred|*|ref|); both sides are capped to the first 4000 whitespace
tokens so pathological 100K-char traces stay tractable (median traces ~3K
tokens are unaffected; applied uniformly so numbers stay comparable).
"""
from __future__ import annotations

import re

from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ROUGE_WORD_CAP = 4000


def _cap(t: str) -> str:
    parts = t.split()
    return " ".join(parts[:_ROUGE_WORD_CAP]) if len(parts) > _ROUGE_WORD_CAP else t


def _rouge_l_f1(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    return _ROUGE.score(_cap(pred), _cap(ref))["rougeL"].fmeasure


def _strip_think_tags(t: str) -> str:
    return t.replace("<think>", "").replace("</think>", "").strip()


def _extract_structure(text):
    matches = list(_THINK_BLOCK_RE.finditer(text or ""))
    blocks = [m.group(0).strip() for m in matches]
    if not matches:
        return blocks, (text or "").strip()
    return blocks, text[matches[-1].end():].strip()


def _split_trailing_r2_and_answer(trailing):
    cleaned = (trailing or "").strip()
    if not cleaned:
        return "", ""
    lines = cleaned.splitlines()
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonempty:
        return "", ""
    ans_idx = nonempty[-1]
    return "\n".join(lines[:ans_idx]).strip(), lines[ans_idx].strip()


def score_fidelity(text: str, gold_ri: str) -> dict:
    """Return structural + the three ROUGE-L values for one generation."""
    ri = _strip_think_tags(gold_ri or "")
    blocks, trailing = _extract_structure(text or "")
    r1 = blocks[0] if blocks else ""
    r1_inner = _strip_think_tags(r1)
    r2, gen_answer = _split_trailing_r2_and_answer(trailing)
    structural = 1.0 if (r1 and r2 and gen_answer) else 0.0
    return {
        "structural_success": structural,
        "rir1_rouge_l": _rouge_l_f1(r1_inner, ri),
        "rir2_rouge_l": _rouge_l_f1(r2, ri),          # <- fidelity metric
        "r1r2_rouge_l": _rouge_l_f1(r1_inner, r2),
        "r1_len_chars": len(r1_inner),
        "r2_len_chars": len(r2),
        "n_think_blocks": len(blocks),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score one generation file: jsonl rows {idx, text}; ref json from make_ref.py")
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    import json
    ref = json.load(open(a.ref))
    vals = []
    for line in open(a.outputs):
        if line.strip():
            r = json.loads(line)
            if r.get("text") is not None and r["idx"] in ref:
                vals.append(score_fidelity(r["text"], ref[r["idx"]]["ri"]))
    n = len(vals)
    print(f"n={n} fidelity={sum(v['rir2_rouge_l'] for v in vals) / n:.4f} "
          f"structural={sum(v['structural_success'] for v in vals) / n:.4f}")
