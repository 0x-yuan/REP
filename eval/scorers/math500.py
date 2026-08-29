from __future__ import annotations

import re
from typing import Any

from ._boxed import extract_last_boxed_content
from .base import ScorerResult

_FINAL_ANSWER_RE = re.compile(
    r"(?i)final\s+answer\s*[:\-]\s*([^\n\r]+?)(?:\.|$)"
)


def _strip_latex_envelope(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("$$") and cleaned.endswith("$$"):
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith("$") and cleaned.endswith("$"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _math_verify_equal(pred: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
    except ImportError:  # pragma: no cover - dependency missing
        return False
    try:
        gold_parsed = parse(f"${gold}$")
        pred_parsed = parse(f"${pred}$")
        if not gold_parsed or not pred_parsed:
            gold_parsed = parse(gold)
            pred_parsed = parse(pred)
        if not gold_parsed or not pred_parsed:
            return False
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False


def _hendrycks_strip(text: str) -> str:
    """Hendrycks `_strip_string` simplified port — used as a fallback equivalence check."""
    s = text
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = re.sub(r"\\text\{[^{}]*?\}", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s.startswith("."):
        s = "0" + s
    if "=" in s:
        parts = s.split("=", 1)
        if len(parts[0]) <= 2:
            s = parts[1]
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = s.replace(" ", "")
    if s == "0.5":
        s = "\\frac{1}{2}"
    return s


def _hendrycks_equal(pred: str, gold: str) -> bool:
    try:
        return _hendrycks_strip(pred) == _hendrycks_strip(gold)
    except Exception:
        return False


class Math500Scorer:
    name = "math500"

    def extract_pred(self, generation: str) -> str | None:
        text = str(generation or "")
        boxed = extract_last_boxed_content(text)
        if boxed is not None:
            return _strip_latex_envelope(boxed.strip())
        match = _FINAL_ANSWER_RE.search(text)
        if match is not None:
            return _strip_latex_envelope(match.group(1).strip())
        return None

    def extract_gold(self, gold: str) -> str | None:
        text = str(gold or "").strip()
        if text == "":
            return None
        boxed = extract_last_boxed_content(text)
        if boxed is not None:
            return _strip_latex_envelope(boxed.strip())
        return _strip_latex_envelope(text)

    def score(
        self,
        generation: str,
        gold: str,
        meta: dict[str, Any] | None = None,
    ) -> ScorerResult:
        pred_value = self.extract_pred(generation)
        gold_value = self.extract_gold(gold)
        if pred_value is None or gold_value is None:
            return ScorerResult(
                answer_match=0.0,
                answer_match_partial=0.0,
                extracted_pred=pred_value,
                extracted_gold=gold_value,
                extra={
                    "answer_math_verify_match": 0.0,
                    "answer_hendrycks_match": 0.0,
                },
            )
        mv_equal = _math_verify_equal(pred_value, gold_value)
        hd_equal = _hendrycks_equal(pred_value, gold_value)
        match = float(mv_equal or hd_equal)
        return ScorerResult(
            answer_match=match,
            answer_match_partial=match,
            extracted_pred=pred_value,
            extracted_gold=gold_value,
            extra={
                "answer_math_verify_match": float(mv_equal),
                "answer_hendrycks_match": float(hd_equal),
            },
        )
