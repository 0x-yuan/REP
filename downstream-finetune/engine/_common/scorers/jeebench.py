from __future__ import annotations

import re
from typing import Any

from ._boxed import extract_last_boxed_content
from .base import ScorerResult

_MCQ_LETTERS = ("A", "B", "C", "D")
_MCQ_LETTER_RE = re.compile(r"[A-D]")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FINAL_ANSWER_RE = re.compile(r"(?i)final\s+answer\s*[:\-]\s*([^\n\r]+?)(?:\.|$)")
_OPTION_LIST_RE = re.compile(
    r"(?i)(?:options?|answers?)[^A-Za-z0-9]{0,8}([A-D](?:\s*[, ]\s*[A-D]){0,3})"
)
_VALID_TYPES = {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}
_NUMERIC_TOLERANCE = 1e-2


def _letters_from_string(text: str) -> set[str]:
    if text is None:
        return set()
    found = set(_MCQ_LETTER_RE.findall(text.upper()))
    return {c for c in _MCQ_LETTERS if c in found}


def _extract_mcq_pred(generation: str) -> str | None:
    text = str(generation or "")
    boxed = extract_last_boxed_content(text)
    if boxed is not None:
        letters = _letters_from_string(boxed)
        if letters:
            return "".join(sorted(letters))
    match = _FINAL_ANSWER_RE.search(text)
    if match is not None:
        letters = _letters_from_string(match.group(1))
        if letters:
            return "".join(sorted(letters))
    options_match = _OPTION_LIST_RE.search(text)
    if options_match is not None:
        letters = _letters_from_string(options_match.group(1))
        if letters:
            return "".join(sorted(letters))
    return None


def _extract_numeric_pred(generation: str) -> str | None:
    text = str(generation or "")
    boxed = extract_last_boxed_content(text)
    if boxed is not None:
        match = _NUMBER_RE.search(boxed)
        if match is not None:
            return match.group(0)
    match = _FINAL_ANSWER_RE.search(text)
    if match is not None:
        number_match = _NUMBER_RE.search(match.group(1))
        if number_match is not None:
            return number_match.group(0)
    matches = list(_NUMBER_RE.finditer(text))
    if matches:
        return matches[-1].group(0)
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_mcq(pred: str | None, gold: str) -> tuple[float, float]:
    gold_set = _letters_from_string(gold)
    pred_set = _letters_from_string(pred or "")
    match = float(pred_set == gold_set and len(gold_set) > 0)
    return match, match


def _score_mcq_multiple(pred: str | None, gold: str) -> tuple[float, float]:
    gold_set = _letters_from_string(gold)
    pred_set = _letters_from_string(pred or "")
    if pred_set == gold_set and len(gold_set) > 0:
        return 1.0, 1.0
    if pred_set and pred_set.issubset(gold_set):
        return 0.0, 0.25 * len(pred_set)
    return 0.0, 0.0


def _score_numeric(pred: str | None, gold: str) -> tuple[float, float]:
    pred_value = _to_float(pred)
    gold_value = _to_float(gold)
    if pred_value is None or gold_value is None:
        return 0.0, 0.0
    match = float(abs(pred_value - gold_value) <= _NUMERIC_TOLERANCE)
    return match, match


class JEEBenchScorer:
    name = "jeebench"

    def extract_pred(self, generation: str, meta: dict[str, Any] | None = None) -> str | None:
        question_type = (meta or {}).get("type", "")
        if question_type in {"MCQ", "MCQ(multiple)"}:
            return _extract_mcq_pred(generation)
        return _extract_numeric_pred(generation)

    def extract_gold(self, gold: str, meta: dict[str, Any] | None = None) -> str | None:
        question_type = (meta or {}).get("type", "")
        text = str(gold or "").strip()
        if text == "":
            return None
        if question_type in {"MCQ", "MCQ(multiple)"}:
            letters = _letters_from_string(text)
            return "".join(sorted(letters)) if letters else None
        match = _NUMBER_RE.search(text)
        return match.group(0) if match is not None else None

    def score(
        self,
        generation: str,
        gold: str,
        meta: dict[str, Any] | None = None,
    ) -> ScorerResult:
        meta = dict(meta or {})
        question_type = meta.get("type", "")
        if question_type not in _VALID_TYPES:
            return ScorerResult(
                answer_match=0.0,
                answer_match_partial=0.0,
                extracted_pred=None,
                extracted_gold=None,
                extra={"jeebench_unknown_type": 1.0},
            )
        pred_value = self.extract_pred(generation, meta)
        gold_value = self.extract_gold(gold, meta)
        if question_type == "MCQ":
            match, partial = _score_mcq(pred_value, gold)
        elif question_type == "MCQ(multiple)":
            match, partial = _score_mcq_multiple(pred_value, gold)
        else:
            match, partial = _score_numeric(pred_value, gold)
        return ScorerResult(
            answer_match=match,
            answer_match_partial=partial,
            extracted_pred=pred_value,
            extracted_gold=gold_value,
            extra={
                f"jeebench_type_{question_type.replace('(', '_').replace(')', '')}_match": match,
            },
        )
