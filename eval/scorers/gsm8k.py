from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from ._boxed import extract_last_boxed_content
from .base import ScorerResult


_GSM8K_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,./]*)\s*$")
_NUMERIC_OR_FRACTION_RE = re.compile(
    r"[-+]?\d+\s*/\s*\d+|[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
)
_FRACTION_FULLMATCH_RE = re.compile(r"\s*([-+]?\d+)\s*/\s*(\d+)\s*")


def _normalize_decimal_string(value: str) -> str | None:
    cleaned = str(value or "").strip().replace(",", "")
    if cleaned == "":
        return None
    try:
        decimal_value = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def _normalize_fraction_string(value: str) -> str | None:
    match = _FRACTION_FULLMATCH_RE.fullmatch(str(value or ""))
    if match is None:
        return None
    numerator = Decimal(match.group(1))
    denominator = Decimal(match.group(2))
    if denominator == 0:
        return None
    return _normalize_decimal_string(str(numerator / denominator))


def _normalize_numeric_token(token: str) -> str | None:
    fraction = _normalize_fraction_string(token)
    if fraction is not None:
        return fraction
    return _normalize_decimal_string(token)


def _extract_from_text(text: str) -> str | None:
    boxed = extract_last_boxed_content(text)
    if boxed is not None:
        normalized = _normalize_numeric_token(boxed.strip())
        if normalized is not None:
            return normalized
        text = boxed
    matches = list(_NUMERIC_OR_FRACTION_RE.finditer(text))
    if not matches:
        return None
    candidate = matches[-1].group(0).strip()
    return _normalize_numeric_token(candidate)


class GSM8KScorer:
    name = "gsm8k"

    def extract_pred(self, generation: str) -> str | None:
        return _extract_from_text(str(generation or ""))

    def extract_gold(self, gold: str) -> str | None:
        text = str(gold or "")
        match = _GSM8K_GOLD_RE.search(text.strip())
        if match is not None:
            normalized = _normalize_numeric_token(match.group(1).strip())
            if normalized is not None:
                return normalized
        return _extract_from_text(text)

    def score(
        self,
        generation: str,
        gold: str,
        meta: dict[str, Any] | None = None,
    ) -> ScorerResult:
        pred_value = self.extract_pred(generation)
        gold_value = self.extract_gold(gold)
        match = float(
            pred_value is not None
            and gold_value is not None
            and pred_value == gold_value
        )
        return ScorerResult(
            answer_match=match,
            answer_match_partial=match,
            extracted_pred=pred_value,
            extracted_gold=gold_value,
            extra={"answer_gsm8k_exact_match": match},
        )
