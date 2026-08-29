from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ScorerResult:
    answer_match: float
    answer_match_partial: float
    extracted_pred: str | None
    extracted_gold: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "answer_match": float(self.answer_match),
            "answer_match_partial": float(self.answer_match_partial),
        }
        for key, value in self.extra.items():
            if isinstance(value, (bool, int, float)):
                metrics[key] = float(value)
        return metrics


class Scorer(Protocol):
    name: str

    def extract_pred(self, generation: str) -> str | None:
        ...

    def extract_gold(self, gold: str) -> str | None:
        ...

    def score(self, generation: str, gold: str, meta: dict[str, Any] | None = None) -> ScorerResult:
        ...
