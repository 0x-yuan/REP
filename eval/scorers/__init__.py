from __future__ import annotations

from .base import Scorer, ScorerResult
from .gsm8k import GSM8KScorer
from .jeebench import JEEBenchScorer
from .math500 import Math500Scorer
from .openthoughts import OpenThoughtsScorer

_REGISTRY: dict[str, type[Scorer]] = {
    "gsm8k": GSM8KScorer,
    "math500": Math500Scorer,
    "jeebench": JEEBenchScorer,
    "openthoughts500": OpenThoughtsScorer,
}


def get_scorer(name: str) -> Scorer:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown scorer '{name}'. Registered scorers: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]()


def list_scorers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Scorer",
    "ScorerResult",
    "GSM8KScorer",
    "Math500Scorer",
    "JEEBenchScorer",
    "OpenThoughtsScorer",
    "get_scorer",
    "list_scorers",
]
