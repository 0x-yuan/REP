"""OpenThoughts scorer — math-domain answer matcher.

OpenThoughts-114k is overwhelmingly math + reasoning, with answers
extractable via the same `\\boxed{...}` / final-answer pattern as
MATH500. We reuse the Math500 extract + math-verify + Hendrycks logic
exactly; only the registered name differs.
"""
from __future__ import annotations

from .math500 import Math500Scorer


class OpenThoughtsScorer(Math500Scorer):
    name = "openthoughts500"
