"""JEEBench scorer implementing the official JEEBench (dair-iitd) protocol.

Used by the multi-bench eval so the Modal container is self-contained.
Do not modify — the scoring logic must match the official protocol.
"""
from .jeebench import JEEBenchScorer  # noqa: F401
from .base import ScorerResult  # noqa: F401
