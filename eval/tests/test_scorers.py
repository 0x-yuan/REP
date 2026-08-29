"""Hermetic unit tests for the answer-scoring library (no network / no datasets).

Each scorer is exercised via get_scorer() and a gold answer wrapped in a
realistic \\boxed{...} generation envelope, plus wrong-answer and (for
JEEBench MCQ(multiple)) partial-credit checks.
"""
import sys
from pathlib import Path

import pytest

# Make the sibling `scorers` package importable when running from anywhere.
EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from scorers import get_scorer, list_scorers  # noqa: E402


def _gen(ans: str) -> str:
    return f"Reasoning ... therefore the final answer is \\boxed{{{ans}}}."


def test_registry_lists_all():
    names = set(list_scorers())
    assert {"gsm8k", "math500", "jeebench", "openthoughts500"}.issubset(names)


# ---------------------------------------------------------------- GSM8K
def test_gsm8k_correct():
    s = get_scorer("gsm8k")
    gold = "She has 6 eggs left.\n#### 6"
    res = s.score(_gen("6"), gold)
    assert res.answer_match == 1.0
    assert res.answer_match_partial == 1.0


def test_gsm8k_wrong():
    s = get_scorer("gsm8k")
    gold = "She has 6 eggs left.\n#### 6"
    res = s.score(_gen("7"), gold)
    assert res.answer_match == 0.0


def test_gsm8k_decimal_fraction_normalization():
    s = get_scorer("gsm8k")
    # 1/2 and 0.5 must normalize equal.
    res = s.score(_gen("1/2"), "#### 0.5")
    assert res.answer_match == 1.0


# ---------------------------------------------------------------- MATH500
def test_math500_correct():
    s = get_scorer("math500")
    res = s.score(_gen("\\frac{1}{2}"), "\\frac{1}{2}")
    assert res.answer_match == 1.0


def test_math500_wrong():
    s = get_scorer("math500")
    res = s.score(_gen("\\frac{1}{3}"), "\\frac{1}{2}")
    assert res.answer_match == 0.0


def test_math500_equivalent_forms():
    s = get_scorer("math500")
    # tfrac vs frac should be treated equal by the Hendrycks fallback / math-verify.
    res = s.score(_gen("\\tfrac{1}{2}"), "\\frac{1}{2}")
    assert res.answer_match == 1.0


# ---------------------------------------------------------------- OpenThoughts (math)
def test_openthoughts_reuses_math500():
    s = get_scorer("openthoughts500")
    res = s.score(_gen("42"), "42")
    assert res.answer_match == 1.0


# ---------------------------------------------------------------- JEEBench
def test_jeebench_mcq_single_correct():
    s = get_scorer("jeebench")
    res = s.score(_gen("B"), "B", meta={"type": "MCQ"})
    assert res.answer_match == 1.0


def test_jeebench_mcq_single_wrong():
    s = get_scorer("jeebench")
    res = s.score(_gen("C"), "B", meta={"type": "MCQ"})
    assert res.answer_match == 0.0


def test_jeebench_integer():
    s = get_scorer("jeebench")
    res = s.score(_gen("7"), "7", meta={"type": "Integer"})
    assert res.answer_match == 1.0


def test_jeebench_numeric_tolerance():
    s = get_scorer("jeebench")
    res = s.score(_gen("3.14"), "3.14", meta={"type": "Numeric"})
    assert res.answer_match == 1.0


def test_jeebench_mcq_multiple_partial_credit():
    """gold 'ABC', pred 'AB' → strict answer_match 0 but partial > 0."""
    s = get_scorer("jeebench")
    res = s.score(_gen("AB"), "ABC", meta={"type": "MCQ(multiple)"})
    assert res.answer_match == 0.0
    assert res.answer_match_partial > 0.0


def test_jeebench_mcq_multiple_exact():
    s = get_scorer("jeebench")
    res = s.score(_gen("ABC"), "ABC", meta={"type": "MCQ(multiple)"})
    assert res.answer_match == 1.0
    assert res.answer_match_partial == 1.0


def test_jeebench_mcq_multiple_wrong_superset_scores_zero():
    """pred with a letter not in gold is not a subset → 0 partial."""
    s = get_scorer("jeebench")
    res = s.score(_gen("ABD"), "ABC", meta={"type": "MCQ(multiple)"})
    assert res.answer_match == 0.0
    assert res.answer_match_partial == 0.0


def test_jeebench_unknown_type():
    s = get_scorer("jeebench")
    res = s.score(_gen("B"), "B", meta={"type": "Essay"})
    assert res.answer_match == 0.0


def test_unknown_scorer_raises():
    with pytest.raises(KeyError):
        get_scorer("does-not-exist")
