"""Hermetic unit tests for the trace-elicitation metrics (no network).

Runs the core scorer on synthetic rows: one structurally-valid
`<think>...</think>...\\boxed{}` generation and one malformed generation,
and checks struct detection, boxed extraction, ROUGE sanity, and aggregation.
"""
import sys
from pathlib import Path

# Make trace_metrics importable when running from anywhere.
EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from trace_metrics.score import (  # noqa: E402
    _extract_last_boxed,
    _norm_answer,
    _rouge_l,
    aggregate,
    extract_r1_r2_answer,
    reassemble_qwen3,
    score_generation,
)


GOOD = (
    "<think>\n"
    "Let me add the fractions. 1/2 + 1/2 = 1. The gold reasoning path.\n"
    "</think>\n"
    "We add the two halves to get one whole.\n"
    "The final answer is \\boxed{1}."
)

MALFORMED = "I think the answer is probably 1 but I never opened a think block."

# r^i: the victim's clean-baseline hidden trace we compare leakage against.
REFERENCE_TRACE = "Let me add the fractions. 1/2 + 1/2 = 1. The gold reasoning path."


# ---------------------------------------------------------------- boxed extractor
def test_extract_last_boxed_simple():
    assert _extract_last_boxed("x \\boxed{1} y") == "1"


def test_extract_last_boxed_nested_braces():
    # The lazy regex would butcher this into '\\frac{77'; the brace-balanced
    # scanner must return the whole nested expression.
    assert _extract_last_boxed("ans: \\boxed{\\frac{77}{100}}") == "\\frac{77}{100}"


def test_extract_last_boxed_returns_last():
    assert _extract_last_boxed("\\boxed{1} ... \\boxed{2}") == "2"


def test_extract_last_boxed_none():
    assert _extract_last_boxed("no box here") is None


def test_norm_answer_prefers_boxed():
    assert _norm_answer("blah \\boxed{42}.") == "42"


# ---------------------------------------------------------------- reassemble + extract
def test_reassemble_qwen3_detects_think():
    canonical, ok = reassemble_qwen3(GOOD)
    assert ok is True
    assert "<think>" in canonical


def test_extract_r1_r2_answer_good():
    canonical, _ = reassemble_qwen3(GOOD)
    r1, r2, answer, structural = extract_r1_r2_answer(canonical)
    assert structural is True
    assert "add the fractions" in r1
    assert "one whole" in r2
    assert answer == "1"


def test_extract_r1_r2_answer_malformed():
    canonical, _ = reassemble_qwen3(MALFORMED)
    r1, r2, answer, structural = extract_r1_r2_answer(canonical)
    assert structural is False
    assert r1 == ""


# ---------------------------------------------------------------- rouge sanity
def test_rouge_l_bounds():
    assert _rouge_l("the cat sat", "the cat sat") == 1.0
    mid = _rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < mid < 1.0
    assert _rouge_l("", "anything") == 0.0


# ---------------------------------------------------------------- score_generation
def test_score_generation_structural_row():
    s = score_generation(GOOD, ri=REFERENCE_TRACE, gold_answer="1", victim="qwen3-32b")
    assert s["structural_success"] is True
    assert s["has_ri"] is True
    assert s["answer_em"] is True
    # r1 is nearly identical to the reference trace → high R01.
    assert s["rouge_l_ri_r1"] > 0.8
    # All ROUGE metrics are within [0, 1].
    for k in ("rouge_l_r1_r2", "rouge_l_ri_r1", "rouge_l_ri_r2"):
        assert 0.0 <= s[k] <= 1.0


def test_score_generation_malformed_row():
    s = score_generation(MALFORMED, ri=REFERENCE_TRACE, gold_answer="1", victim="qwen3-32b")
    assert s["structural_success"] is False
    assert s["rouge_l_r1_r2"] == 0.0


def test_score_generation_without_reference():
    s = score_generation(GOOD, victim="qwen3-32b")
    assert s["has_ri"] is False
    assert s["rouge_l_ri_r2"] == 0.0


def test_aggregate_over_two_rows():
    rows = [
        score_generation(GOOD, ri=REFERENCE_TRACE, gold_answer="1", victim="qwen3-32b"),
        score_generation(MALFORMED, ri=REFERENCE_TRACE, gold_answer="1", victim="qwen3-32b"),
    ]
    agg = aggregate(rows)
    assert agg["n"] == 2
    assert agg["n_struct"] == 1
    assert agg["structural_success_rate"] == 0.5
    # struct-only R01 is taken over the single good row → high.
    assert agg["rir1_struct"] > 0.8
