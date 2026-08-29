"""Hermetic unit tests for the 3-task QA utility scorer (no network)."""
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from qa_eval.scoring import (  # noqa: E402
    aggregate,
    em_binary,
    extract_boxed,
    hotpot_scores,
    norm_squad,
    score_generations_file,
    score_row,
)


# ------------------------------------------------------------ extract_boxed
def test_extract_boxed_last_and_nested():
    assert extract_boxed("first \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed("\\boxed{\\text{Yes}}") == "\\text{Yes}"


def test_extract_boxed_fallback_last_line():
    assert extract_boxed("reasoning...\n\nTherefore: No\n") == "Therefore: No"
    assert extract_boxed("") == ""
    assert extract_boxed("\\boxed{unterminated") == "unterminated"


# ---------------------------------------------------------------- binary EM
def test_em_binary_token_match():
    assert em_binary("Yes", "\\text{Yes}") == 1.0
    assert em_binary("No", "Yes") == 0.0
    assert em_binary("True", "true.") == 1.0
    assert em_binary("False", "The statement is not false") == 1.0  # token present
    assert em_binary("Yes", "") == 0.0


# ---------------------------------------------------------- hotpot EM / F1
def test_norm_squad():
    assert norm_squad("The Pompano  Beach, Florida!") == "pompano beach florida"


def test_hotpot_scores_exact_and_partial():
    assert hotpot_scores("Pompano Beach, Florida", "the Pompano Beach Florida.") == (1.0, 1.0)
    em, f1 = hotpot_scores("Pompano Beach", "Pompano Beach Florida")
    assert em == 0.0 and abs(f1 - 0.8) < 1e-9
    assert hotpot_scores("yes", "no") == (0.0, 0.0)
    assert hotpot_scores("", "") == (1.0, 1.0)


# ---------------------------------------------------------------- score_row
def test_score_row_by_answer_type():
    sqa = {"answer": "Yes", "answer_type": "yes_no", "output": "so \\boxed{Yes}"}
    assert score_row(sqa) == {"pred": "Yes", "em": 1.0}
    hp = {"answer": "Lima, Peru", "source": "hotpotqa", "answer_type": "span_or_yesno",
          "output": "He was born in \\boxed{Lima}"}
    s = score_row(hp)
    assert s["em"] == 0.0 and abs(s["f1"] - 2 / 3) < 1e-9


# ---------------------------------------------------------------- aggregate
def test_aggregate_and_file(tmp_path):
    rows = [
        {"id": "a", "source": "strategyqa", "answer": "Yes", "answer_type": "yes_no", "output": "\\boxed{Yes}"},
        {"id": "b", "source": "strategyqa", "answer": "No", "answer_type": "yes_no", "output": "\\boxed{Yes}"},
    ]
    assert aggregate(rows) == {"n": 2, "acc": 0.5}
    assert aggregate([]) == {"n": 0}
    hp = [{"id": "c", "source": "hotpotqa", "answer": "Paris", "answer_type": "span_or_yesno",
           "output": "\\boxed{Paris, France}"}]
    agg = aggregate(hp)
    assert agg["n"] == 1 and agg["em"] == 0.0 and abs(agg["f1"] - 2 / 3) < 1e-9

    p = tmp_path / "strategyqa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    per_row, agg = score_generations_file(p)
    assert [r["em"] for r in per_row] == [1.0, 0.0]
    assert agg == {"n": 2, "acc": 0.5}
