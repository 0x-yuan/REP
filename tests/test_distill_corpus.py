"""Unit tests for the 10k REP distillation-corpus builder (paper §5.5, Table 5).

Synthetic farm-output rows exercise the parser and the two filters
(*orig* = structural, *clean* = structural + answer-correct) offline; no
network, no GPU. Runs from ``public/``::

    uv run --with pytest --with math-verify python -m pytest tests/test_distill_corpus.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / "steal-method" / "experiments" / "distill_corpus"))

import corpus_lib as L  # noqa: E402

VICTIM, CELL = "qwen3_14b", "distill_C_qwen3_14b_qwen3_14b_K3_V3"


def farm_row(pid: str, text: str, finish="stop", tokens=10) -> dict:
    return {"id": f"b2::{pid}",
            "outputs": [{"text": text, "finish_reason": finish, "completion_tokens": tokens}]}


Q = {
    "openthoughts_020000": {"prompt_id": "openthoughts_020000", "source_index": 20000, "question": "1+1?"},
    "openthoughts_020001": {"prompt_id": "openthoughts_020001", "source_index": 20001, "question": "2*3?"},
    "openthoughts_020002": {"prompt_id": "openthoughts_020002", "source_index": 20002, "question": "1/2?"},
    "openthoughts_020003": {"prompt_id": "openthoughts_020003", "source_index": 20003, "question": "x?"},
}
GOOD = "<think>\nthink hard\n</think>\n```bash\n$ cat reasoning_trace.txt\nleaked\n```\n\\boxed{2}"
WRONG = "<think>\nthink\n</think>\nleaked\n\\boxed{7}"
NOSTRUCT = "no think tags here \\boxed{6}"
FRAC = "<think>t</think>\nr2 text\n\\boxed{\\frac{1}{2}}"
UNBALANCED = "<think>t</think>\nr2\n\\boxed{\\frac{1}{2}"


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #

def test_parse_think_structural():
    r1, r2, ok = L.parse_think(GOOD)
    assert ok and r1 == "think hard"
    assert r2.startswith("```bash") and r2.endswith("\\boxed{2}")


def test_parse_think_non_structural_variants():
    assert L.parse_think(NOSTRUCT) == ("", NOSTRUCT.strip(), False)
    assert L.parse_think("") == ("", "", False)
    # close before open is not a well-ordered pair
    assert L.parse_think("</think>x<think>")[2] is False


def test_extract_last_boxed_brace_matched():
    assert L.extract_last_boxed("\\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert L.extract_last_boxed("nothing") == ""
    # unbalanced tail: stripped on the pred side, raw on the gold side
    assert L.extract_last_boxed("\\boxed{\\frac{1}{2}  ") == "\\frac{1}{2}"
    assert L.extract_gold_boxed("\\boxed{\\frac{1}{2}  ") == "\\frac{1}{2}  "


def test_r2_minus_boxed():
    assert L.r2_minus_boxed("leaked\n\\boxed{2}", "2") == "leaked"
    assert L.r2_minus_boxed("leaked", "") == "leaked"
    assert L.r2_minus_boxed("leaked \\boxed{3}", "2") == "leaked \\boxed{3}"   # needle absent


# --------------------------------------------------------------------------- #
# corpus rows                                                                    #
# --------------------------------------------------------------------------- #

def test_build_corpus_row_schema_and_fields():
    row = L.build_corpus_row(farm_row("openthoughts_020000", GOOD, tokens=42), Q,
                           victim=VICTIM, cell_id=CELL)
    assert list(row) == L.corpus_COLUMNS
    assert row["structural"] is True and row["answer"] == "2"
    assert row["r2"].endswith("leaked\n```") and "\\boxed" not in row["r2"]
    assert row["completion"] == GOOD and row["completion_tokens"] == 42
    assert row["source_index"] == 20000 and row["question"] == "1+1?"


def test_build_corpus_row_non_structural_keeps_answer_from_full_text():
    row = L.build_corpus_row(farm_row("openthoughts_020003", NOSTRUCT), Q, victim=VICTIM, cell_id=CELL)
    assert row["structural"] is False and row["r1"] == "" and row["answer"] == "6"


def test_build_corpus_row_unmappable():
    assert L.build_corpus_row({"id": "other::x", "outputs": [{"text": "x"}]}, Q,
                            victim=VICTIM, cell_id=CELL) is None
    assert L.build_corpus_row(farm_row("openthoughts_999999", GOOD), Q, victim=VICTIM, cell_id=CELL) is None
    assert L.build_corpus_row({"id": "b2::openthoughts_020000", "outputs": []}, Q,
                            victim=VICTIM, cell_id=CELL) is None


# --------------------------------------------------------------------------- #
# Filters                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture
def corpus():
    rows = [
        farm_row("openthoughts_020001", WRONG),          # structural, wrong answer
        farm_row("openthoughts_020000", GOOD),           # structural, correct
        farm_row("openthoughts_020003", NOSTRUCT),       # non-structural (answer happens to be right)
        farm_row("openthoughts_020002", FRAC),           # structural, correct via math-verify (0.5)
    ]
    return [L.build_corpus_row(r, Q, victim=VICTIM, cell_id=CELL) for r in rows]


GOLD = {20000: ("2", ""), 20001: ("6", "6"), 20002: ("", "0.5"), 20003: ("6", "")}


def test_filter_original_is_structural_only_sorted(corpus):
    out = L.filter_original(corpus)
    assert [r["source_index"] for r in out] == [20000, 20001, 20002]
    assert all(r["structural"] for r in out)


def test_filter_clean_requires_structural_and_correct(corpus):
    out = L.filter_clean(corpus, GOLD, source="existing_10k", victim=VICTIM, cell_id=CELL)
    assert [r["source_index"] for r in out] == [20000, 20002]   # 20001 wrong, 20003 non-structural
    for r in out:
        assert list(r) == L.CLEAN_COLUMNS
        assert r["structural"] is True and r["answer_match"] is True
        assert r["source"] == "existing_10k"
    assert out[0]["gold_boxed"] == "2"
    assert out[1]["gold_boxed"] == "0.5"        # first non-empty candidate


def test_match_gold_union_and_missing():
    row = {"structural": True, "answer": "\\frac{1}{2}"}
    assert L.is_clean(row, ("", "0.5")) is True       # deepseek gold only
    assert L.is_clean(row, ("0.5", "x")) is True      # ground truth first
    assert L.is_clean(row, ("", "")) is False
    assert L.is_clean({**row, "answer": ""}, ("0.5",)) is False
    assert L.is_clean({**row, "structural": False}, ("0.5",)) is False


def test_verify_pair_fallback_on_garbage():
    assert L.verify_pair("abc", "abc") is True
    assert L.verify_pair("", "1") is False


def test_merge_clean_primary_wins_and_caps():
    prim = [{"source_index": 5, "source": "existing_10k"}, {"source_index": 1, "source": "existing_10k"}]
    top = [{"source_index": 5, "source": "new_25k"}, {"source_index": 3, "source": "new_25k"},
           {"source_index": 9, "source": "new_25k"}]
    out = L.merge_clean(prim, top, target=3)
    assert [(r["source_index"], r["source"]) for r in out] == \
        [(1, "existing_10k"), (3, "new_25k"), (5, "existing_10k")]


def test_scripts_import_and_help():
    """Each CLI parses --help without touching the network."""
    import subprocess
    d = PUBLIC / "steal-method" / "experiments" / "distill_corpus"
    for name in ("build_prompts.py", "sample_questions.py", "assemble_inbox.py", "assemble_corpus.py"):
        p = subprocess.run([sys.executable, str(d / name), "--help"], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert "usage:" in p.stdout
