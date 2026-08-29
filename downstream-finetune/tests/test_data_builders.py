"""Offline tests for the Table-2 control / oracle teacher-data builders.

    cd public && uv run --with pytest --with math-verify python -m pytest downstream-finetune/tests/test_data_builders.py -q
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

DB = Path(__file__).resolve().parents[1] / "data_builders"
sys.path.insert(0, str(DB))

import _builders as B  # noqa: E402

META = {"source_index": 20000, "question": "What is 1+1?", "gold_boxed": "2"}


def rec(text: str, finish="stop", tokens=5, error=None) -> dict:
    r = {"id": "openthoughts_020000",
         "outputs": [{"text": text, "finish_reason": finish, "completion_tokens": tokens}]}
    if error:
        r["error"] = error
    return r


# --------------------------------------------------------------------------- #
# Oracle                                                                       #
# --------------------------------------------------------------------------- #

def test_extract_think_block_both_layouts():
    assert B.extract_think_block("<think>\nreason\n</think>\nans") == ("reason", "ans")
    # opening tag lived in the prompt (served Qwen3): output starts with the trace
    assert B.extract_think_block("reason\n</think>\n  ans") == ("reason", "ans")
    assert B.extract_think_block("no tags") == ("", "no tags")


def test_extract_last_boxed_none_on_unbalanced():
    assert B.extract_last_boxed("\\boxed{1} \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert B.extract_last_boxed("\\boxed{\\frac{1}{2}") is None
    assert B.extract_last_boxed("plain") is None


def test_oracle_row_happy_path():
    row, why = B.oracle_row(rec("<think>\nadd them\n</think>\nSo \\boxed{2}", tokens=9), META,
                            victim="qwen3_14b")
    assert why is None
    assert row["r1"] == "add them" and row["r2"] == "add them" and row["answer"] == "2"
    assert row["completion"] == "<think>\nadd them\n</think>\n\nadd them\n\n**Final Answer**\n\\boxed{2}\n"
    assert row["structural"] is True and row["answer_match"] is True
    assert row["completion_tokens"] == 9 and row["finish_reason"] == "stop"
    assert row["victim"] == "qwen3_14b" and row["cell_id"] == "ideal_no_attack_qwen3_14b"
    assert row["source"] == "ideal_no_attack" and row["source_index"] == 20000
    # the synthesized completion round-trips through the same extractor
    r1, post = B.extract_think_block(row["completion"])
    assert r1 == "add them" and B.extract_last_boxed(post) == "2"


def test_oracle_row_wrong_answer_kept_but_flagged():
    row, why = B.oracle_row(rec("<think>x</think>\n\\boxed{3}"), META, victim="qwen3_14b")
    assert why is None and row["answer_match"] is False


@pytest.mark.parametrize("text,reason", [
    ("no trace \\boxed{2}", "no_think"),
    ("<think>only trace</think>", "no_think"),         # empty post-think
    ("<think>t</think>\nno box", "no_boxed"),
    ("<think>t</think>\n\\boxed{unbalanced", "no_boxed"),
])
def test_oracle_row_rejects(text, reason):
    row, why = B.oracle_row(rec(text), META, victim="qwen3_14b")
    assert row is None and why == reason


def test_oracle_row_error_and_empty_outputs():
    assert B.oracle_row(rec("<think>t</think>\\boxed{2}", error="boom"), META, victim="v")[1] == "row_error"
    assert B.oracle_row({"id": "x", "outputs": []}, META, victim="v")[1] == "row_error"


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #

def test_summary_messages_contract():
    msgs = B.summary_messages("Q?", "long trace")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == B.SUMMARY_SYSTEM
    u = msgs[1]["content"]
    assert "# Problem\nQ?\n" in u and "# Verbose reasoning trace\nlong trace\n" in u
    assert u.rstrip().endswith("# Compressed solution")
    assert "`\\boxed{...}`" in u              # literal braces survive .format


def test_extract_summary():
    s = B.extract_summary({"id": "p", "model": "m", "outputs": [
        {"text": "  sol \\boxed{1} ", "completion_tokens": 7, "finish_reason": "length"}]}, "dflt")
    assert s == {"summary": "sol \\boxed{1}", "summary_completion_tokens": 7,
                 "summary_finish_reason": "length", "summary_model": "m"}
    assert B.extract_summary({"outputs": [{"text": "   "}]}, "d") is None
    assert B.extract_summary({"outputs": []}, "d") is None
    assert B.extract_summary({"outputs": [{"text": "x"}]}, "dflt")["summary_model"] == "dflt"


# --------------------------------------------------------------------------- #
# Variants                                                                     #
# --------------------------------------------------------------------------- #

SRC = {"prompt_id": "p", "source_index": 1, "question": " Q? ", "gold_boxed": "2", "answer": "2",
       "r1": "trace ending \\boxed{2}", "r2": "trace", "completion": "<think>c</think>",
       "summary": "short solution", "structural": True, "source": "ideal_no_attack"}


def test_make_target_all_kinds():
    assert B.make_target(SRC, "answer_only") == "**Final Answer**\n\\boxed{2}"
    assert B.make_target(SRC, "summary_answer") == "short solution\n\n**Final Answer**\n\\boxed{2}"
    assert B.make_target(SRC, "ideal_think_answer") == "trace ending \\boxed{2}"   # already boxed
    # gold already wrapped is not double-wrapped
    assert B.make_target({**SRC, "gold_boxed": "\\boxed{5}"}, "answer_only") == "**Final Answer**\n\\boxed{5}"
    # gold falls back to `answer`
    assert B.make_target({**SRC, "gold_boxed": ""}, "answer_only") == "**Final Answer**\n\\boxed{2}"
    with pytest.raises(ValueError):
        B.make_target(SRC, "bogus")


def test_make_target_none_cases():
    assert B.make_target({**SRC, "gold_boxed": "", "answer": ""}, "answer_only") is None
    assert B.make_target({**SRC, "summary": None}, "summary_answer") is None
    assert B.make_target({**SRC, "summary": "  "}, "summary_answer") is None
    assert B.make_target({**SRC, "r1": ""}, "ideal_think_answer") is None


def test_variant_row_engine_contract():
    row, why = B.variant_row(SRC, "summary_answer", victim="qwen3_32b")
    assert why is None
    target = "short solution\n\n**Final Answer**\n\\boxed{2}"
    assert row["completion"] == "<think>\n</think>\n\n" + target
    assert row["completion"].split("</think>", 1)[1].strip() == target   # loader extraction
    assert row["question"] == "Q?" and row["structural"] is True
    assert row["victim"] == "qwen3_32b" and row["teacher"] == "qwen3_32b"
    assert row["variant"] == "summary_answer" and row["target"] == target
    assert row["original_r1"] == SRC["r1"] and row["original_completion"] == SRC["completion"]


@pytest.mark.parametrize("patch,reason", [
    ({"structural": False}, "struct_drop"),
    ({"question": "  "}, "empty_q"),
    ({"summary": ""}, "no_target"),
])
def test_variant_row_rejects(patch, reason):
    row, why = B.variant_row({**SRC, **patch}, "summary_answer", victim="qwen3_14b")
    assert row is None and why == reason


def test_cli_help():
    for name in ("build_oracle_corpus.py", "build_summary_dataset.py", "build_variant_datasets.py"):
        p = subprocess.run([sys.executable, str(DB / name), "--help"], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert "usage:" in p.stdout
