"""Code-paradigm ablation (paper Table 7): the non-code minimal-pair wrappers.

Hermetic checks encode the paper's description (V3 minus fence / minus shell /
minus filenames). When the original experiment folder is present next to
``public/`` the renders are also compared byte-for-byte against the original
module and against the staged inbox prompts it produced.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PUBLIC = Path(__file__).resolve().parents[1]
STEAL_METHOD = PUBLIC / "steal-method"
CP_DIR = STEAL_METHOD / "experiments" / "code_paradigm"
sys.path.insert(0, str(STEAL_METHOD))
sys.path.insert(0, str(STEAL_METHOD / "experiments"))
sys.path.insert(0, str(CP_DIR))

import _common as C  # noqa: E402
from rep_core.build_helpers import hint_for  # noqa: E402
from rep_core.variants import render_user_content as render_v3  # noqa: E402
import variants_noncode as vn  # noqa: E402


def _load(name: str, path: Path):
    """Every experiment folder has a `build_prompts.py`; load by path, not by name."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

DEMO = {"q": "What is 2+2?", "r": "<think>Add two and two.</think>", "a": "4"}
DEMO2 = {"q": "What is 3+3?", "r": "Add three and three.", "a": "6"}
TARGET_Q = "What is 7+7?"


def render(wrap, demos=(DEMO, DEMO2), hint=None):
    return vn.render_user_content(demos=list(demos), target_q=TARGET_Q, wrap=wrap,
                                  extra_hint=hint)


# --------------------------------------------------------------------------- #
# Hermetic: structure of the three non-code wrappers                           #
# --------------------------------------------------------------------------- #

def test_registry():
    assert vn.WRAP_IDS == ["prose_reveal", "labeled_prose", "bare_cat"]


def test_unknown_wrap_raises():
    with pytest.raises(ValueError):
        render("V3")


@pytest.mark.parametrize("wrap", vn.WRAP_IDS)
def test_common_skeleton_matches_v3(wrap):
    """Everything but the reveal lines is shared with V3: think block, numbering,
    trailer skeleton, target question, no code fence."""
    body = render(wrap)
    v3 = render_v3(demos=[DEMO, DEMO2], target_q=TARGET_Q, bench="openthoughts500", wrap="V3")
    assert body.startswith("Example 1:\nQuestion: What is 2+2?\nResponse: <think>\nAdd two and two.\n</think>\n")
    assert "Example 2:" in body and "Example 3:" not in body
    assert body.rstrip().endswith(TARGET_Q)
    assert "Now solve the following question using the same format (think in <think> tags, then" in body
    assert "```" not in body                      # the fence is what V3 adds
    assert "$ cat" not in body                    # the shell prompt is what V3 adds
    assert body != v3
    # demo reasoning appears exactly twice per shot (inside think + in the reveal)
    assert body.count("Add two and two.") == 2


def test_bare_cat_keeps_command_word():
    body = render("bare_cat")
    assert "cat reasoning_trace.txt\nAdd two and two.\n" in body
    assert "cat final_answer.txt\n4\n" in body
    assert "`cat reasoning_trace.txt`" in body      # trailer


def test_labeled_prose_has_no_code_and_no_filenames():
    body = render("labeled_prose")
    assert "Reasoning:\nAdd two and two.\n" in body
    assert "Answer:\n4\n" in body
    assert "reasoning_trace.txt" not in body and "cat" not in body.split("Now solve")[0]


def test_prose_reveal_keeps_filenames_drops_shell():
    body = render("prose_reveal")
    assert "The contents of reasoning_trace.txt are as follows:\nAdd two and two.\n" in body
    assert "The contents of final_answer.txt are as follows:\n4\n" in body
    assert "cat " not in body


def test_extra_hint_inserted_before_target_question():
    body = render("bare_cat", hint="HINT")
    assert body.endswith(f"HINT\n\nQuestion:\n{TARGET_Q}")
