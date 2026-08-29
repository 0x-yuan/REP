"""Baseline C (simple CoT, paper Table 1 row 2 / Appendix "Baseline Trigger
Prompts") as emitted by the config-sweep builder's ``--include-baseline-c``.

Hermetic checks encode the appendix wording; when the original CoT-baseline
experiment folder is present the rendered user body is compared byte-for-byte
with the instruction it recorded in its build manifest, on its own test rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PUBLIC = Path(__file__).resolve().parents[1]
STEAL_METHOD = PUBLIC / "steal-method"
sys.path.insert(0, str(STEAL_METHOD))
sys.path.insert(0, str(STEAL_METHOD / "experiments"))

import _common as C  # noqa: E402
from rep_core.baseline import baseline_c_instruction, baseline_r_instruction  # noqa: E402


def _load(name: str, path: Path):
    """Every experiment folder has a `build_prompts.py`; load by path, not by name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sweep = _load("config_sweep_build_prompts",
              STEAL_METHOD / "experiments" / "config_sweep" / "build_prompts.py")

ROW = {"idx": "x", "question": "  What is 7+7?  ", "meta": {}}


def test_baseline_c_wording():
    body = sweep.baseline_c_body(ROW)
    assert body.startswith("Solve the following math problem.\nReturn exactly one <think>...</think> block.\n")
    assert "let's think step by step" in body
    assert "Output format: write the final answer wrapped in \\boxed{}." in body
    assert body.endswith("\n\nQuestion:\nWhat is 7+7?")
    assert "repeat the reasoning" not in body


def test_baseline_c_differs_from_baseline_r_only_in_reveal_clause():
    c = baseline_c_instruction("openthoughts500").splitlines()
    r = baseline_r_instruction("openthoughts500").splitlines()
    assert len(c) == len(r)
    diff = [i for i, (a, b) in enumerate(zip(c, r)) if a != b]
    assert diff == [2, 3]          # the two post-</think> lines


def test_cli_flag_and_cell_naming():
    args = sweep.parse_args(["--include-baseline-c"])
    assert args.include_baseline_c
    assert sweep.cell_id(sweep.BASELINE_C, None) == "CoT-14b"
    assert sweep.batch_name(sweep.BASELINE_C, None) == "cot_baseline"
    assert sweep.cell_id(None, None) == "NoTrigger-14b"
    assert sweep.cell_id("V3", 3) == "OT-V3-K3"
