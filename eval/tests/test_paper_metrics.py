"""Hermetic unit tests for the full-trace metric panel (paper Tables 3/4).

Synthetic rows only: checks the row filter, the three pair scores, LEN,
struct%, the CSV layout, and the raw-generation adapter (r1/r2 extraction +
r0 join against a baseline outbox).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from trace_metrics import paper_metrics as pm  # noqa: E402


ROWS = [
    # perfect leak: r2 == r0, r1 shares half of r0
    {"id": "a", "r0": "alpha beta gamma delta", "r1": "alpha beta", "r2": "alpha beta gamma delta",
     "structural_success": 1.0},
    # dropped: no r2 (non-structural row) — still counts in the struct denominator
    {"id": "b", "r0": "alpha beta gamma delta", "r1": "alpha beta gamma delta", "r2": "",
     "structural_success": 0.0},
]


def test_compute_cell_filters_rows_and_scores_pairs():
    n_total, n_struct = pm.struct_counts(ROWS)
    assert (n_total, n_struct) == (2, 1)
    r = pm.compute_cell(ROWS, "toy", n_total=n_total, n_struct=n_struct)
    assert r["n_rows"] == 1                      # row b dropped
    assert r["struct_pct"] == pytest.approx(50.0)
    p = r["pairs"]
    assert set(p) == {"r0r1", "r0r2", "r1r2"}
    assert p["r0r2"]["rougeL_f1"] == pytest.approx(1.0)
    assert p["r0r2"]["rouge1_f1"] == pytest.approx(1.0)
    assert p["r0r2"]["rouge2_f1"] == pytest.approx(1.0)
    # r0 (4 tokens) vs r1 (2 tokens): P=1, R=0.5 -> F1 = 2/3
    assert p["r0r1"]["rougeL_f1"] == pytest.approx(2 / 3)
    assert p["r1r2"]["rougeL_f1"] == pytest.approx(2 / 3)
    assert p["r0r2"]["n"] == 1
    assert r["len_tok"] == {"r0": 4.0, "r1": 2.0, "r2": 4.0}


def test_compute_cell_returns_none_when_nothing_scorable():
    assert pm.compute_cell([ROWS[1]], "empty") is None


def test_trace_alias_ri_becomes_r0():
    t = pm.trace_rows_from_traces([{"prompt_id": "q1|cell=X", "ri": "R", "r1": "A", "r2": "B"}])
    assert t == [{"id": "q1", "r0": "R", "r1": "A", "r2": "B", "structural_success": None}]


def test_generation_adapter_extracts_full_traces_and_joins_r0():
    long_r2 = " ".join(["tok"] * 4000)          # > the 3000-char per-row cap of score.py
    attack = [{"id": "q1|cell=OT-V3-K3", "meta_test_idx": "q1",
               "generation": f"<think>\nthink one\n</think>\n{long_r2}\n\\boxed{{1}}"}]
    baseline = [{"id": "q1|cell=NoTrigger", "outputs": [{"text": "<think>\nbenign trace\n</think>\n\\boxed{1}"}]}]
    ref = pm.reference_map_from_generations(baseline, "qwen3-14b")
    assert ref == {"q1": "benign trace"}
    t = pm.trace_rows_from_generations(attack, "qwen3-14b", ref)
    assert t[0]["id"] == "q1" and t[0]["r0"] == "benign trace" and t[0]["r1"] == "think one"
    assert t[0]["r2"].startswith(long_r2) and len(t[0]["r2"]) > 3000   # untruncated
    assert t[0]["structural_success"] is True


def test_cli_writes_csv_with_paper_columns(tmp_path):
    inp = tmp_path / "traces.jsonl"
    inp.write_text("\n".join(__import__("json").dumps(r) for r in ROWS) + "\n")
    out = tmp_path / "pm.csv"
    tex = tmp_path / "pm.tex"
    assert pm.main([str(inp), "--cell", "toy", "--out-csv", str(out), "--latex", str(tex)]) == 0
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == pm.CSV_COLUMNS
    assert [r["pair"] for r in rows] == ["r0r1", "r0r2", "r1r2"]
    r0r2 = {r["pair"]: r for r in rows}["r0r2"]
    assert r0r2["cell"] == "toy" and r0r2["n_rows"] == "1" and r0r2["rougeL_f1"] == "1.000"
    assert r0r2["len_tok_r2"] == "4"
    assert "toy & 50.0 & 1.000" in tex.read_text()


def test_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        pm.main(["--help"])
    assert e.value.code == 0
