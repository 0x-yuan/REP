"""Self-containment tests: vendored datasets load offline; OpenRouter is the
single external-API path.

These guard the reduced-external-dependency contract: the 500-row test set, the
shadow shot pool, and the small benchmark splits ship inside public/data/, so
prompt building and the scorer coverage gate run with no network.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

PUBLIC = Path(__file__).resolve().parents[1]
DATA = PUBLIC / "data"
STEAL_METHOD = PUBLIC / "steal-method"
sys.path.insert(0, str(STEAL_METHOD))
sys.path.insert(0, str(STEAL_METHOD / "experiments"))


# --------------------------------------------------------------------------- #
# Vendored files exist                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rel", [
    "openthoughts_test_500/ri_qwen3_14b.jsonl.gz",
    "openthoughts_test_500/ri_qwen3_32b.jsonl.gz",
    "qa_seed/reasoning_seed_6k.jsonl.gz",
    "openthoughts_10k/questions.jsonl.gz",
    "shot_pool/qwen3_14b.jsonl.gz",
    "shot_pool/math500_qwen3_14b.jsonl.gz",
    "shot_pool/gsm8k_qwen3_14b.jsonl.gz",
    "shot_pool/jeebench_qwen3_14b.jsonl.gz",
    "benchmarks/math500_test.jsonl.gz",
    "benchmarks/jeebench_test.jsonl.gz",
    "benchmarks/gsm8k_test.jsonl.gz",
])
def test_vendored_file_present(rel):
    assert (DATA / rel).exists(), f"missing vendored dataset: {rel}"


def _count_gz(path: Path) -> int:
    with gzip.open(path, "rt") as f:
        return sum(1 for line in f if line.strip())


def test_vendored_row_counts():
    assert _count_gz(DATA / "openthoughts_test_500/ri_qwen3_14b.jsonl.gz") == 500
    assert _count_gz(DATA / "openthoughts_test_500/ri_qwen3_32b.jsonl.gz") == 500
    assert _count_gz(DATA / "qa_seed/reasoning_seed_6k.jsonl.gz") == 6000
    assert _count_gz(DATA / "openthoughts_10k/questions.jsonl.gz") == 10000
    # shot pools are the resolved seed-7 50-row samples
    assert _count_gz(DATA / "shot_pool/qwen3_14b.jsonl.gz") == 50
    for name in ("math500", "gsm8k", "jeebench"):
        assert _count_gz(DATA / f"shot_pool/{name}_qwen3_14b.jsonl.gz") == 50
    assert _count_gz(DATA / "benchmarks/math500_test.jsonl.gz") == 500
    assert _count_gz(DATA / "benchmarks/jeebench_test.jsonl.gz") == 515
    assert _count_gz(DATA / "benchmarks/gsm8k_test.jsonl.gz") == 1319


# --------------------------------------------------------------------------- #
# Loaders read vendored data (no network needed)                              #
# --------------------------------------------------------------------------- #

def test_load_ot500_offline_has_question_and_r0():
    import _common as C
    rows = C.load_ot500_test()
    assert len(rows) == 500
    assert rows[0]["question"]
    assert rows[0]["ri"]           # benign internal trace r0 (for R02 scoring)


def test_load_ot500_32b_offline():
    import _common as C
    rows = C.load_ot500_test("ri_qwen3_32b")
    assert len(rows) == 500 and rows[0]["question"] and rows[0]["ri"]


def test_qa_seed_and_10k_questions_offline():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_eval_sets", PUBLIC / "eval" / "qa_eval" / "build_eval_sets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    seed = m.load_seed()
    assert len(seed) == 6000
    assert {r["source"] for r in seed} == {"strategyqa", "prontoqa", "hotpotqa"}
    assert set(seed[0]) >= {"id", "source", "question", "answer", "answer_type", "meta_json"}

    spec = importlib.util.spec_from_file_location(
        "distill_build_prompts",
        STEAL_METHOD / "experiments" / "distill_corpus" / "build_prompts.py")
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    qs = bp.load_questions(str(bp.QUESTIONS_DEFAULT))
    assert len(qs) == 10000
    assert set(qs[0]) == {"prompt_id", "source_index", "question"}


def test_load_shot_pool_offline():
    import _common as C
    demos = C.load_ot_shot_pool(C.SHOT_POOL_LOCAL)
    assert len(demos) == 50
    assert set(demos[0]) >= {"q", "r", "a"}


@pytest.mark.parametrize("source", ["openthoughts", "math500", "gsm8k", "jeebench"])
def test_cross_dataset_pools_load_offline_in_stored_order(source):
    """The off-domain pools are pre-sampled; the builder must slice, not resample."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location(
        "cross_dataset_build_prompts",
        STEAL_METHOD / "experiments" / "cross_dataset" / "build_prompts.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    demos = cd.load_shots(source)
    assert len(demos) == 50
    assert all(d["q"] and d["r"] and d["a"] for d in demos)
    if source != "openthoughts":
        with gzip.open(cd.CROSS_SHOT_PATHS[source], "rt") as f:
            first = [json.loads(next(f)) for _ in range(3)]
        assert [d["q"] for d in demos[:3]] == [r["q"] for r in first]
        assert [d["src_idx"] for d in demos[:3]] == [r["src_idx"] for r in first]
    # the four pools are distinct demo sets
    assert demos[0]["q"] != cd.load_shots("openthoughts" if source != "openthoughts" else "gsm8k")[0]["q"]


# --------------------------------------------------------------------------- #
# OpenRouter is the single external-API path                                   #
# --------------------------------------------------------------------------- #

def test_openrouter_runner_importable_and_schema():
    pytest.importorskip("aiohttp")
    import importlib.util
    p = STEAL_METHOD / "experiments" / "cross_victim" / "openrouter_runner.py"
    spec = importlib.util.spec_from_file_location("openrouter_runner", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # points at OpenRouter, not any other provider
    assert m.URL == "https://openrouter.ai/api/v1/chat/completions"
    # error result mirrors the farm's get_results schema
    err = m._err_result({"id": "x|cell=y"}, "boom")
    assert err["id"] == "x|cell=y" and err["error"] == "boom"
    assert err["outputs"][0].keys() >= {"text", "reasoning", "finish_reason",
                                        "completion_tokens", "reasoning_tokens"}


def test_openrouter_key_required(monkeypatch):
    pytest.importorskip("aiohttp")
    import importlib.util
    p = STEAL_METHOD / "experiments" / "cross_victim" / "openrouter_runner.py"
    spec = importlib.util.spec_from_file_location("openrouter_runner2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(m, "ENV_PATH", Path("/nonexistent/.env"))
    with pytest.raises(RuntimeError):
        m.load_api_key()
