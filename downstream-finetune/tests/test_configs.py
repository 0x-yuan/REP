"""Schema tests for every per-row config in ../configs/.

Real training needs Modal + B200/H200, so these are static/offline checks:
each config loads as a plain module and carries the fields the canonical
engine scripts read, with sane values.

    cd public && uv run --with pytest python -m pytest downstream-finetune/tests -q
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Required, always-non-empty fields the engine scripts dereference.
REQUIRED_STR = [
    "PROFILE", "RUN_ID", "MODEL_FAMILY", "MODEL_NAME",
    "SOURCE_REPO", "DATASET_SUBDIR",
    "CKPTS_VOL", "RESULTS_VOL", "HF_CACHE_VOL", "TRAIN_APP", "EVAL_APP",
    "SAVE_STRATEGY", "GPU_TYPE", "ENGINE",
]
REQUIRED_INT = ["EPOCHS", "BLOCK_SIZE", "GRAD_ACCUM", "MICRO_BATCH", "GPU_COUNT"]

HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
RECONSTRUCTED = {"oracle_14b"}
VALID_ENGINES = {"engine"}


def _config_paths() -> list[Path]:
    return sorted(p for p in CONFIGS_DIR.glob("*.py") if not p.name.startswith("_"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"cfg_{path.stem}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_count():
    assert len(_config_paths()) == 12, [p.stem for p in _config_paths()]


def test_configs_exist():
    names = {p.stem for p in _config_paths()}
    expected = {
        "rep_14b_clean", "rep_14b_orig", "rep_32b_clean", "rep_32b_orig",
        "oracle_14b", "oracle_32b", "answer_only_14b", "answer_only_32b",
        "summary_14b", "summary_32b",
        # supplementary rows (14B / 32B students, FSDP)
        "rep_q3_14b_to_qwen25_14b", "rep_q3_14b_to_qwen25_32b",
    }
    assert expected <= names, f"missing configs: {expected - names}"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_required_fields(path: Path):
    cfg = _load(path)

    for field in REQUIRED_STR:
        val = getattr(cfg, field, None)
        assert isinstance(val, str) and val.strip(), f"{path.stem}: {field} empty/missing"

    for field in REQUIRED_INT:
        val = getattr(cfg, field, None)
        assert isinstance(val, int) and val > 0, f"{path.stem}: {field} must be positive int"

    # RUN_ID is the run identity; HF_REPO is optional ("" => skip upload).
    assert getattr(cfg, "RUN_ID", "").strip(), f"{path.stem}: RUN_ID empty"
    hf = getattr(cfg, "HF_REPO", None)
    assert isinstance(hf, str), f"{path.stem}: HF_REPO must be a str ('' to skip)"
    if hf:
        assert HF_ID.match(hf), f"{path.stem}: HF_REPO '{hf}' is not a HF repo id"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_source_repo_is_hf_id(path: Path):
    cfg = _load(path)
    assert HF_ID.match(cfg.SOURCE_REPO), f"{path.stem}: SOURCE_REPO '{cfg.SOURCE_REPO}' bad"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_eval_pool_length(path: Path):
    cfg = _load(path)
    suffixes = getattr(cfg, "EVAL_APP_SUFFIXES", None)
    assert suffixes is not None, f"{path.stem}: EVAL_APP_SUFFIXES missing"
    assert len(suffixes) == cfg.EPOCHS + 1, (
        f"{path.stem}: len(EVAL_APP_SUFFIXES)={len(suffixes)} != EPOCHS+1={cfg.EPOCHS + 1}"
    )


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_engine_marker(path: Path):
    cfg = _load(path)
    assert cfg.ENGINE in VALID_ENGINES, f"{path.stem}: ENGINE '{cfg.ENGINE}' invalid"
    assert cfg.ENGINE == "engine", f"{path.stem}: row must use the distillation engine"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_profile_scrubbed(path: Path):
    cfg = _load(path)
    assert cfg.PROFILE == "<your-modal-profile>", (
        f"{path.stem}: PROFILE '{cfg.PROFILE}' not scrubbed"
    )


@pytest.mark.parametrize("path", sorted(RECONSTRUCTED))
def test_reconstructed_marker(path: str):
    text = (CONFIGS_DIR / f"{path}.py").read_text()
    assert "RECONSTRUCTED" in text, f"{path}: missing RECONSTRUCTED marker comment"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_epochs(path: Path):
    """REP + oracle rows ran 5 epochs; the answer-only / summary controls ran 6."""
    cfg = _load(path)
    if path.stem.startswith(("rep_", "oracle_")):
        assert cfg.EPOCHS == 5, f"{path.stem}: must be EPOCHS=5"
    else:
        assert cfg.EPOCHS == 6, f"{path.stem}: control rows must be EPOCHS=6"


HUB_CONFIGS = {
    "rep_14b_clean": "distill_q3_14b_clean", "rep_14b_orig": "distill_q3_14b_original",
    "rep_32b_clean": "distill_q3_32b_clean", "rep_32b_orig": "distill_q3_32b_original",
}


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_source_config(path: Path):
    """The 4 REP rows read a config of `Chia-Mu-Lab/REP-datasets`; other rows have none."""
    cfg = _load(path)
    conf = getattr(cfg, "SOURCE_CONFIG", None)
    if path.stem in HUB_CONFIGS:
        assert cfg.SOURCE_REPO == "Chia-Mu-Lab/REP-datasets", path.stem
        assert conf == HUB_CONFIGS[path.stem], f"{path.stem}: SOURCE_CONFIG={conf!r}"
    else:
        assert conf is None, f"{path.stem}: unexpected SOURCE_CONFIG={conf!r}"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_no_hf_upload_target(path: Path):
    cfg = _load(path)
    assert cfg.HF_REPO == "", f"{path.stem}: HF_REPO must be '' in the release"


# Student size -> required sharding knobs. 7B fits plain DDP on 4 B200; the
# 14B/32B students need FSDP full_shard on 8 B200 (see README "Student scale").
SHARDING_BY_STUDENT = {
    "Qwen/Qwen2.5-7B-Instruct": dict(USE_FSDP=False, GPU_COUNT=4, GRAD_ACCUM=4),
    "Qwen/Qwen2.5-14B-Instruct": dict(USE_FSDP=True, GPU_COUNT=8, GRAD_ACCUM=2),
    "Qwen/Qwen2.5-32B-Instruct": dict(USE_FSDP=True, GPU_COUNT=8, GRAD_ACCUM=2),
}


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_sharding_fields(path: Path):
    cfg = _load(path)
    assert isinstance(cfg.USE_FSDP, bool), f"{path.stem}: USE_FSDP must be a bool"
    assert isinstance(cfg.GPU_COUNT, int) and cfg.GPU_COUNT in (1, 2, 4, 8), (
        f"{path.stem}: GPU_COUNT={cfg.GPU_COUNT} not a valid single-node count"
    )
    assert cfg.MODEL_NAME in SHARDING_BY_STUDENT, f"{path.stem}: unknown student {cfg.MODEL_NAME}"
    for k, v in SHARDING_BY_STUDENT[cfg.MODEL_NAME].items():
        assert getattr(cfg, k) == v, f"{path.stem}: {k}={getattr(cfg, k)!r}, expected {v!r} for {cfg.MODEL_NAME}"
    # Effective batch is frozen at 16 across all student sizes.
    assert cfg.MICRO_BATCH * cfg.GRAD_ACCUM * cfg.GPU_COUNT == 16, f"{path.stem}: eff-batch != 16"
    assert cfg.MODEL_FAMILY == "qwen25"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_unique_namespaces(path: Path):
    """RUN_ID / volumes / apps must not collide across rows."""
    cfg = _load(path)
    others = [_load(p) for p in _config_paths() if p != path]
    for field in ("RUN_ID", "CKPTS_VOL", "RESULTS_VOL", "TRAIN_APP", "EVAL_APP", "DATASET_SUBDIR"):
        mine = getattr(cfg, field)
        clashes = [o.RUN_ID for o in others if getattr(o, field) == mine]
        assert not clashes, f"{path.stem}: {field}={mine!r} also used by {clashes}"
