"""Modal resource naming with optional EXP_ID prefix.

The same code base can be `cp -r`'d to multiple experiment folders
(`exp-01/`, `exp-02/`, ...) and each copy will deploy to its own
isolated namespace of Modal apps / volumes / dicts as long as
`EXP_ID` is set differently in each copy.

# Naming policy

When `EXP_ID` is unset / empty, names are unprefixed (legacy mode):
    sglang-slave-qwen3-8b
    sglang-slave-qwen3-8b-r0
    inference-farm-data
    sglang-hf-cache
    sglang-slave-checkpoints
    sglang-slave-qwen3-8b-progress
    sglang-slave-qwen3-8b-fc

When `EXP_ID="exp-01"`, every per-experiment resource gets prefixed:
    exp-01-sglang-slave-qwen3-8b
    exp-01-sglang-slave-qwen3-8b-r0
    exp-01-data
    exp-01-checkpoints
    exp-01-sglang-slave-qwen3-8b-progress
    exp-01-sglang-slave-qwen3-8b-fc

The HF cache volume (`sglang-hf-cache`) stays SHARED in both modes:
model weights are immutable, so cross-experiment reuse only races on
the very first download and is otherwise pure win (each model takes
~30 GB on disk).

# Where EXP_ID comes from

Resolved from the `EXP_ID` env var. The convention is to set it once
in `<server-root>/experiment.env` and source that file from
`master/run_pipeline.sh` before any Modal command runs. The slave's
container image bakes the value in at build time so deployed
containers see the same EXP_ID their app was created under.

Validation: `[a-zA-Z0-9._-]{0,40}`. Empty is the legacy escape hatch.
"""
from __future__ import annotations

import os
import re

_LEGACY_DATA_VOLUME = "inference-farm-data"
_LEGACY_CHECKPOINT_VOLUME = "sglang-slave-checkpoints"
_SHARED_HF_CACHE = "sglang-hf-cache"

_VALID_EXP_ID = re.compile(r"^[a-zA-Z0-9._-]{0,40}$")


def resolve_exp_id() -> str:
    """Read and validate the `EXP_ID` env var. Returns "" when unset."""
    raw = os.environ.get("EXP_ID", "").strip()
    if not raw:
        return ""
    if not _VALID_EXP_ID.match(raw):
        raise ValueError(
            f"EXP_ID must match {_VALID_EXP_ID.pattern!r} (alnum + . _ -, "
            f"≤ 40 chars); got {raw!r}"
        )
    return raw


def _prefixed(base: str, exp_id: str) -> str:
    return f"{exp_id}-{base}" if exp_id else base


def app_name(model: str, replica_id: str = "", exp_id: str | None = None) -> str:
    """Modal app name for a single slave deployment."""
    if exp_id is None:
        exp_id = resolve_exp_id()
    suffix = f"-{replica_id}" if replica_id else ""
    return _prefixed(f"sglang-slave-{model}{suffix}", exp_id)


def progress_dict_name(
    model: str, replica_id: str = "", exp_id: str | None = None
) -> str:
    return f"{app_name(model, replica_id, exp_id)}-progress"


def fc_dict_name(
    model: str, replica_id: str = "", exp_id: str | None = None
) -> str:
    return f"{app_name(model, replica_id, exp_id)}-fc"


def data_volume_name(exp_id: str | None = None) -> str:
    """Modal Volume holding inbox JSONL uploads. Per-experiment when EXP_ID is set."""
    if exp_id is None:
        exp_id = resolve_exp_id()
    return f"{exp_id}-data" if exp_id else _LEGACY_DATA_VOLUME


def checkpoint_volume_name(exp_id: str | None = None) -> str:
    """Modal Volume holding per-batch checkpoints. Per-experiment when EXP_ID is set."""
    if exp_id is None:
        exp_id = resolve_exp_id()
    return f"{exp_id}-checkpoints" if exp_id else _LEGACY_CHECKPOINT_VOLUME


def hf_cache_volume_name(exp_id: str | None = None) -> str:
    """Modal Volume holding HF model weights. ALWAYS shared across experiments."""
    return _SHARED_HF_CACHE


def prefetch_app_name(model: str, exp_id: str | None = None) -> str:
    if exp_id is None:
        exp_id = resolve_exp_id()
    return _prefixed(f"sglang-prefetch-{model}", exp_id)
