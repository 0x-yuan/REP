"""REP core — the shared, model-agnostic Reasoning Exposure Prompting library.

Every steal-method experiment builds on this package:

* :mod:`rep_core.variants`          — the six wrappers V0-V5 + k-shot assembler
* :mod:`rep_core.prompt_primitives` — the deployed defender system prompt
* :mod:`rep_core.baseline`          — no-trigger Baseline R / Baseline C bodies
* :mod:`rep_core.build_helpers`     — output-format hints + shared tokenizer id
"""
from __future__ import annotations

from .variants import (
    VARIANT_IDS,
    VARIANT_NAMES,
    render_user_content,
)
from .prompt_primitives import (
    CANON_SYSTEM_PROMPT,
    CANON_REASONING_BOUNDARY_PROMPT,
    build_defender_system,
)
from .baseline import baseline_r_instruction, baseline_c_instruction
from .build_helpers import RENDER_MODEL_ID, hint_for, meta_for

__all__ = [
    "VARIANT_IDS",
    "VARIANT_NAMES",
    "render_user_content",
    "CANON_SYSTEM_PROMPT",
    "CANON_REASONING_BOUNDARY_PROMPT",
    "build_defender_system",
    "baseline_r_instruction",
    "baseline_c_instruction",
    "RENDER_MODEL_ID",
    "hint_for",
    "meta_for",
]
