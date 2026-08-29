"""Per-benchmark output-format hints + tokenizer id shared by every builder.

The Qwen3 family shares one chat template, so prompts are rendered once with the
8B tokenizer (``RENDER_MODEL_ID``). ``hint_for`` supplies the output-format
instruction injected before the final target question (boxed answer for math;
type-aware instruction for JEEBench, matching the official protocol).
"""
from __future__ import annotations

# Qwen3 family shares one chat template — render once with the 8B tokenizer.
RENDER_MODEL_ID = "Qwen/Qwen3-8B"

MATH500_TYPE_HINT = "Output format: write the final answer wrapped in \\boxed{}."

JEE_TYPE_INSTRUCTIONS: dict[str, str] = {
    "MCQ": (
        "Question type: multiple-choice with exactly one correct option.\n"
        "Output format: \\boxed{X} where X is one letter from {A, B, C, D}."
    ),
    "MCQ(multiple)": (
        "Question type: multiple-choice; one or more options can be correct.\n"
        "Output format: \\boxed{XYZ} listing every correct letter from {A, B, C, D}."
    ),
    "Integer": (
        "Question type: the final answer is a non-negative integer.\n"
        "Output format: \\boxed{N} with N a non-negative integer."
    ),
    "Numeric": (
        "Question type: the final answer is a decimal number; give it correct\n"
        "to the second decimal digit.\n"
        "Output format: \\boxed{N.NN}."
    ),
}


def hint_for(bench: str, meta: dict | None) -> str:
    if bench in ("math500", "openthoughts500"):
        return MATH500_TYPE_HINT
    if bench == "jeebench":
        qtype = (meta or {}).get("type", "MCQ")
        return JEE_TYPE_INSTRUCTIONS.get(qtype, JEE_TYPE_INSTRUCTIONS["MCQ"])
    raise ValueError(bench)


def meta_for(bench: str, row: dict) -> dict | None:
    if bench in ("jeebench", "openthoughts500"):
        m = row.get("meta")
        return m if isinstance(m, dict) else None
    return None


# Backwards-compatible aliases (the ported experiment builders import these).
_hint_for = hint_for
_meta_for = meta_for
