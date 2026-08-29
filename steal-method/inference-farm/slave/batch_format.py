"""Canonical batch file schema + (de)serialization.

The slave ingests **JSON Lines** files. Each line is one independent
inference request. External users are expected to write files in
exactly this format.

# Required per row
    id        : str  — stable identifier; output rows reuse it.
    prompt    : str  — raw text passed verbatim to vLLM (no chat template), OR
    messages  : list[ {"role": "system"|"user"|"assistant", "content": str} ]
                       — chat-template will be applied with the slave's tokenizer.
    Exactly one of `prompt` / `messages` must be present.

# Optional sampling overrides (per row, fall back to slave defaults)
    max_tokens             : int           default 1024
    temperature            : float         default 0.0
    top_p                  : float         default 1.0
    top_k                  : int           default null (disabled)
    stop                   : list[str]     default null
    n                      : int           default 1   (number of samples)
    seed                   : int           default null
    repetition_penalty     : float         default 1.0

# Optional chat-template overrides (only consulted if `messages` is set)
    enable_thinking            : bool      default true   (Qwen3 thinking mode)
    add_generation_prompt      : bool      default true
    continue_final_message     : bool      default false  (prefill attacks)

# Output row schema (written to --output JSONL by the slave)
    id                : str   — same id as input
    model             : str   — model_key that produced this row (e.g. "qwen3-8b")
    prompt_tokens     : int
    outputs           : list[ {"text": str,
                               "finish_reason": "stop"|"length"|...,
                               "completion_tokens": int} ]
    error             : str   — present only if the row failed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchRow:
    id: str
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None

    # None → fall back to slave's per-model default_max_tokens (≥8192 floor).
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: list[str] | None = None
    n: int = 1
    seed: int | None = None
    repetition_penalty: float = 1.0

    enable_thinking: bool = True
    add_generation_prompt: bool = True
    continue_final_message: bool = False

    raw: dict[str, Any] = field(default_factory=dict)


def parse_row(d: dict[str, Any]) -> BatchRow:
    if "id" not in d:
        raise ValueError(f"row missing 'id': {d!r}")
    has_prompt = "prompt" in d and d["prompt"] is not None
    has_messages = "messages" in d and d["messages"] is not None
    if has_prompt == has_messages:
        raise ValueError(
            f"row {d['id']!r}: exactly one of 'prompt' or 'messages' must be set "
            f"(got prompt={has_prompt}, messages={has_messages})"
        )
    if has_messages:
        msgs = d["messages"]
        if not isinstance(msgs, list) or not all(
            isinstance(m, dict) and "role" in m and "content" in m for m in msgs
        ):
            raise ValueError(
                f"row {d['id']!r}: 'messages' must be list of "
                "{role, content} dicts"
            )

    return BatchRow(
        id=str(d["id"]),
        prompt=d.get("prompt"),
        messages=d.get("messages"),
        max_tokens=(int(d["max_tokens"]) if d.get("max_tokens") is not None else None),
        temperature=float(d.get("temperature", 0.0)),
        top_p=float(d.get("top_p", 1.0)),
        top_k=d.get("top_k"),
        stop=d.get("stop"),
        n=int(d.get("n", 1)),
        seed=d.get("seed"),
        repetition_penalty=float(d.get("repetition_penalty", 1.0)),
        enable_thinking=bool(d.get("enable_thinking", True)),
        add_generation_prompt=bool(d.get("add_generation_prompt", True)),
        continue_final_message=bool(d.get("continue_final_message", False)),
        raw=d,
    )


def make_output_row(
    row_id: str,
    prompt_tokens: int,
    outputs: list[dict[str, Any]],
    model: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": row_id,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "outputs": outputs,
    }
    if error is not None:
        rec["error"] = error
    return rec


def make_error_row(row_id: str, error: str, model: str | None = None) -> dict[str, Any]:
    return {
        "id": row_id,
        "model": model,
        "outputs": [],
        "prompt_tokens": 0,
        "error": error,
    }
