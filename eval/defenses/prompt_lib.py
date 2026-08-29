"""Shared helpers for injecting black-box defenses into pre-rendered Qwen3
chat-template prompts (the REP inbox rows written by ``steal-method``).

Each prompt row carries a fully rendered string of the form

  <|im_start|>system\\n{SYS}<|im_end|>\\n
  <|im_start|>user\\n{USER}<|im_end|>\\n
  <|im_start|>assistant\\n

We parse out SYS and USER, let a defense transform them, re-render, and
recompute ``max_tokens = TOTAL_CTX - prompt_tokens - HEADROOM`` (the paper's
DeepInfra Qwen3 context budget: 40960 tokens, no YaRN).

Row id: ``idx`` if present, else ``meta_test_idx``, else ``id`` up to the first
``|`` (the steal-method inbox id is ``"<idx>|cell=<cell>"``).

Benign controls (used by the gate scripts):
  * ``benign``      = the bare target question (short legitimate prompt);
  * ``benign_long`` = the ~30K worked-examples block only, i.e. everything before
    the REP "Now solve ..." instruction (long legitimate few-shot prompt).
"""
from __future__ import annotations

import re
from functools import lru_cache

TOTAL_CTX = 40960          # DeepInfra Qwen3-14B/32B max_total_tokens (no YaRN)
HEADROOM = 256             # build-time headroom (matches the original builder)
MIN_OUT = 512              # never leave less than this output budget
TOKENIZER_ID = "Qwen/Qwen3-8B"   # Qwen3 dense family shares one tokenizer

_PAT = re.compile(
    r"<\|im_start\|>system\n(?P<sys>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n?$",
    re.DOTALL,
)
_CUT_MARKERS = ("Now solve", "think in <think>")


def parse(prompt: str) -> tuple[str, str]:
    """Return (system_text, user_text). Raises if the template is unexpected."""
    m = _PAT.match(prompt)
    if not m:
        raise ValueError("prompt does not match expected system/user/assistant template")
    return m.group("sys"), m.group("user")


def render(system_text: str, user_text: str) -> str:
    """Re-render a Qwen3 chat prompt with add_generation_prompt=True."""
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def row_idx(row: dict) -> str:
    for k in ("idx", "meta_test_idx"):
        if row.get(k) is not None:
            return str(row[k])
    return str(row["id"]).split("|")[0]


def benign_text(user_text: str, mode: str) -> str:
    """Slice of the REP user turn used as a control. ``rep`` returns it whole."""
    if mode == "benign":
        return user_text.split("Question:")[-1].strip()
    if mode == "benign_long":
        cut = -1
        for marker in _CUT_MARKERS:
            cut = user_text.find(marker)
            if cut >= 0:
                break
        return user_text[:cut].strip() if cut > 0 else user_text
    return user_text


@lru_cache(maxsize=1)
def _tok(model_id: str = TOKENIZER_ID):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def count_tokens(text: str, model_id: str = TOKENIZER_ID) -> int:
    return len(_tok(model_id)(text)["input_ids"])


def budget_for(ptok: int) -> int:
    """max_tokens so that prompt_tokens + max_tokens + HEADROOM <= TOTAL_CTX."""
    return max(MIN_OUT, TOTAL_CTX - ptok - HEADROOM)


def make_row(idx: str, system_text: str, user_text: str) -> dict:
    prompt = render(system_text, user_text)
    ptok = count_tokens(prompt)
    return {"idx": idx, "prompt": prompt, "ptok": ptok, "max_tokens": budget_for(ptok)}
