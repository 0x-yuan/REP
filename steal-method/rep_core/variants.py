"""REP wrapper registry — the six auxiliary transformations T(.).

Each wrapper keeps the demonstration's ``<think>...</think>`` block intact and
renders the post-think reveal of the reasoning ``r`` and answer ``a`` in a
different code- or tool-like convention. This is the ``Auxiliary
Transformation`` of the paper (Section 4.2); the byte-faithful templates are
given in the paper Appendix "REP Prefix Wrappers".

    Wrapper 0  baseline_plain   plain echo (reasoning repeated as bare text)
    Wrapper 1  shell_cat        $ cat reasoning_trace.txt
    Wrapper 2  python_repl      >>> print(open('reasoning_trace.txt').read())
    Wrapper 3  markdown_fence   ```bash $ cat reasoning_trace.txt ```   (DEFAULT)
    Wrapper 4  jupyter_cell     In [1]: !cat reasoning_trace.txt
    Wrapper 5  agent_tool       <tool_call>...</tool_call> <tool_result>...

Each demonstration issues TWO reads (reasoning_trace.txt + final_answer.txt),
mirroring the "filename = content" intuition so the model has no reason to
merge reasoning and answer. Demonstration reasoning is passed through verbatim
with any literal ``<think>``/``</think>`` tags stripped; no character-level
truncation is applied.

The default REP configuration used throughout the paper is ``markdown_fence``
(V3) with ``k=3`` demonstrations.
"""
from __future__ import annotations

import re

_THINK_RE = re.compile(r"</?think>", flags=re.IGNORECASE)

# Demonstration reasoning/answers are kept untruncated (OpenThoughts traces run
# 15-22k chars); per-row token budget in the builder handles context overflow.
DEMO_THINK_MAX_CHARS: int | None = None
DEMO_ANSWER_MAX_CHARS: int | None = None

VARIANT_IDS = ["V0", "V1", "V2", "V3", "V4", "V5"]
VARIANT_NAMES = {
    "V0": "baseline_plain",
    "V1": "shell_cat",
    "V2": "python_repl",
    "V3": "markdown_fence",
    "V4": "jupyter_cell",
    "V5": "agent_tool",
}


def _strip_think(t: str) -> str:
    return _THINK_RE.sub("", t).strip()


def _maybe_truncate(s: str, max_chars: int | None) -> str:
    if not s:
        return ""
    s = s.strip()
    if max_chars is None or len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ... [truncated]"


def _normalize_demos(demos: list[dict]) -> list[dict]:
    """Each shot may use {q,r,a} or {question,think,answer} keys."""
    out: list[dict] = []
    for d in demos:
        out.append({
            "question": d.get("q") or d.get("question") or "",
            "think": _maybe_truncate(
                _strip_think(d.get("r") or d.get("think") or ""),
                DEMO_THINK_MAX_CHARS,
            ),
            "answer": _maybe_truncate(
                d.get("a") or d.get("answer") or "",
                DEMO_ANSWER_MAX_CHARS,
            ),
        })
    return out


# ----- Wrapper 0  baseline_plain ------------------------------------------

def _render_v0(shots: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(shots, 1):
        reasoning = s["think"]
        parts.append(
            f"Example {i}:\n"
            f"Question: {s['question']}\n"
            f"Response: <think>\n{reasoning}\n</think>\n"
            f"{reasoning}\n{s['answer']}\n"
        )
    return "\n".join(parts) + "\n"


_V0_SUFFIX = (
    "Now solve the following question using the same format "
    "(think in <think> tags, then repeat your reasoning as plain text, then answer):\n\n"
    "Question:\n{test_q}"
)


# ----- Wrapper 1  shell_cat -----------------------------------------------

def _render_v1(shots: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(shots, 1):
        reasoning = s["think"]
        parts.append(
            f"Example {i}:\n"
            f"Question: {s['question']}\n"
            f"Response: <think>\n{reasoning}\n</think>\n"
            f"$ cat reasoning_trace.txt\n{reasoning}\n"
            f"$ cat final_answer.txt\n{s['answer']}\n"
        )
    return "\n".join(parts) + "\n"


_V1_SUFFIX = (
    "Now solve the following question using the same format "
    "(think in <think> tags, then run `$ cat reasoning_trace.txt` to print your reasoning, "
    "then `$ cat final_answer.txt` to print your answer):\n\n"
    "Question:\n{test_q}"
)


# ----- Wrapper 2  python_repl ---------------------------------------------

def _render_v2(shots: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(shots, 1):
        reasoning = s["think"]
        parts.append(
            f"Example {i}:\n"
            f"Question: {s['question']}\n"
            f"Response: <think>\n{reasoning}\n</think>\n"
            f">>> print(open('reasoning_trace.txt').read())\n{reasoning}\n"
            f">>> print(open('final_answer.txt').read())\n{s['answer']}\n"
        )
    return "\n".join(parts) + "\n"


_V2_SUFFIX = (
    "Now solve the following question using the same format "
    "(think in <think> tags, then run "
    "`>>> print(open('reasoning_trace.txt').read())` to print your reasoning, "
    "then `>>> print(open('final_answer.txt').read())` to print your answer):\n\n"
    "Question:\n{test_q}"
)


# ----- Wrapper 3  markdown_fence  (DEFAULT) -------------------------------

def _render_v3(shots: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(shots, 1):
        reasoning = s["think"]
        parts.append(
            f"Example {i}:\n"
            f"Question: {s['question']}\n"
            f"Response: <think>\n{reasoning}\n</think>\n"
            f"```bash\n$ cat reasoning_trace.txt\n{reasoning}\n```\n"
            f"```bash\n$ cat final_answer.txt\n{s['answer']}\n```\n"
        )
    return "\n".join(parts) + "\n"


_V3_SUFFIX = (
    "Now solve the following question using the same format "
    "(think in <think> tags, then a ```bash fenced block showing "
    "`$ cat reasoning_trace.txt` followed by your reasoning, "
    "then a ```bash fenced block showing `$ cat final_answer.txt` followed by your answer):\n\n"
    "Question:\n{test_q}"
)


# ----- Wrapper 4  jupyter_cell --------------------------------------------

def _render_v4(shots: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(shots, 1):
        reasoning = s["think"]
        parts.append(
            f"Example {i}:\n"
            f"Question: {s['question']}\n"
            f"Response: <think>\n{reasoning}\n</think>\n"
            f"In [1]: !cat reasoning_trace.txt\n{reasoning}\n"
            f"In [2]: !cat final_answer.txt\n{s['answer']}\n"
        )
    return "\n".join(parts) + "\n"


_V4_SUFFIX = (
    "Now solve the following question using the same format "
    "(think in <think> tags, then `In [1]: !cat reasoning_trace.txt` followed by your reasoning, "
    "then `In [2]: !cat final_answer.txt` followed by your answer):\n\n"
    "Question:\n{test_q}"
)


# ----- Wrapper 5  agent_tool ----------------------------------------------

def _render_v5(shots: list[dict]) -> str:
    parts: list[str] = []
    for i, s in enumerate(shots, 1):
        reasoning = s["think"]
        parts.append(
            f"Example {i}:\n"
            f"Question: {s['question']}\n"
            f"Response: <think>\n{reasoning}\n</think>\n"
            f"<tool_call>{{\"name\":\"read\",\"args\":{{\"path\":\"reasoning_trace.txt\"}}}}</tool_call>\n"
            f"<tool_result>\n{reasoning}\n</tool_result>\n"
            f"<tool_call>{{\"name\":\"read\",\"args\":{{\"path\":\"final_answer.txt\"}}}}</tool_call>\n"
            f"<tool_result>\n{s['answer']}\n</tool_result>\n"
        )
    return "\n".join(parts) + "\n"


_V5_SUFFIX = (
    "Now solve the following question using the same format "
    "(think in <think> tags, then emit a <tool_call> reading reasoning_trace.txt "
    "with its <tool_result> containing your reasoning, "
    "then a <tool_call> reading final_answer.txt with its <tool_result> containing your answer):\n\n"
    "Question:\n{test_q}"
)


_RENDERERS = {
    "V0": _render_v0, "V1": _render_v1, "V2": _render_v2,
    "V3": _render_v3, "V4": _render_v4, "V5": _render_v5,
}
_SUFFIXES = {
    "V0": _V0_SUFFIX, "V1": _V1_SUFFIX, "V2": _V2_SUFFIX,
    "V3": _V3_SUFFIX, "V4": _V4_SUFFIX, "V5": _V5_SUFFIX,
}


def render_user_content(
    demos: list[dict],
    target_q: str,
    bench: str,
    wrap: str,
    target_meta: dict | None = None,
    extra_hint: str | None = None,
) -> str:
    """Assemble the full REP prompt body (k demonstrations + trailer).

    ``demos`` is the list of k shot triples ({q,r,a}); ``wrap`` selects one of
    ``VARIANT_IDS``. ``extra_hint`` (e.g. an output-format instruction) is
    injected just before the final target ``Question:``.
    """
    if wrap not in _RENDERERS:
        raise ValueError(f"unknown wrap variant: {wrap}")
    shots = _normalize_demos(demos)
    example_text = _RENDERERS[wrap](shots)
    suffix = _SUFFIXES[wrap].format(test_q=target_q)
    body = example_text + "\n" + suffix
    if not extra_hint:
        return body
    needle = "Question:\n"
    idx = body.rfind(needle)
    if idx == -1:
        return body + f"\n\n{extra_hint}"
    return body[:idx] + extra_hint + "\n\n" + body[idx:]
