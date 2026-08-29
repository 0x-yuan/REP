"""Pure row transforms for the Table-2 control / oracle teacher datasets.

Three teacher-data flavours consumed by ``configs/{oracle,summary,answer_only}_*.py``:

* **oracle** (``configs/oracle_*.py``): the victim's *own* internal ``<think>``
  trace, obtained WITHOUT any attack (plain user turn, thinking enabled), on the
  same 10k clean-corpus questions. ``oracle_row`` re-synthesises a raw-harvest-shaped
  row so the canonical training loader (``engine/_common/dataset_prep.py``)
  sees ``completion = <think>r1</think>\\n\\n r1 + **Final Answer** \\boxed{}``:
  the post-``</think>`` target IS the internal reasoning.
* **summary** (``configs/summary_*.py``): Qwen2.5-7B-Instruct compresses the
  oracle trace ``r1`` into a short solution ending in ``\\boxed{}``
  (``summary_messages`` is the exact prompt); the summary is merged back as a
  ``summary`` column, then ``variant_row(kind="summary_answer")`` wraps it.
* **answer-only** (``configs/answer_only_*.py``): ``variant_row(kind="answer_only")``
  keeps only ``**Final Answer**\\n\\boxed{gold}``.

Variant rows use the engineered ``completion = "<think>\\n</think>\\n\\n{target}"``
so the canonical loader's post-``</think>`` extraction returns ``target``
unchanged.
"""
from __future__ import annotations

import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
BOXED_RE = re.compile(r"\\boxed\{")

ORACLE_CELL_ID = "ideal_no_attack_{victim}"
ORACLE_SOURCE = "ideal_no_attack"

# --------------------------------------------------------------------------- #
# Oracle (no-attack) trace                                                     #
# --------------------------------------------------------------------------- #

def extract_think_block(text: str) -> tuple[str, str]:
    """Return ``(r1, post_think)``.

    Handles both ``<think>r1</think>rest`` and the served-Qwen3 layout where
    the opening ``<think>\\n`` lives in the prompt, so the output starts with
    the trace and contains only ``</think>``. ``("", text)`` if no close tag.
    """
    if THINK_CLOSE not in text:
        return "", text
    if THINK_OPEN in text:
        a = text.index(THINK_OPEN) + len(THINK_OPEN)
        b = text.index(THINK_CLOSE)
        if a < b:
            return text[a:b].strip(), text[b + len(THINK_CLOSE):].lstrip()
    b = text.index(THINK_CLOSE)
    return text[:b].strip(), text[b + len(THINK_CLOSE):].lstrip()


def extract_last_boxed(text: str) -> str | None:
    """Brace-matched last ``\\boxed{...}`` inner content; None if absent or
    unbalanced (oracle builder semantics — an unbalanced box is no answer)."""
    last_start = None
    for m in BOXED_RE.finditer(text):
        last_start = m.end()
    if last_start is None:
        return None
    depth, i = 1, last_start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[last_start:i].strip()
        i += 1
    return None


def math_equiv(gold: str, pred: str) -> bool:
    """math-verify equivalence on raw answer strings; exact-string fallback."""
    try:
        from math_verify import parse, verify  # type: ignore
        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        return gold.strip() == pred.strip()


def oracle_completion(r1: str, answer: str) -> str:
    """The synthesised raw-harvest-compatible completion for an oracle row."""
    return f"<think>\n{r1}\n</think>\n\n{r1}\n\n**Final Answer**\n\\boxed{{{answer}}}\n"


def oracle_row(rec: dict, meta: dict, *, victim: str) -> tuple[dict | None, str | None]:
    """Farm output row (no-attack generation) + source metadata -> oracle row.

    ``rec`` = ``{"id": prompt_id, "outputs": [{"text", "finish_reason",
    "completion_tokens"}], "error"?}``; ``meta`` = ``{source_index, question,
    gold_boxed}`` of the same prompt_id. Returns ``(row, None)`` or
    ``(None, reject_reason)`` with reason in {row_error, no_think, no_boxed}.
    Rows are all ``structural=True``; ``answer_match`` is recorded, not filtered.
    """
    if rec.get("error"):
        return None, "row_error"
    outs = rec.get("outputs") or []
    if not outs:
        return None, "row_error"
    o0 = outs[0]
    text = o0.get("text") or ""
    r1, post = extract_think_block(text)
    if not r1 or not post:
        return None, "no_think"
    ans = extract_last_boxed(post)
    if ans is None:
        return None, "no_boxed"
    gold = (meta.get("gold_boxed") or "").strip()
    match = math_equiv(gold, ans) if gold and ans else False
    return {
        "prompt_id": rec.get("id"),
        "source_index": meta.get("source_index"),
        "question": meta.get("question"),
        "gold_boxed": gold,
        "r1": r1,
        "r2": r1,                       # no attack: post-think target IS the reasoning
        "answer": ans,
        "completion": oracle_completion(r1, ans),
        "completion_tokens": int(o0.get("completion_tokens") or 0),
        "finish_reason": o0.get("finish_reason") or "stop",
        "structural": True,
        "answer_match": bool(match),
        "victim": victim,
        "cell_id": ORACLE_CELL_ID.format(victim=victim),
        "source": ORACLE_SOURCE,
    }, None


# --------------------------------------------------------------------------- #
# Summary prompt (Qwen2.5-7B-Instruct summarises the oracle trace)             #
# --------------------------------------------------------------------------- #

SUMMARY_SYSTEM = (
    "You are an expert mathematics tutor. You will compress a verbose "
    "chain-of-thought reasoning trace into a tighter solution that another "
    "model can imitate during knowledge distillation. The compressed "
    "solution must preserve the full logical chain that leads to the final "
    "answer; it must NOT skip steps."
)

SUMMARY_USER_TEMPLATE = """\
I will give you a math problem and a verbose reasoning trace produced by a strong teacher model. Rewrite the trace as a clean, well-organised solution that another model can learn from during distillation.

REQUIREMENTS
1. Preserve every load-bearing reasoning step the teacher used to reach the answer: problem setup, key definitions, algebraic manipulations, substitutions, identities invoked, numerical computations, case splits, lemmas, and the final verification.
2. Remove pure filler: repeated self-doubt ("wait, let me check"), restatements of the same step in different words, meta-commentary about the model itself, and dead-end branches that the teacher abandoned without using.
3. Keep the solution self-contained — the student should be able to follow it without referring back to the original trace.
4. Write in a flowing, pedagogical style (numbered steps, short paragraphs, inline math). Do NOT write a bullet-point outline.
5. End with the same final answer as the teacher, wrapped exactly as `\\boxed{{...}}`.
6. Output only the compressed solution. No preamble, no "Here is the summary", no commentary outside the solution.

# Problem
{question}

# Verbose reasoning trace
{r1}

# Compressed solution
"""

SUMMARY_COLUMNS = ["summary", "summary_completion_tokens", "summary_finish_reason", "summary_model"]


def summary_messages(question: str, r1: str) -> list[dict]:
    """The exact chat messages sent to the summariser."""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(question=question, r1=r1)},
    ]


def extract_summary(rec: dict, default_model: str) -> dict | None:
    """Farm output row -> the four ``summary*`` columns (None if empty)."""
    outs = rec.get("outputs") or []
    if not outs:
        return None
    o = outs[0]
    text = o.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    ct = o.get("completion_tokens")
    fr = o.get("finish_reason")
    return {
        "summary": text.strip(),
        "summary_completion_tokens": int(ct) if ct is not None else None,
        "summary_finish_reason": str(fr) if fr is not None else None,
        "summary_model": rec.get("model") or default_model,
    }


# --------------------------------------------------------------------------- #
# Variant datasets (answer-only / summary+answer / ideal-think+answer)          #
# --------------------------------------------------------------------------- #

VARIANT_KINDS = ("answer_only", "summary_answer", "ideal_think_answer")


def _boxed(gold: str) -> str:
    return gold if gold.startswith("\\boxed{") and gold.endswith("}") else f"\\boxed{{{gold}}}"


def _ensure_boxed_tail(s: str, gold: str) -> str:
    if "\\boxed{" not in s and gold:
        s = s.rstrip() + f"\n\n**Final Answer**\n{_boxed(gold)}"
    return s


def make_target(row: dict, kind: str) -> str | None:
    """The clean assistant target for one source row, or None to drop."""
    gold = (row.get("gold_boxed") or row.get("answer") or "").strip()
    if kind == "ideal_think_answer":
        r1 = row.get("r1")
        if not isinstance(r1, str) or not r1.strip():
            return None
        return _ensure_boxed_tail(r1.strip(), gold)
    if kind == "answer_only":
        return f"**Final Answer**\n{_boxed(gold)}" if gold else None
    if kind == "summary_answer":
        s = row.get("summary")
        if not isinstance(s, str) or not s.strip():
            return None
        return _ensure_boxed_tail(s.strip(), gold)
    raise ValueError(f"unknown target kind: {kind}")


def wrap_variant_completion(target: str) -> str:
    """Engineered completion the canonical loader unwraps back to ``target``."""
    return f"<think>\n</think>\n\n{target}"


def variant_row(row: dict, kind: str, *, victim: str) -> tuple[dict | None, str | None]:
    """Source (oracle-corpus) row -> variant training row, or a reject reason
    in {struct_drop, empty_q, no_target}."""
    if not bool(row.get("structural")):
        return None, "struct_drop"
    q = row.get("question")
    q = q.strip() if isinstance(q, str) else ""
    if not q:
        return None, "empty_q"
    target = make_target(row, kind)
    if not target:
        return None, "no_target"
    return {
        # engine-facing columns (canonical dataset_prep contract)
        "prompt_id": row.get("prompt_id"),
        "source_index": row.get("source_index"),
        "question": q,
        "gold_boxed": row.get("gold_boxed"),
        "answer": row.get("answer"),
        "completion": wrap_variant_completion(target),
        "structural": True,
        "victim": victim,
        "source": row.get("source"),
        # preview / provenance columns
        "target": target,
        "summary": row.get("summary"),
        "original_completion": row.get("completion"),
        "original_r1": row.get("r1"),
        "original_r2": row.get("r2"),
        "variant": kind,
        "teacher": victim,
    }, None


__all__ = [
    "extract_think_block", "extract_last_boxed", "math_equiv", "oracle_completion",
    "oracle_row", "SUMMARY_SYSTEM", "SUMMARY_USER_TEMPLATE", "SUMMARY_COLUMNS",
    "summary_messages", "extract_summary", "VARIANT_KINDS", "make_target",
    "wrap_variant_completion", "variant_row",
]
