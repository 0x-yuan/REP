"""Pure row transforms for the 10k REP distillation corpus (paper §5.5, Table 5).

Everything here is offline text processing, ported byte-for-byte from the
research scripts that produced ``Chia-Mu-Lab/REP-datasets`` configs
``distill_q3_{14b,32b}_{original,clean}`` (the Hub configs are the row schema
below projected to ``question, r1, r2, answer, completion``):

* harvested farm output row  -> corpus row      (``build_corpus_row``)
* corpus row + gold answer     -> clean row     (``to_clean_row``)
* the two Table-5 filters                     (``filter_original`` / ``filter_clean``)

Terminology (paper §5.5):

* **original** ("orig") split — rows of the 10k harvest whose victim output
  parses structurally (``<think>…</think>`` + body); rows without it are marked
  ``structural=False`` and dropped.
* **clean** split — ``structural=True`` AND the extracted ``\\boxed{}`` answer is
  math-verify-equivalent to the OpenThoughts ground-truth answer.
"""
from __future__ import annotations

import re

THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{")

# Prefix stamped on every inbox row id so harvested outputs can be mapped back
# to their prompt_id (``<ID_PREFIX><prompt_id>``).
ID_PREFIX = "b2::"

# Full row schema of the builders; the Hub configs keep only
# ``question, r1, r2, answer, completion``.
corpus_COLUMNS = [
    "prompt_id", "source_index", "question", "r1", "r2", "answer",
    "completion", "completion_tokens", "finish_reason", "structural",
    "victim", "cell_id",
]
CLEAN_COLUMNS = [
    "prompt_id", "source_index", "question", "gold_boxed", "r1", "r2", "answer",
    "completion", "completion_tokens", "finish_reason", "structural",
    "answer_match", "victim", "cell_id", "source",
]


# --------------------------------------------------------------------------- #
# Parsing the victim output                                                    #
# --------------------------------------------------------------------------- #

def parse_think(text: str) -> tuple[str, str, bool]:
    """Return ``(r1, r2, structural)``.

    ``r1`` is the text inside the first ``<think>…</think>`` block, ``r2`` the
    text after ``</think>``. ``structural`` is True iff a well-ordered
    open/close pair exists; otherwise ``r1=""`` and ``r2`` is the whole output.
    """
    if not text:
        return "", "", False
    open_m = THINK_OPEN_RE.search(text)
    close_m = THINK_CLOSE_RE.search(text)
    if not open_m or not close_m or close_m.start() <= open_m.end():
        return "", text.strip(), False
    r1 = text[open_m.end(): close_m.start()].strip()
    r2 = text[close_m.end():].strip()
    return r1, r2, True


def extract_last_boxed(text: str) -> str:
    """Contents of the last ``\\boxed{...}`` (brace-matched). ``""`` if none.

    If the braces never balance, returns the tail after the last ``\\boxed{``
    (stripped) — this mirrors the harvest-side extractor so ``answer`` values
    are identical to the published corpus.
    """
    if not text:
        return ""
    last = -1
    for m in BOXED_RE.finditer(text):
        last = m.end()
    if last < 0:
        return ""
    depth = 1
    i = last
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[last:i]
        i += 1
    return text[last:].strip()


def extract_gold_boxed(text: str) -> str:
    """Gold-side variant used on OpenThoughts solutions: unbalanced tail is
    returned *unstripped* (byte-faithful to the corpus builder)."""
    if not text:
        return ""
    last = -1
    for m in BOXED_RE.finditer(text):
        last = m.end()
    if last < 0:
        return ""
    depth, i = 1, last
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[last:i]
        i += 1
    return text[last:]


def r2_minus_boxed(r2: str, boxed: str) -> str:
    """Strip the trailing ``\\boxed{<boxed>}`` from r2 so r2 is the pre-answer
    exposed-reasoning tail only. Unchanged if the needle is not found."""
    if not boxed:
        return r2
    needle = "\\boxed{" + boxed + "}"
    idx = r2.rfind(needle)
    return r2[:idx].rstrip() if idx >= 0 else r2


# --------------------------------------------------------------------------- #
# Farm output -> corpus row                                                      #
# --------------------------------------------------------------------------- #

def build_corpus_row(out_row: dict, q_lookup: dict[str, dict], *,
                   victim: str, cell_id: str,
                   id_prefix: str = ID_PREFIX) -> dict | None:
    """Convert one inference-farm output row into a corpus row.

    ``out_row`` is the farm's output schema
    ``{"id": "<prefix><prompt_id>", "outputs": [{"text", "finish_reason",
    "completion_tokens"}]}``; ``q_lookup`` maps prompt_id -> ``{prompt_id,
    source_index, question}``. Returns None for rows that cannot be mapped.
    """
    rid = out_row.get("id", "")
    if not rid.startswith(id_prefix):
        return None
    prompt_id = rid[len(id_prefix):]
    q = q_lookup.get(prompt_id)
    if not q:
        return None
    outputs = out_row.get("outputs") or []
    if not outputs:
        return None
    text = outputs[0].get("text") or ""
    finish_reason = outputs[0].get("finish_reason") or ""
    completion_tokens = int(outputs[0].get("completion_tokens", 0) or 0)
    r1, r2_raw, structural = parse_think(text)
    answer = extract_last_boxed(r2_raw if structural else text)
    r2 = r2_minus_boxed(r2_raw, answer)
    return {
        "prompt_id": prompt_id,
        "source_index": int(q["source_index"]),
        "question": q["question"],
        "r1": r1,
        "r2": r2,
        "answer": answer,
        "completion": text,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "structural": bool(structural),
        "victim": victim,
        "cell_id": cell_id,
    }


# --------------------------------------------------------------------------- #
# Answer equivalence + the two filters                                         #
# --------------------------------------------------------------------------- #

def verify_pair(gold: str, pred: str) -> bool:
    """math-verify equivalence of two boxed contents; exact-string fallback
    when math-verify is unavailable or raises (parse failure / timeout)."""
    if not gold or not pred:
        return False
    try:
        from math_verify import parse, verify  # type: ignore
        return bool(verify(parse("\\boxed{" + gold + "}"),
                           parse("\\boxed{" + pred + "}")))
    except Exception:
        return gold.strip() == pred.strip()


def is_original(row: dict) -> bool:
    """Table-5 *orig* split membership: structural extraction succeeded."""
    return bool(row.get("structural"))


def match_gold(row: dict, golds: tuple[str, ...] | list[str]) -> bool:
    """True iff the row's extracted answer matches ANY of the gold candidates.

    The corpus builder checks both the OpenThoughts ``ground_truth_solution``
    and ``deepseek_solution`` boxed values (union), which absorbs LaTeX
    rendering differences such as ``\\frac{1}{2}`` vs ``0.5``.
    """
    pred = row.get("answer") or ""
    golds = [g for g in golds if g]
    if not pred or not golds:
        return False
    return any(verify_pair(g, pred) for g in golds)


def is_clean(row: dict, golds: tuple[str, ...] | list[str]) -> bool:
    """Table-5 *clean* split membership: structural AND answer-correct."""
    return is_original(row) and match_gold(row, golds)


def to_clean_row(row: dict, golds: tuple[str, ...] | list[str], *,
                 source: str, victim: str, cell_id: str) -> dict | None:
    """Promote a corpus row to the clean-corpus schema (``CLEAN_COLUMNS``).

    Returns None if the row is not clean. ``gold_boxed`` records the first
    non-empty gold candidate (ground-truth solution preferred).
    """
    if not is_clean(row, golds):
        return None
    gold_boxed = next((g for g in golds if g), "")
    return {
        "prompt_id":         row["prompt_id"],
        "source_index":      int(row["source_index"]),
        "question":          row["question"],
        "gold_boxed":        gold_boxed,
        "r1":                row.get("r1") or "",
        "r2":                row.get("r2") or "",
        "answer":            row.get("answer") or "",
        "completion":        row.get("completion") or "",
        "completion_tokens": int(row.get("completion_tokens") or 0),
        "finish_reason":     row.get("finish_reason") or "",
        "structural":        True,
        "answer_match":      True,
        "victim":            row.get("victim") or victim,
        "cell_id":           row.get("cell_id") or cell_id,
        "source":            source,
    }


def filter_original(rows: list[dict]) -> list[dict]:
    """*orig* split: keep structurally valid rows, sorted by source_index."""
    return sorted((r for r in rows if is_original(r)),
                  key=lambda r: int(r["source_index"]))


def filter_clean(rows: list[dict], gold_lookup: dict[int, tuple[str, ...]], *,
                 source: str, victim: str, cell_id: str) -> list[dict]:
    """*clean* split: structural + answer-correct rows in clean schema,
    sorted by source_index. ``gold_lookup`` maps source_index -> gold
    candidates (see ``match_gold``)."""
    out: list[dict] = []
    for r in rows:
        golds = gold_lookup.get(int(r["source_index"]), ())
        c = to_clean_row(r, golds, source=source, victim=victim, cell_id=cell_id)
        if c is not None:
            out.append(c)
    out.sort(key=lambda r: r["source_index"])
    return out


def merge_clean(primary: list[dict], topup: list[dict], target: int) -> list[dict]:
    """Union clean rows from the primary harvest and a top-up harvest, keyed by
    source_index (primary wins), sorted by source_index, capped at ``target``.

    This is the selection rule that produced the published 10k clean corpus:
    every clean row of the primary 10k harvest, topped up with clean rows from
    a second disjoint question sample until 10 000 rows.
    """
    by_si: dict[int, dict] = {}
    for r in primary:
        by_si[int(r["source_index"])] = r
    for r in topup:
        by_si.setdefault(int(r["source_index"]), r)
    merged = sorted(by_si.values(), key=lambda r: r["source_index"])
    return merged[:target]


__all__ = [
    "ID_PREFIX", "corpus_COLUMNS", "CLEAN_COLUMNS",
    "parse_think", "extract_last_boxed", "extract_gold_boxed", "r2_minus_boxed",
    "build_corpus_row", "verify_pair", "is_original", "match_gold", "is_clean",
    "to_clean_row", "filter_original", "filter_clean", "merge_clean",
]
