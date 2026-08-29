from __future__ import annotations

import re

_BOXED_TOKENS = ("\\boxed", "\\fbox")


def last_boxed_only_string(text: str) -> str | None:
    """Return the last `\\boxed{...}` (or `\\fbox{...}`) substring with brace matching.

    Returns None when no boxed expression is found. Brace matching is required because
    boxed contents commonly contain nested braces such as `\\frac{a}{b}`.

    Adapted from hendrycks/math/modeling/dataset/util.py:last_boxed_only_string.
    """
    if not text:
        return None
    last_idx = -1
    last_token = None
    for token in _BOXED_TOKENS:
        idx = text.rfind(token)
        if idx > last_idx:
            last_idx = idx
            last_token = token
    if last_idx < 0 or last_token is None:
        return None

    i = last_idx
    depth = 0
    started = False
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return text[last_idx : i + 1]
        i += 1
    return None


def remove_boxed(boxed_text: str) -> str | None:
    """Strip the `\\boxed{...}` or `\\fbox{...}` wrapper from a boxed string."""
    if boxed_text is None:
        return None
    text = boxed_text.strip()
    for token in _BOXED_TOKENS:
        prefix = token + "{"
        if text.startswith(prefix) and text.endswith("}"):
            return text[len(prefix) : -1]
    legacy = re.fullmatch(r"\\(?:boxed|fbox)\s+(.*)", text)
    if legacy is not None:
        return legacy.group(1)
    return None


def extract_last_boxed_content(text: str) -> str | None:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)
