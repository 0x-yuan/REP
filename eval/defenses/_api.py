"""Minimal OpenAI-compatible HTTP client (stdlib only) shared by the defense
scripts. Defaults to DeepInfra, which is where the paper's defense numbers were
produced; any OpenAI-compatible endpoint works via ``--api-base``/env.

Env:
  DEFENSE_API_KEY   (falls back to DEEPINFRA_API_KEY)   bearer token
  DEFENSE_API_BASE  (default https://api.deepinfra.com/v1/openai)

The key is read lazily so ``--help`` and unit tests never need it.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"
RETRY_STATUS = {429, 500, 502, 503, 504}


def api_base() -> str:
    return os.environ.get("DEFENSE_API_BASE", DEFAULT_API_BASE).rstrip("/")


def api_key() -> str:
    key = os.environ.get("DEFENSE_API_KEY") or os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise SystemExit("set DEFENSE_API_KEY (or DEEPINFRA_API_KEY) to call the API")
    return key


class Timeout(Exception):
    """Raised when the endpoint did not answer within the serving budget."""


def post_json(path: str, payload: dict, timeout: float = 300.0, retries: int = 4) -> dict:
    """POST ``payload`` to ``<api_base>/<path>``; retry on transient status.
    Raises ``Timeout`` on socket timeout and ``RuntimeError`` on hard errors."""
    url = f"{api_base()}/{path.lstrip('/')}"
    data = json.dumps(payload).encode()
    err = "unknown"
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            if e.code in RETRY_STATUS:
                err = f"HTTP{e.code}:{body}"
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP{e.code}:{body}") from None
        except TimeoutError:
            raise Timeout() from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason):
                raise Timeout() from None
            err = str(e.reason)
            time.sleep(2 * (attempt + 1))
        except OSError as e:  # socket.timeout subclasses OSError on old pythons
            if "timed out" in str(e):
                raise Timeout() from None
            err = str(e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(err)


def chat(model: str, messages: list[dict], max_tokens: int, timeout: float = 300.0) -> tuple[str, str | None]:
    """Chat completion, greedy. Returns (content, finish_reason)."""
    j = post_json("chat/completions", {"model": model, "messages": messages,
                                       "max_tokens": max_tokens, "temperature": 0.0}, timeout)
    ch = j["choices"][0]
    return ch["message"]["content"], ch.get("finish_reason")


def complete(model: str, prompt: str, max_tokens: int, timeout: float = 900.0) -> dict:
    """Raw-prompt completion (pre-rendered chat template), greedy."""
    j = post_json("completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                  "temperature": 0.0, "top_p": 1.0}, timeout)
    ch = j["choices"][0]
    usage = j.get("usage", {})
    return {"text": ch["text"], "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}
