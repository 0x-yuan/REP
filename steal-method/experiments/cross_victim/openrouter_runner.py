"""OpenRouter harvest path for reasoning-channel victims (e.g. Gemma-4-31B).

Some cross-victim models expose their hidden reasoning as a *separate* API field
(`message.reasoning`) rather than inline in the visible response, and are served
via OpenRouter rather than the self-hosted SGLang farm. This runner is the API
counterpart to the farm: it reads the same inbox JSONL that
``build_prompts.py`` writes and produces results in the same schema the farm's
slave emits, so ``../../../eval/trace_metrics/score.py`` works unchanged.

All external API calls in this artifact go through OpenRouter. Set the key in
``.env`` (or the environment):

    OPENROUTER_API_KEY=sk-or-...

Output schema (mirrors the SGLang slave's get_results format):
    {"id", "prompt_tokens",
     "outputs": [{"text", "reasoning", "finish_reason",
                  "completion_tokens", "reasoning_tokens"}],
     "error"}

CLI::

    python openrouter_runner.py \
        --model gemma-4-31b --openrouter-id google/gemma-4-31b-it \
        --phases baseline,attack --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp

_HERE = Path(__file__).resolve()
STEAL_METHOD = _HERE.parents[2]
PUBLIC_ROOT = STEAL_METHOD.parent
ENV_PATH = PUBLIC_ROOT / ".env"
FARM = STEAL_METHOD / "inference-farm"
INBOX_DIR = FARM / "inbox"
RESULT_DIR = FARM / "result"
PROCESSED_DIR = FARM / "processed"

URL = "https://openrouter.ai/api/v1/chat/completions"


def load_api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    raise RuntimeError(
        "OPENROUTER_API_KEY not set (env or .env). All API victims go "
        "through OpenRouter; see .env.example."
    )


async def one_request(session, row, openrouter_model, api_key, timeout_s,
                      reasoning_effort, max_attempts=3) -> dict:
    payload = {
        "model": openrouter_model,
        "messages": row["messages"],
        "temperature": row.get("temperature", 0.0),
        "top_p": row.get("top_p", 1.0),
        "max_tokens": row.get("max_tokens", 20000),
        "stream": False,
        "reasoning": {"effort": reasoning_effort},
        "include_reasoning": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "REP cross-victim transfer",
    }
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with session.post(
                URL, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as r:
                if r.status >= 500 or r.status == 429:
                    last_err = f"HTTP {r.status}: {(await r.text())[:200]}"
                    await asyncio.sleep(2 ** attempt)
                    continue
                if r.status >= 400:
                    return _err_result(row, f"HTTP {r.status}: {(await r.text())[:500]}")
                data = await r.json()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(2 ** attempt)
            continue
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or msg.get("reasoning_text", "") or ""
        usage = data.get("usage", {}) or {}
        return {
            "id": row["id"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "outputs": [{
                "text": text,
                "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            }],
            "error": None,
        }
    return _err_result(row, last_err or "exhausted retries")


def _err_result(row: dict, msg: str) -> dict:
    return {
        "id": row["id"], "prompt_tokens": 0,
        "outputs": [{"text": "", "reasoning": "", "finish_reason": None,
                     "completion_tokens": 0, "reasoning_tokens": 0}],
        "error": msg,
    }


async def run_phase(model, openrouter_model, phase, concurrency, api_key,
                   timeout_s, reasoning_effort, limit) -> dict:
    inbox = INBOX_DIR / f"{model}__cross_victim_{phase}.jsonl"
    out_path = RESULT_DIR / f"{model}__cross_victim_{phase}.jsonl"
    if not inbox.exists():
        print(f"  SKIP {phase} — inbox missing: {inbox}")
        return {"phase": phase, "n": 0, "skipped": True}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in inbox.open() if l.strip()]
    if limit:
        rows = rows[:limit]

    # Resume by id if a partial result file exists.
    seen_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                seen_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    pending = [r for r in rows if r["id"] not in seen_ids]
    print(f"\n=== {model} :: {phase} ===  inbox n={len(rows)} pending={len(pending)} "
          f"concurrency={concurrency}")
    if not pending:
        return {"phase": phase, "n": len(rows), "n_new": 0, "n_err": 0}

    sem = asyncio.Semaphore(concurrency)
    out_lock = asyncio.Lock()
    n_done = len(seen_ids)
    n_err = 0
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        async def worker(row):
            nonlocal n_done, n_err
            async with sem:
                res = await one_request(session, row, openrouter_model, api_key,
                                        timeout_s, reasoning_effort)
            async with out_lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                n_done += 1
                if res.get("error"):
                    n_err += 1
                if n_done % 25 == 0 or n_done == len(rows):
                    dt = time.time() - t0
                    print(f"    {n_done}/{len(rows)} ({n_err} err) "
                          f"elapsed={dt/60:.1f}m", flush=True)
        await asyncio.gather(*[asyncio.create_task(worker(r)) for r in pending])

    if n_done >= len(rows):
        try:
            inbox.rename(PROCESSED_DIR / inbox.name)
        except FileExistsError:
            inbox.unlink()
    return {"phase": phase, "n": len(rows), "n_done": n_done, "n_err": n_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-31b", help="local victim key (file naming)")
    ap.add_argument("--openrouter-id", default="google/gemma-4-31b-it")
    ap.add_argument("--phases", default="baseline,attack",
                    help="baseline first so per-victim r0 is available for scoring")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=None, help="cap rows per phase (smoke)")
    args = ap.parse_args()

    api_key = load_api_key()
    print(f"OpenRouter runner: {args.openrouter_id} (local key={args.model}) "
          f"phases={args.phases} effort={args.reasoning_effort}")
    for phase in args.phases.split(","):
        res = asyncio.run(run_phase(
            args.model, args.openrouter_id, phase, args.concurrency,
            api_key, args.timeout_s, args.reasoning_effort, args.limit,
        ))
        print(f"  {res}")


if __name__ == "__main__":
    main()
