"""Send pre-rendered prompts to DeepInfra `/v1/openai/completions` (raw prompt
mode), byte-identical to what the SGLang farm receives. Greedy
(temperature=0, top_p=1), per-row max_tokens. Checkpointed + resumable +
retry. Auth: env DEEPINFRA_API_KEY.

Context rule: DeepInfra serves Qwen3-14B/32B at 40960 total tokens (no YaRN).
Each row's max_tokens must satisfy prompt + output <= 40960 (build_prompts.py
sets ctx - ptok - 256); on an HTTP 400 "context length" error the runner
shrinks max_tokens by 512 and retries.

Output rows: {"idx", "text", "finish_reason", "prompt_tokens",
              "completion_tokens", "error"}; errored rows are retried on rerun.

Usage:
  python run_deepinfra.py --model Qwen/Qwen3-14B --prompts prompts/qwen3-14b.jsonl \\
      --out outputs/deepinfra_14b.jsonl --concurrency 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

API = "https://api.deepinfra.com/v1/openai/completions"
ID_KEY = "idx"


def api_key() -> str:
    k = os.environ.get("DEEPINFRA_API_KEY")
    if not k:
        raise SystemExit("DEEPINFRA_API_KEY not set")
    return k


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in out_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("text") is not None and r.get("error") is None:
                done.add(r[ID_KEY])
    return done


async def one(client, sem, row, model, key, out_f, lock, stats, attempts):
    idx = row[ID_KEY]
    max_tokens = int(row["max_tokens"])
    err = "exhausted"
    async with sem:
        for attempt in range(attempts):
            payload = {"model": model, "prompt": row["prompt"], "max_tokens": max_tokens,
                       "temperature": 0.0, "top_p": 1.0}
            try:
                resp = await client.post(API, headers={"Authorization": f"Bearer {key}"}, json=payload,
                                         timeout=httpx.Timeout(900.0, connect=30.0))
                if resp.status_code == 400 and "context length" in resp.text.lower():
                    max_tokens = max(256, max_tokens - 512)   # provider tokenizes slightly differently
                    continue
                if resp.status_code == 200:
                    d = resp.json()
                    ch = d["choices"][0]
                    rec = {ID_KEY: idx, "text": ch["text"], "finish_reason": ch.get("finish_reason"),
                           "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
                           "completion_tokens": d.get("usage", {}).get("completion_tokens"),
                           "error": None}
                    async with lock:
                        out_f.write(json.dumps(rec) + "\n")
                        out_f.flush()
                        stats["ok"] += 1
                        if stats["ok"] % 25 == 0:
                            print(f"  ok={stats['ok']} err={stats['err']} ({time.time() - stats['t0']:.0f}s)", flush=True)
                    return
                if resp.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 * (attempt + 1) + attempt * 3)
                    continue
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(2 * (attempt + 1))
    async with lock:
        out_f.write(json.dumps({ID_KEY: idx, "text": None, "error": err}) + "\n")
        out_f.flush()
        stats["err"] += 1
        print(f"  ERR idx={idx}: {err}", flush=True)


async def run(args) -> None:
    key = api_key()
    prompts = [json.loads(l) for l in Path(args.prompts).open() if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [p for p in prompts if p[ID_KEY] not in done]
    print(f"[{args.model}] total={len(prompts)} done={len(done)} todo={len(todo)} concurrency={args.concurrency}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "err": 0, "t0": time.time()}
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        with out_path.open("a") as out_f:
            await asyncio.gather(*[one(client, sem, row, args.model, key, out_f, lock, stats, args.attempts)
                                   for row in todo])
    print(f"[{args.model}] DONE ok={stats['ok']} err={stats['err']} ({time.time() - stats['t0']:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="DeepInfra model id, e.g. Qwen/Qwen3-14B")
    ap.add_argument("--prompts", required=True, help="jsonl from build_prompts.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
