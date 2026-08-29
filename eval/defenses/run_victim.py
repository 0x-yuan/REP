"""Send pre-rendered prompt rows (raw-prompt mode) to an OpenAI-compatible
``/completions`` endpoint, greedy, per-row ``max_tokens``. Checkpointed,
resumable, retried. This is how the paper's DeepInfra victim outputs (no-defense
baseline and Defense B) were produced; output rows feed ``score_defense.py``.

    DEEPINFRA_API_KEY=... uv run python run_victim.py --model Qwen/Qwen3-14B \\
        --prompts prompts/defenseB_agarwal.jsonl --out outputs/defB_14b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from defenses import _api  # noqa: E402
from defenses.prompt_lib import row_idx  # noqa: E402


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("text") is not None and r.get("error") is None:
                done.add(str(r["idx"]))
    return done


def one(row: dict, model: str) -> dict:
    idx = row_idx(row)
    max_tokens = int(row["max_tokens"])
    err = "unknown"
    for _ in range(6):
        try:
            rec = _api.complete(model, row["prompt"], max_tokens)
            return {"idx": idx, **rec, "error": None}
        except _api.Timeout:
            err = "TIMEOUT"
        except RuntimeError as e:
            err = str(e)
            if "context length" in err.lower():
                # server tokenizes slightly differently; shrink and retry
                max_tokens = max(256, max_tokens - 512)
                continue
            if not err.startswith("HTTP"):
                time.sleep(2)
                continue
            break
    return {"idx": idx, "text": None, "error": err}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="e.g. Qwen/Qwen3-14B")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.prompts).read_text().splitlines() if ln.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if row_idx(r) not in done]
    print(f"[{args.model}] total={len(rows)} done={len(done)} todo={len(todo)} "
          f"concurrency={args.concurrency}", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    stats = {"ok": 0, "err": 0, "t0": time.time()}
    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, r, args.model) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                stats["ok" if rec["error"] is None else "err"] += 1
                n = stats["ok"] + stats["err"]
                if rec["error"]:
                    print(f"  ERR idx={rec['idx']}: {rec['error']}", flush=True)
                elif n % 25 == 0:
                    print(f"  ok={stats['ok']} err={stats['err']} ({time.time()-stats['t0']:.0f}s)", flush=True)
    print(f"[{args.model}] DONE ok={stats['ok']} err={stats['err']} ({time.time()-stats['t0']:.0f}s)")


if __name__ == "__main__":
    main()
