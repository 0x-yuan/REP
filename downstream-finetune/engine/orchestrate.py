"""Orchestrate per-checkpoint multi-bench evals for ONE distill run.

For every new ckpt that appears under the train volume, spawn ONE eval
container that runs the configured `EVAL_BENCH_SUBSET`. The runner is
idempotent (skip-if-summary-exists), so a re-spawn after a partial completion
only fills the missing benches.

Run as a detached nohup daemon (preferred):

    nohup uv run --with modal python orchestrate.py --include-base \
        > logs/orchestrator.stdout.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
sys.path.insert(0, str(HERE))
from _config_loader import cfg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=cfg.RUN_ID)
    p.add_argument("--max-parallel", type=int, default=8,
                   help="Cap on simultaneous in-flight eval FCs.")
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--max-loops", type=int, default=2000)
    p.add_argument("--include-base", action="store_true", default=True)
    p.add_argument("--no-base", dest="include_base", action="store_false")
    p.add_argument("--base-model", default=cfg.MODEL_NAME)
    return p.parse_args()


def _state_path(run_id: str) -> Path:
    return LOG_DIR / f"in_flight.{run_id}.json"


def _load_state(run_id: str) -> dict[str, str]:
    sp = _state_path(run_id)
    if not sp.exists():
        return {}
    with sp.open() as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def _save_state(run_id: str, state: dict[str, str]) -> None:
    sp = _state_path(run_id)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w") as f:
        json.dump(state, f, indent=2)


def _ls_volume(volume: str, prefix: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["modal", "volume", "ls", volume, prefix],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if proc.returncode != 0:
            return []
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


# Summary files we expect for a completed eval. Used by _eval_complete to
# distinguish a real "done" state from a partial / placeholder state (e.g.
# the math_block init bug, which leaves only a status=failed lcb_v5 placeholder).
_REQUIRED_SUMMARIES = ["aime24", "aime25", "math500", "jeebench", "lcb_v5"]


def _eval_complete(volume: str, run_id: str, ckpt_label: str) -> bool:
    """True iff ALL required summary files are present AND lcb_v5 is not status=failed."""
    rows = set(_ls_volume(volume, f"{run_id}/{ckpt_label}/"))
    for b in _REQUIRED_SUMMARIES:
        if not any(ln.endswith(f"{b}.summary.json") for ln in rows):
            return False
    try:
        proc = subprocess.run(
            ["modal", "volume", "get", volume,
             f"{run_id}/{ckpt_label}/lcb_v5.summary.json", "-", "--force"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if proc.returncode != 0:
            return False
        body = proc.stdout
        i = body.find("{")
        if i < 0:
            return False
        import json as _json
        data, _ = _json.JSONDecoder().raw_decode(body[i:])
        return data.get("status") != "failed"
    except Exception:
        return False


def _ckpt_label(step: int) -> str:
    return "base" if step == 0 else f"step-{step:05d}"


def _ckpt_path(step: int, run_id: str, base_model: str) -> str:
    if step == 0:
        return base_model
    return f"/ckpts/{run_id}/checkpoint-{step}"


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    list_ckpts_fn = modal.Function.from_name(cfg.TRAIN_APP, "list_ckpts")

    # Eval pool: one Modal App per ckpt, round-robin assigned. Each app has
    # its own `score_one` function; calling .spawn() on one runs in its own
    # container (own GPU). The mapping ckpt_step → eval_app is sticky for the
    # lifetime of the orchestrator process so retries land on the same app
    # and benefit from any warm vLLM caches.
    eval_pool_names: list[str] = list(cfg.EVAL_APP_POOL)
    eval_pool_fns = {
        name: modal.Function.from_name(name, "score_one")
        for name in eval_pool_names
    }
    step_to_app: dict[int, str] = {}

    def _assign_app(step: int) -> str:
        if step in step_to_app:
            return step_to_app[step]
        used = set(step_to_app.values())
        for name in eval_pool_names:
            if name not in used:
                step_to_app[step] = name
                return name
        name = eval_pool_names[len(step_to_app) % len(eval_pool_names)]
        step_to_app[step] = name
        return name

    in_flight: dict[str, str] = _load_state(args.run_id)
    if in_flight:
        print(f"[orch] resumed in_flight: {sorted(in_flight)}", flush=True)

    def _need_main(label: str) -> bool:
        return not _eval_complete(cfg.RESULTS_VOL, args.run_id, label)

    def _spawn_main(step: int) -> str:
        label = _ckpt_label(step)
        app_name = _assign_app(step)
        fn = eval_pool_fns[app_name]
        fc = fn.spawn(
            ckpt_path=_ckpt_path(step, args.run_id, args.base_model),
            ckpt_label=label,
            run_id=args.run_id,
        )
        print(f"[orch] spawned MAIN step={step} app={app_name} fc={fc.object_id}", flush=True)
        return fc.object_id

    # Seed: base-model eval (step=0) queued first.
    if args.include_base:
        key_main = "0:main"
        if _need_main("base") and key_main not in in_flight:
            in_flight[key_main] = _spawn_main(0)
            _save_state(args.run_id, in_flight)

    loops = 0
    while loops < args.max_loops:
        loops += 1
        try:
            ckpts = list_ckpts_fn.remote(run_id=args.run_id)
        except Exception as e:  # noqa: BLE001
            print(f"[orch] list_ckpts err: {e}", flush=True)
            ckpts = []

        # Reap completed FCs (test by summary file presence).
        for key in list(in_flight):
            step_str, kind = key.split(":")
            step = int(step_str)
            label = _ckpt_label(step)
            if kind == "main" and not _need_main(label):
                print(f"[orch] reaped step={step} kind={kind}", flush=True)
                del in_flight[key]
        _save_state(args.run_id, in_flight)

        # Walk new ckpts in step order, queue missing lanes.
        spawned = 0
        for c in sorted(ckpts, key=lambda x: int(x["step"])):
            step = int(c["step"])
            label = _ckpt_label(step)
            key_main = f"{step}:main"
            if (_need_main(label) and key_main not in in_flight
                    and len(in_flight) < args.max_parallel):
                in_flight[key_main] = _spawn_main(step)
                spawned += 1
        if spawned:
            _save_state(args.run_id, in_flight)

        ckpt_str = ",".join(str(int(c["step"])) for c in sorted(ckpts, key=lambda x: int(x["step"])))
        print(
            f"[orch loop={loops}] ckpts={ckpt_str} | in_flight={sorted(in_flight)} | spawned={spawned}",
            flush=True,
        )
        steps_to_check = {int(c["step"]) for c in ckpts}
        if args.include_base:
            steps_to_check.add(0)
        all_done = (
            not in_flight
            and len(ckpts) >= 1
            and all(not _need_main(_ckpt_label(s)) for s in steps_to_check)
        )
        if all_done:
            print("[orch] all done", flush=True)
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
