"""30-min heartbeat tick for ONE distill run.

  1. Check that the orchestrator daemon is alive; relaunch if dead.
  2. Pull the list of ckpts on the train volume + the list of summary files
     on the results volume. Print a 5-bullet snapshot.
  3. Auto-repair the math_block-init failure mode (lcb_v5 placeholder with
     status=failed and no other summary): wipe it + drop the FC from
     in_flight so the orchestrator re-spawns on its next loop.
  4. Update logs/state.json with the latest snapshot.
  5. When ALL N ckpts (base + EPOCHS epoch ckpts) hit `lcb_v5.summary.json`
     successfully, emit final_results.json + touch .DONE.

This script is safe to run more often than every 30 min — it is idempotent.

Recommended crontab:
    */30 * * * * cd /path/to/engine && MODAL_PROFILE=... uv run \\
        --with modal python cron_tick.py >> logs/cron.log 2>&1
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _config_loader import cfg  # noqa: E402

LOG_DIR = ROOT / "logs"
STATE_PATH = LOG_DIR / "state.json"
DONE_PATH = ROOT / ".DONE"
FINAL_PATH = ROOT / "final_results.json"

BENCHES = ["aime24", "aime25", "math500", "jeebench", "lcb_v5"]


def _env() -> dict:
    return {**os.environ, "MODAL_PROFILE": cfg.PROFILE}


def _modal_ls(volume: str, prefix: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["modal", "volume", "ls", volume, prefix],
            capture_output=True, text=True, check=False, timeout=30,
            env=_env(),
        )
        if proc.returncode != 0:
            return []
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    except Exception as exc:  # noqa: BLE001
        print(f"[ls warn] {volume}:{prefix}: {exc}", flush=True)
        return []


def _list_ckpts_on_volume() -> list[int]:
    rows = _modal_ls(cfg.CKPTS_VOL, f"{cfg.RUN_ID}/")
    steps: list[int] = []
    for ln in rows:
        if "checkpoint-" in ln:
            try:
                s = ln.split("checkpoint-")[-1].rstrip("/").split("/")[0]
                steps.append(int(s))
            except Exception:
                pass
    return sorted(set(steps))


def _summary_present(ckpt_label: str, bench: str) -> bool:
    rows = _modal_ls(cfg.RESULTS_VOL, f"{cfg.RUN_ID}/{ckpt_label}/")
    for ln in rows:
        if ln.endswith(f"{bench}.summary.json"):
            if bench == "lcb_v5":
                # Distinguish a real "done" from the math_block-crash placeholder.
                try:
                    proc = subprocess.run(
                        ["modal", "volume", "get", cfg.RESULTS_VOL,
                         f"{cfg.RUN_ID}/{ckpt_label}/lcb_v5.summary.json", "-", "--force"],
                        capture_output=True, text=True, check=False, timeout=30,
                        env=_env(),
                    )
                    if proc.returncode != 0:
                        return False
                    body = proc.stdout
                    i = body.find("{")
                    if i < 0:
                        return False
                    data, _ = json.JSONDecoder().raw_decode(body[i:])
                    return data.get("status") != "failed"
                except Exception:
                    return False
            return True
    return False


def _get_summary_metric(ckpt_label: str, bench: str) -> float | None:
    """Pull the .summary.json for one (ckpt, bench) and return the headline metric."""
    try:
        proc = subprocess.run(
            ["modal", "volume", "get", cfg.RESULTS_VOL,
             f"{cfg.RUN_ID}/{ckpt_label}/{bench}.summary.json", "-", "--force"],
            capture_output=True, text=True, check=False, timeout=30,
            env=_env(),
        )
        if proc.returncode != 0:
            return None
        body = proc.stdout
        i = body.find("{")
        if i < 0:
            return None
        data, _ = json.JSONDecoder().raw_decode(body[i:])
        # multibench_runner stores its headline number under `metric` for most
        # benches; LCB also writes a `pass@1` style key. Try both.
        for key in ("metric", "pass@1", "accuracy", "score"):
            v = data.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return None
    except Exception:
        return None


def _orchestrator_pid() -> int | None:
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid,command"],
            capture_output=True, text=True, check=False, timeout=10,
        ).stdout
    except Exception:
        return None
    needle = str(ROOT / "orchestrate.py")
    needle_short = "orchestrate.py"
    for line in out.splitlines():
        if "grep" in line:
            continue
        # Prefer absolute-path match when we can find one.
        if needle in line:
            try:
                return int(line.strip().split()[0])
            except Exception:
                continue
    # Fallback: any orchestrate.py whose CWD is THIS folder (best effort —
    # `ps` doesn't show CWD, so we accept any orchestrate.py if no exact match).
    for line in out.splitlines():
        if "grep" in line or needle_short not in line:
            continue
        try:
            return int(line.strip().split()[0])
        except Exception:
            continue
    return None


def _relaunch_orchestrator() -> str:
    log = LOG_DIR / "orchestrator.stdout.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"MODAL_PROFILE={cfg.PROFILE} nohup uv run --with modal python "
        f"{ROOT}/orchestrate.py --include-base "
        f">> {log} 2>&1 < /dev/null & disown"
    )
    subprocess.Popen(["bash", "-c", cmd], cwd=str(ROOT),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return cmd


def _kill_orchestrator() -> None:
    pid = _orchestrator_pid()
    if pid is None:
        return
    try:
        subprocess.run(["kill", "-TERM", str(pid)], check=False, timeout=10)
        for _ in range(10):
            if _orchestrator_pid() is None:
                break
            time.sleep(0.5)
        print(f"killed orchestrator pid={pid} for autorepair", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"kill warn: {exc}", flush=True)


def _autorepair_stuck_lcb_failed() -> int:
    """For each ckpt whose ONLY landed file is lcb_v5.summary.json status=failed,
    wipe the placeholder + drop the matching entry from in_flight so the
    orchestrator re-spawns.
    """
    in_flight_path = LOG_DIR / f"in_flight.{cfg.RUN_ID}.json"
    if not in_flight_path.exists():
        return 0
    try:
        in_flight = json.loads(in_flight_path.read_text())
    except Exception:
        return 0
    cleaned = 0
    for key in list(in_flight):
        step_str, kind = key.split(":")
        if kind != "main":
            continue
        step = int(step_str)
        label = "base" if step == 0 else f"step-{step:05d}"
        rows = _modal_ls(cfg.RESULTS_VOL, f"{cfg.RUN_ID}/{label}/")
        if not rows:
            continue
        has_other = any(
            any(ln.endswith(f"{b}.summary.json") for ln in rows)
            for b in ("aime24", "aime25", "math500", "jeebench")
        )
        if has_other:
            continue
        if not any(ln.endswith("lcb_v5.summary.json") for ln in rows):
            continue
        try:
            proc = subprocess.run(
                ["modal", "volume", "get", cfg.RESULTS_VOL,
                 f"{cfg.RUN_ID}/{label}/lcb_v5.summary.json", "-", "--force"],
                capture_output=True, text=True, check=False, timeout=30, env=_env(),
            )
            body = proc.stdout
            i = body.find("{")
            if i < 0:
                continue
            data, _ = json.JSONDecoder().raw_decode(body[i:])
            if data.get("status") != "failed":
                continue
        except Exception:
            continue
        for sub in (
            f"{cfg.RUN_ID}/{label}/lcb_v5.summary.json",
            f"{cfg.RUN_ID}/{label}/lcb_v5",
        ):
            subprocess.run(
                ["modal", "volume", "rm", cfg.RESULTS_VOL, sub, "--recursive"],
                capture_output=True, text=True, check=False, timeout=30, env=_env(),
            )
        in_flight.pop(key, None)
        cleaned += 1
        print(
            f"auto-repair: cleaned math-block placeholder for {label} "
            f"(FC {key} dropped from in_flight)",
            flush=True,
        )
    if cleaned:
        _kill_orchestrator()
        in_flight_path.write_text(json.dumps(in_flight, indent=2))
    return cleaned


def _snapshot() -> dict:
    steps = _list_ckpts_on_volume()
    labels = ["base"] + [f"step-{s:05d}" for s in steps]
    by_ckpt = {
        lab: {b: _summary_present(lab, b) for b in BENCHES}
        for lab in labels
    }
    n_done_main = sum(1 for lab in labels if by_ckpt[lab]["lcb_v5"])
    return {
        "run_id": cfg.RUN_ID,
        "profile": cfg.PROFILE,
        "steps": steps,
        "n_ckpts": len(steps),
        "n_done_main": n_done_main,
        "by_ckpt": by_ckpt,
        "orchestrator_alive": _orchestrator_pid() is not None,
    }


def _write_final_results(snap: dict) -> None:
    """Pull every (ckpt, bench) metric and dump to final_results.json."""
    labels = ["base"] + [f"step-{s:05d}" for s in snap["steps"]]
    out: dict[str, dict[str, float | None]] = {}
    for lab in labels:
        out[lab] = {b: _get_summary_metric(lab, b) for b in BENCHES}
    FINAL_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {FINAL_PATH}", flush=True)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== cron_tick {datetime.now().isoformat(timespec='seconds')} "
          f"({cfg.RUN_ID}) ===", flush=True)

    repaired = _autorepair_stuck_lcb_failed()
    if repaired:
        print(f"auto-repaired {repaired} stale placeholder(s)", flush=True)

    snap = _snapshot()
    print(
        f"[{snap['run_id']}] ckpts={snap['n_ckpts']} done_main={snap['n_done_main']} "
        f"orch_alive={snap['orchestrator_alive']}",
        flush=True,
    )
    if not snap["orchestrator_alive"]:
        print("orchestrator DEAD — relaunching", flush=True)
        _relaunch_orchestrator()

    STATE_PATH.write_text(json.dumps(
        {
            "ts": time.time(),
            "iso": datetime.now().isoformat(timespec="seconds"),
            "experiment": snap,
        },
        indent=2,
    ))

    expected = cfg.EPOCHS + 1  # base + N epoch ckpts
    if snap["n_done_main"] >= expected:
        print("=== ALL DONE — emitting final_results.json ===", flush=True)
        _write_final_results(snap)
        DONE_PATH.touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
