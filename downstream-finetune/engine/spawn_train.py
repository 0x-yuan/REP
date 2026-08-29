"""Spawn one detached FunctionCall on the training app and log its ID.

Run after `modal deploy train.py`.

    uv run --with modal python spawn_train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _config_loader import cfg  # noqa: E402

LOG = HERE / "logs" / "spawn_train.fclog.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    run_fn = modal.Function.from_name(cfg.TRAIN_APP, "run")
    fc = run_fn.spawn()  # uses all cfg defaults
    rec = {
        "ts": time.time(),
        "fc_id": fc.object_id,
        "run_id": cfg.RUN_ID,
        "source_repo": cfg.SOURCE_REPO,
        "source_config": getattr(cfg, "SOURCE_CONFIG", None),
        "epochs": cfg.EPOCHS,
        "gpu": f"{cfg.GPU_TYPE}:{cfg.GPU_COUNT}",
    }
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[spawn_train] spawned fc={fc.object_id} run_id={cfg.RUN_ID}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
