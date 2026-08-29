"""Import the experiment config — `config.py` by default, or the path in
the `DISTILL_CONFIG` env var (forward-compat for multi-variant folders).

Every script in the engine imports `cfg` via:

    from _config_loader import cfg

so a future multi-variant layout (e.g. `configs/clean.py`, `configs/orig.py`)
just needs `DISTILL_CONFIG=configs/clean.py` in the shell — no script edits.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent


def _load() -> ModuleType:
    env_path = os.environ.get("DISTILL_CONFIG", "").strip()
    candidates: list[Path]
    if env_path:
        candidates = [Path(env_path).expanduser().resolve()]
    else:
        # Try both: the script's neighbor (local CLI use) AND /root (Modal
        # container, where train.py copies config.py to /root/config.py).
        candidates = [HERE / "config.py", Path("/root/config.py")]

    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("config", p)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules["config"] = mod
            spec.loader.exec_module(mod)
            return mod

    raise FileNotFoundError(
        f"engine: could not find config.py. Looked at {candidates}. "
        "Set $DISTILL_CONFIG to a path or put config.py next to the scripts."
    )


cfg = _load()
