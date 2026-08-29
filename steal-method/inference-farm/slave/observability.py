"""Structured event emitter for the slave.

Every life-cycle stage emits a one-line JSON record on stdout, prefixed
with `[slave]` so it's grep-friendly in `modal app logs`. The same line
also doubles as a human-readable summary thanks to the `event` field
being self-explanatory.

Why structured: a future master process can tail Modal logs and parse
these events to drive a progress UI without scraping free-form text.

Usage:
    from observability import emit, phase

    emit("ready", model_key="qwen3-8b")

    with phase("boot_engine", max_model_len=131072):
        ... long thing ...
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"


def emit(event: str, **fields: Any) -> None:
    """Emit one-line JSON event, flushed immediately so Modal log streaming
    sees it in near real-time."""
    payload: dict[str, Any] = {
        "ts": _now_iso(),
        "event": event,
        "model_key": os.environ.get("MODEL_KEY", "unknown"),
    }
    payload.update(fields)
    line = "[slave] " + json.dumps(payload, ensure_ascii=False, default=str)
    print(line, flush=True)
    # Also flush stderr so any pending tqdm bar finishes its line.
    try:
        sys.stderr.flush()
    except Exception:
        pass


@contextmanager
def phase(name: str, **fields: Any) -> Iterator[None]:
    """Wrap a block of work, emitting `<name>.start` and `<name>.ok` /
    `<name>.fail` with elapsed seconds.

    Use for any step that takes more than a second (model download,
    tokenizer load, vLLM engine boot, batch generation)."""
    emit(f"{name}.start", **fields)
    t0 = time.monotonic()
    try:
        yield
    except BaseException as e:
        emit(
            f"{name}.fail",
            duration_s=round(time.monotonic() - t0, 3),
            error_type=type(e).__name__,
            error=str(e)[:500],
            **fields,
        )
        raise
    else:
        emit(
            f"{name}.ok",
            duration_s=round(time.monotonic() - t0, 3),
            **fields,
        )


def heartbeat_every(seconds: float = 5.0) -> "Heartbeat":
    """Convenience to emit periodic heartbeats during long blocking calls
    (model snapshot download). The returned object can be used as a
    context manager — it spawns a daemon thread and stops on exit."""
    return Heartbeat(seconds)


class Heartbeat:
    def __init__(self, interval_s: float, event: str = "heartbeat") -> None:
        self.interval_s = interval_s
        self.event = event
        self._stop = False
        self._thread = None

    def __enter__(self) -> "Heartbeat":
        import threading

        def _loop() -> None:
            t0 = time.monotonic()
            while not self._stop:
                time.sleep(self.interval_s)
                if self._stop:
                    break
                emit(self.event, elapsed_s=round(time.monotonic() - t0, 1))

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
