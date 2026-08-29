"""SGLang slave — single deployable Modal app, one model per deployment.

# Template usage (parallel experiments)

Set `EXP_ID` in `<server-root>/experiment.env` (sourced by
`master/run_pipeline.sh`) before deploying. Every Modal app / volume /
dict name then gets an `${EXP_ID}-` prefix, isolating this copy of the
repo from any sibling copy that's also deploying. The HF cache volume
stays shared across experiments since model weights are immutable.

Per-model field overrides live in `<server-root>/experiment.toml`:

    [models.qwen3-8b]
    default_max_tokens = 4096
    context_length = 65536

The TOML is loaded on the host at deploy time and re-emitted into the
container as a single `EXPERIMENT_OVERRIDES_JSON` env var, so containers
see the same overrides without any filesystem fiddling.


Uses `sgl.Engine` for generation. JSONL batch format, checkpoint+resume
pattern, and RPC + ASGI surface are stable across engines — a drop-in
replacement for the master.

# Public input surface
    1. model_name (chosen at deploy time via MODEL_KEY env var)
    2. batch_file (passed at run time via local entrypoints)

# RPC endpoints (call from the master via SGLangSlave().method.remote(...))
    ping()                  : liveness + model info
    warmup()                : touches engine; triggers cold start if cold
    cold_down()             : signals intent to scale down
    status()                : engine state, in-flight batches
    inference(records, batch_id?, chunk_size?)        : run batch
    inference_from_volume(vol_path, batch_id?, ...)   : read JSONL from /data
    progress(batch_id)      : current per-batch status
    get_results(batch_id)   : checkpointed result rows

# Deploy
    MODEL_KEY=qwen3-8b modal deploy inference-farm/slave/app.py
    # → Modal app `sglang-slave-qwen3-8b` in the currently active profile.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from naming import (  # noqa: E402
    app_name as _make_app_name,
    checkpoint_volume_name,
    data_volume_name,
    fc_dict_name,
    hf_cache_volume_name,
    progress_dict_name,
    resolve_exp_id,
)
from registry import get_config, list_models, load_overrides  # noqa: E402

_MODEL_KEY = os.environ.get("MODEL_KEY", "").strip()
if not _MODEL_KEY:
    raise RuntimeError(
        "MODEL_KEY env var must be set before deploying / running this slave.\n"
        f"Known models: {', '.join(list_models())}"
    )

# REPLICA_ID supports horizontal fanout: one Modal account hosts N independent
# replicas of the same model under unique app names. Empty (default) keeps the
# legacy single-replica naming `sglang-slave-<model>`. With REPLICA_ID="r0",
# the app becomes `sglang-slave-<model>-r0`. Replicas share the same Modal
# Volumes (hf-cache / checkpoints / data) but get their own progress/fc Dicts
# so the master can track each one independently.
_REPLICA_ID = os.environ.get("REPLICA_ID", "").strip()

# EXP_ID is the duplication-namespace prefix. Empty → legacy unprefixed names.
# Read once at import time so the deployed container's resource lookups match
# what the master used at deploy time.
_EXP_ID = resolve_exp_id()

CFG = get_config(_MODEL_KEY)
APP_NAME = _make_app_name(_MODEL_KEY, _REPLICA_ID, _EXP_ID)

# .env lookup (best-effort)
_DOTENV: Path | None = None
if len(_HERE.parents) >= 2:
    candidate = _HERE.parents[1] / ".env"
    if candidate.exists():
        _DOTENV = candidate

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name(
    data_volume_name(_EXP_ID), create_if_missing=True
)
# HF cache volume is intentionally SHARED across experiments — model weights
# are immutable, so cross-experiment reuse only races on the first download.
hf_cache_vol = modal.Volume.from_name(
    hf_cache_volume_name(_EXP_ID), create_if_missing=True
)
checkpoint_vol = modal.Volume.from_name(
    checkpoint_volume_name(_EXP_ID), create_if_missing=True
)

_PROGRESS_DICT_NAME = progress_dict_name(_MODEL_KEY, _REPLICA_ID, _EXP_ID)
progress_dict = modal.Dict.from_name(_PROGRESS_DICT_NAME, create_if_missing=True)
_FC_DICT_NAME = fc_dict_name(_MODEL_KEY, _REPLICA_ID, _EXP_ID)
fc_dict = modal.Dict.from_name(_FC_DICT_NAME, create_if_missing=True)

_SECRETS = (
    modal.Secret.from_dotenv(str(_DOTENV))
    if _DOTENV is not None
    else modal.Secret.from_dict({})
)

# SGLang's official Docker image — bundled with FlashInfer / FA3 wheels
# matched to the SGLang release. Do NOT pip-install flashinfer separately:
# version drift between the runtime and the image-internal wheels causes
# "could not find kernel" runtime errors.
#   - v0.5.10.post1: 2026-04-09 — current stable, has Qwen3 + YaRN +
#     Hopper FA3 + Blackwell SM100 support. (sgl-project/sglang#22711
#     mscale handling fix is queued for next release but doesn't affect
#     Qwen3 since Qwen3's config.json doesn't set mscale.)
_DEFAULT_SGLANG_IMAGE = (
    "lmsysorg/sglang:v0.5.10.post1-cu130-runtime"
)
_SGLANG_IMAGE = CFG.sglang_image or _DEFAULT_SGLANG_IMAGE

image = (
    modal.Image.from_registry(_SGLANG_IMAGE, add_python=None)
    .entrypoint([])
    # Do NOT pip_install `transformers` here. The SGLang official image
    # ships transformers==5.3.0 pinned to its release; downgrading to the
    # vLLM-style `<5.0` range silently breaks Qwen3 chat-template handling
    # and SGLang's tokenizer-manager. fastapi + hf_transfer are the only
    # extras the slave actually needs on top of the base image.
    .pip_install(
        "hf_transfer>=0.1.6",
        "fastapi>=0.115",
    )
    .env(
        {
            "MODEL_KEY": _MODEL_KEY,
            # REPLICA_ID is read by the container at import time to derive
            # the per-replica progress / fc Dict names. Bake it into the
            # image so the value matches the deploy-time APP_NAME (otherwise
            # the deployed app is `sglang-slave-<m>-r0` but the container's
            # APP_NAME / Dict names fall back to the legacy unsuffixed form).
            "REPLICA_ID": _REPLICA_ID,
            # EXP_ID propagates the namespace prefix into the container so
            # progress / volume / dict lookups resolve to the same names the
            # master used at deploy time.
            "EXP_ID": _EXP_ID,
            # Bake host-side overrides into the container env so registry
            # inside the container sees the same overrides as on the host.
            # Empty dict (no experiment.toml) → empty JSON object.
            "EXPERIMENT_OVERRIDES_JSON": json.dumps(
                load_overrides(), sort_keys=True
            ),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # SGLang JIT DeepGEMM is only useful for FP8 weights; harmless
            # to leave on for bf16 (it just doesn't compile any kernels).
            "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
            # Avoid OOM during torchinductor compile if it ever activates.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # SGLang's tokenizer-manager logs are very chatty by default.
            "SGLANG_LOG_LEVEL": "info",
            # SGLang validates ``context_length`` against the base HF config's
            # ``max_position_embeddings`` BEFORE applying our YaRN override.
            # Qwen3 ships max_position_embeddings=40960; without this var,
            # any context_length > 40960 raises ValueError in
            # _derive_context_length. The runtime warning is fine — the
            # YaRN factor=4.0 actually extends the effective ctx to 131072.
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            # ----- NCCL / Torch stability for long-running TP batches -----
            # Surface stuck NCCL collectives as Python exceptions so Modal
            # restarts the container instead of hanging silently. Required
            # for the 12-h batch jobs we run on Qwen3-32B at TP=4.
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "0",
            # 30 min — long enough to outlast a chunk of 32 × 20K decode
            # tokens, short enough to fail fast on a real hang.
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
            # WORLD-group NCCL timeout. Subgroup timeout is still 600s on
            # 0.5.10.post1 (Issue #21911 fix not yet released) — limits TP
            # benefit, but we already hold TP at the smallest viable value.
            "NCCL_TIMEOUT": "3600",
            # WARN over INFO: 12-h logs would be tens of GB at INFO.
            "NCCL_DEBUG": "WARN",
            # Cleaner re-init after a Modal container restart.
            "NCCL_LAUNCH_MODE": "GROUP",
            # Deterministic stream ordering — stabilizes CUDA-graph capture
            # under multi-stream NCCL.
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    .add_local_python_source(
        "registry", "observability", "batch_format", "naming"
    )
)


def _new_progress_record(batch_id: str, n_total: int, chunk_size: int) -> dict:
    return {
        "batch_id": batch_id,
        "model_key": _MODEL_KEY,
        "replica_id": _REPLICA_ID,
        "app_name": APP_NAME,
        "status": "queued",
        "n_total": n_total,
        "n_done": 0,
        "n_invalid": 0,
        "current_row_id": None,
        "chunk_size": chunk_size,
        "submitted_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "elapsed_s": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "tps_decode": None,
        "error": None,
        # Heartbeat fields — refreshed by _ProgressHeartbeat every few seconds
        # during a blocking Engine.generate call so the master sees liveness
        # even on a single long-running chunk.
        "heartbeat_at": None,
        "chunk_index": -1,
        "chunk_n_chunks": 0,
        "chunk_started_at": None,
        "chunk_active_s": 0.0,
        "chunk_size_actual": 0,
        "chunk_row_offset": 0,
    }


class _ProgressHeartbeat:
    """Background thread that publishes liveness + chunk metadata into
    `progress_dict[bid]` every `interval_s` seconds.

    Why: the slave's main thread blocks for the full duration of each
    `Engine.generate(prompts, sps)` call. For prefix-cached batches with
    large `chunk_size`, that's 1-3+ minutes of zero on-the-wire updates
    if we only commit at chunk boundaries. The heartbeat fills that gap
    so the master can show "in-chunk for 90s" instead of going dark.

    Thread-safety: read-modify-write on modal.Dict is racy across writers,
    so both the heartbeat and the main thread go through `commit(updates)`
    which holds a mutex while merging into the dict.
    """

    def __init__(
        self,
        progress_dict_ref,
        batch_id: str,
        interval_s: float = 5.0,
    ) -> None:
        self._dict = progress_dict_ref
        self._bid = batch_id
        self._interval = max(1.0, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Mutable view of the chunk currently in-flight — set_chunk()
        # updates these between Engine.generate calls.
        self._chunk_index = -1
        self._chunk_n_chunks = 0
        self._chunk_started_at: float | None = None
        self._chunk_size_actual = 0
        self._chunk_row_offset = 0

    # -- public API --

    def commit(self, updates: dict) -> None:
        """Merge `updates` into `progress_dict[bid]` atomically."""
        with self._lock:
            try:
                cur = dict(self._dict.get(self._bid, {}))
            except Exception:
                cur = {}
            cur.update(updates)
            try:
                self._dict[self._bid] = cur
            except Exception:
                pass

    def publish_full(self, full_record: dict) -> None:
        """Atomic replace of `progress_dict[bid]`. Use this from the main
        thread to overwrite the whole progress record without racing with
        the background heartbeat tick (both methods take the same lock).
        Any heartbeat-only fields (`heartbeat_at` / `chunk_active_s` / ...)
        that are stale in `full_record` will be refreshed within
        `interval_s` by the next tick — acceptable for a 5s interval.
        """
        with self._lock:
            try:
                self._dict[self._bid] = dict(full_record)
            except Exception:
                pass

    def set_chunk(
        self,
        *,
        chunk_index: int,
        chunk_n_chunks: int,
        chunk_size_actual: int,
        chunk_row_offset: int,
    ) -> None:
        with self._lock:
            self._chunk_index = chunk_index
            self._chunk_n_chunks = chunk_n_chunks
            self._chunk_started_at = time.time()
            self._chunk_size_actual = chunk_size_actual
            self._chunk_row_offset = chunk_row_offset
        # Push immediately so the master sees the chunk start without waiting
        # for the next heartbeat tick.
        self._tick()

    def clear_chunk(self) -> None:
        with self._lock:
            self._chunk_started_at = None
        self._tick()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"hb-{self._bid[:12]}",
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- internal --

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            csa = self._chunk_started_at
            updates = {
                "heartbeat_at": now,
                "chunk_index": self._chunk_index,
                "chunk_n_chunks": self._chunk_n_chunks,
                "chunk_size_actual": self._chunk_size_actual,
                "chunk_row_offset": self._chunk_row_offset,
                "chunk_started_at": csa,
                "chunk_active_s": (round(now - csa, 1) if csa else 0.0),
            }
        # NOTE: cannot hold self._lock while calling commit — commit
        # acquires the same lock.
        self.commit(updates)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                pass
            self._stop.wait(timeout=self._interval)


@app.cls(
    gpu=CFG.gpu,
    image=image,
    volumes={
        "/data": data_vol,
        "/mnt/hf-cache": hf_cache_vol,
        "/mnt/checkpoints": checkpoint_vol,
    },
    timeout=CFG.timeout_s,
    secrets=[_SECRETS],
    min_containers=0,
    scaledown_window=CFG.scaledown_s,
    startup_timeout=CFG.startup_timeout_s,
)
@modal.concurrent(max_inputs=CFG.concurrent_max_inputs)
class SGLangSlave:
    """One container hosts one SGLang Engine for one model."""

    @modal.enter()
    def setup(self) -> None:
        from observability import emit, phase, heartbeat_every  # noqa: E402
        from registry import get_config  # noqa: E402

        os.environ["HF_HOME"] = "/mnt/hf-cache/hf-home"
        os.environ["TRANSFORMERS_CACHE"] = "/mnt/hf-cache/hf-home/transformers"
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

        cfg = get_config(_MODEL_KEY)
        self.cfg = cfg
        self._ready_at: float | None = None
        self._phase: str = "booting"
        self._phase_started_at: float = time.time()
        self._last_error: str | None = None
        self._inflight: set[str] = set()

        emit(
            "container.start",
            hf_id=cfg.hf_id,
            gpu=cfg.gpu,
            tp=cfg.tp,
            context_length=cfg.context_length,
            mem_fraction_static=cfg.mem_fraction_static,
            attention_backend=cfg.attention_backend,
            schedule_policy=cfg.schedule_policy,
            page_size=cfg.page_size,
            watchdog_timeout=cfg.watchdog_timeout,
            rope_yarn_factor=cfg.rope_yarn_factor,
            replica_id=_REPLICA_ID or "(legacy)",
            app_name=APP_NAME,
            engine="sglang",
        )

        try:
            import torch

            if torch.cuda.is_available():
                emit(
                    "gpu.detected",
                    device_count=torch.cuda.device_count(),
                    device_name=torch.cuda.get_device_name(0),
                    cuda_version=torch.version.cuda,
                )
            else:
                emit("gpu.not_detected")
        except Exception as e:
            emit("gpu.detect_failed", error=str(e))

        self._phase = "loading_tokenizer"
        self._phase_started_at = time.time()
        with phase("download_and_load_tokenizer", hf_id=cfg.hf_id):
            from transformers import AutoTokenizer

            with heartbeat_every(10.0):
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)

        self._phase = "boot_engine"
        self._phase_started_at = time.time()
        with phase(
            "boot_engine",
            hf_id=cfg.hf_id,
            tp=cfg.tp,
            context_length=cfg.context_length,
            mem_fraction_static=cfg.mem_fraction_static,
            attention_backend=cfg.attention_backend,
            ep_size=cfg.ep_size,
        ):
            import sglang as sgl

            # ServerArgs flat kwargs for sgl.Engine — the dataclass field
            # names. Keep this minimal-by-default and only forward optional
            # knobs when the registry sets a non-sentinel value, to keep
            # the engine-boot arguments minimal and stable.
            kwargs: dict = dict(
                model_path=cfg.hf_id,
                tp_size=cfg.tp,
                context_length=cfg.context_length,
                mem_fraction_static=cfg.mem_fraction_static,
                attention_backend=cfg.attention_backend,
                disable_radix_cache=cfg.disable_radix_cache,
                disable_cuda_graph=cfg.disable_cuda_graph,
                enable_torch_compile=cfg.enable_torch_compile,
                # Scheduling — lpm is REQUIRED for shared-prefix workloads.
                schedule_policy=cfg.schedule_policy,
                schedule_conservativeness=cfg.schedule_conservativeness,
                page_size=cfg.page_size,
                # Watchdog — covers cold-start CUDA-graph capture (~12 min on
                # 32B TP=4) and worst-case slow chunks.
                watchdog_timeout=cfg.watchdog_timeout,
                # Stability flags (defaults kept OFF for this workload; see
                # registry comments). Pass explicitly so configs can flip them.
                disable_overlap_schedule=cfg.disable_overlap_schedule,
                enable_dp_attention=cfg.enable_dp_attention,
                enable_two_batch_overlap=cfg.enable_two_batch_overlap,
                enable_mixed_chunk=cfg.enable_mixed_chunk,
                trust_remote_code=False,
                log_level="info",
            )

            if cfg.max_running_requests > 0:
                kwargs["max_running_requests"] = cfg.max_running_requests
            if cfg.chunked_prefill_size > 0:
                kwargs["chunked_prefill_size"] = cfg.chunked_prefill_size
            if cfg.cuda_graph_max_bs > 0:
                kwargs["cuda_graph_max_bs"] = cfg.cuda_graph_max_bs
            if cfg.cuda_graph_bs:
                # Explicit bucket list shortens cold-start ~10× by skipping
                # buckets we won't hit (default infers ~58 buckets, capturing
                # 22s each on 32B TP=4 = ~21 min). With max_running_requests
                # capped, only ~6 buckets are actually used.
                kwargs["cuda_graph_bs"] = list(cfg.cuda_graph_bs)
            if cfg.kv_cache_dtype and cfg.kv_cache_dtype != "auto":
                kwargs["kv_cache_dtype"] = cfg.kv_cache_dtype
            if cfg.ep_size and cfg.ep_size > 1:
                kwargs["ep_size"] = cfg.ep_size

            # YaRN — Qwen3's published config has no rope_scaling/yarn
            # block, so we override via json_model_override_args. We set
            # BOTH ``rope_parameters`` (the canonical field in transformers
            # 5.x, which SGLang 0.5.10's qwen3.py reads as
            # ``config.rope_parameters["rope_theta"]``) AND ``rope_scaling``
            # (the legacy alias still consulted by some code paths). The
            # rope_theta value is required because the override path
            # otherwise drops the model's top-level rope_theta and
            # qwen3.py raises KeyError.
            if cfg.rope_yarn_factor:
                rope_block = {
                    "rope_type": "yarn",
                    "factor": cfg.rope_yarn_factor,
                    "original_max_position_embeddings": cfg.rope_orig_max,
                    "rope_theta": cfg.rope_theta,
                }
                kwargs["json_model_override_args"] = json.dumps({
                    "rope_parameters": rope_block,
                    "rope_scaling": rope_block,
                    "rope_theta": cfg.rope_theta,
                })

            with heartbeat_every(15.0):
                self.llm = sgl.Engine(**kwargs)

        self.ckpt_root = Path("/mnt/checkpoints") / cfg.key
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        self._ready_at = time.time()
        self._phase = "ready"
        self._phase_started_at = self._ready_at
        emit("ready", hf_id=cfg.hf_id, ckpt_root=str(self.ckpt_root))

    @modal.exit()
    def shutdown(self) -> None:
        """Tear down SGLang sub-processes (tokenizer-manager, scheduler,
        detokenizer) cleanly. atexit handles this too, but explicit beats
        implicit."""
        try:
            from observability import emit  # noqa: E402

            emit("shutdown.start")
            if hasattr(self, "llm"):
                self.llm.shutdown()
            emit("shutdown.ok")
        except Exception as e:
            try:
                from observability import emit  # noqa: E402

                emit("shutdown.fail", error=str(e)[:200])
            except Exception:
                pass

    # -------- helpers --------

    def _state(self) -> dict:
        now = time.time()
        return {
            "phase": getattr(self, "_phase", "booting"),
            "phase_for_s": (
                round(now - self._phase_started_at, 2)
                if hasattr(self, "_phase_started_at")
                else 0.0
            ),
            "model_key": self.cfg.key,
            "ready_at": getattr(self, "_ready_at", None),
            "warm_for_s": (
                round(now - self._ready_at, 2)
                if getattr(self, "_ready_at", None) is not None
                else 0.0
            ),
            "inflight_batches": sorted(getattr(self, "_inflight", set())),
            "last_error": getattr(self, "_last_error", None),
            "engine": "sglang",
        }

    def _wrap(self, payload: dict) -> dict:
        out = dict(payload)
        out["_state"] = self._state()
        return out

    # -------- liveness / lifecycle endpoints --------

    @modal.method()
    def ping(self) -> dict:
        from observability import emit  # noqa: E402

        emit("ping")
        return self._wrap(
            {
                "ok": True,
                "hf_id": self.cfg.hf_id,
                "context_length": self.cfg.context_length,
            }
        )

    @modal.method()
    def warmup(self) -> dict:
        from observability import emit  # noqa: E402

        emit("warmup")
        return self._wrap({"ok": True})

    @modal.method()
    def cold_down(self) -> dict:
        from observability import emit  # noqa: E402

        emit("cold_down.requested", scaledown_window_s=self.cfg.scaledown_s)
        if not self._inflight:
            self._phase = "draining"
            self._phase_started_at = time.time()
        return self._wrap(
            {
                "ok": True,
                "note": (
                    f"Container will release after ~{self.cfg.scaledown_s}s of idle "
                    "(Modal scaledown_window). Stop sending requests."
                ),
            }
        )

    @modal.method()
    def status(self) -> dict:
        return self._wrap(
            {
                "ok": True,
                "hf_id": self.cfg.hf_id,
                "gpu": self.cfg.gpu,
                "tp": self.cfg.tp,
                "context_length": self.cfg.context_length,
                "default_max_tokens": self.cfg.default_max_tokens,
                "attention_backend": self.cfg.attention_backend,
            }
        )

    # -------- progress endpoint --------

    @modal.method()
    def progress(self, batch_id: str) -> dict:
        try:
            rec = dict(progress_dict[batch_id])
            if rec.get("status") in {"rendering", "generating"} and rec.get("started_at"):
                rec["elapsed_s"] = round(time.time() - rec["started_at"], 2)
        except KeyError:
            rec = {"batch_id": batch_id, "status": "unknown"}
        return self._wrap(rec)

    # -------- main batch endpoint --------

    @modal.method()
    def inference(
        self,
        records: list[dict],
        batch_id: str | None = None,
        chunk_size: int | None = None,
    ) -> list[dict]:
        return self._inference_impl(records, batch_id=batch_id, chunk_size=chunk_size)

    @modal.method()
    def inference_from_volume(
        self,
        vol_path: str,
        batch_id: str | None = None,
        chunk_size: int | None = None,
    ) -> dict:
        from observability import emit  # noqa: E402

        try:
            data_vol.reload()
        except Exception as _e:
            emit("data_vol.reload_failed", error=str(_e)[:200])

        full_path = Path("/data") / vol_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"/data/{vol_path} not found on slave; upload it first."
            )

        emit("inference_from_volume.read_start", path=str(full_path))
        records: list[dict] = []
        with full_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        n = len(records)
        bid = batch_id or str(uuid.uuid4())
        emit("inference_from_volume.read_done", path=str(full_path), n=n, batch_id=bid)

        self._inference_impl(records, batch_id=bid, chunk_size=chunk_size)
        return {"batch_id": bid, "n_total": n}

    def _render_prompt(self, row) -> str:
        """Apply chat template if `messages`, else return raw `prompt`."""
        if row.prompt is not None:
            return row.prompt
        kw: dict = {"tokenize": False}
        if row.continue_final_message:
            kw["continue_final_message"] = True
            kw["add_generation_prompt"] = False
        else:
            kw["add_generation_prompt"] = row.add_generation_prompt
        try:
            return self.tokenizer.apply_chat_template(
                row.messages, enable_thinking=row.enable_thinking, **kw
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(row.messages, **kw)

    def _build_sampling_params(self, row) -> dict:
        """SGLang dict-form sampling params (NOT vLLM's SamplingParams class).
        Field name differences: `max_new_tokens` (not `max_tokens`)."""
        sp: dict = {
            "max_new_tokens": (
                row.max_tokens
                if row.max_tokens is not None
                else self.cfg.default_max_tokens
            ),
            "temperature": row.temperature,
            "top_p": row.top_p,
        }
        if row.top_k is not None:
            sp["top_k"] = row.top_k
        if row.stop:
            sp["stop"] = row.stop
        if row.repetition_penalty and row.repetition_penalty != 1.0:
            sp["repetition_penalty"] = row.repetition_penalty
        # n>1 handled by replicating prompts at the caller (SGLang Engine
        # does not have a native n-samples flag in offline batch mode).
        return sp

    def _inference_impl(
        self,
        records: list[dict],
        batch_id: str | None = None,
        chunk_size: int | None = None,
    ) -> list[dict]:
        from batch_format import (  # noqa: E402
            make_error_row,
            make_output_row,
            parse_row,
        )
        from observability import emit, phase  # noqa: E402

        bid = batch_id or str(uuid.uuid4())
        # Chunk size: caller > registry max_running_requests > 32. We use
        # `max_running_requests` as the hint because that's the true
        # in-flight cap; chunked_prefill_size is a token budget, not a
        # request count.
        cs = int(chunk_size or self.cfg.max_running_requests or 32)
        n_total = len(records)

        # Wipe any stale progress dict from a prior failed run with the
        # same batch_id so the master doesn't race on the previous attempt's
        # status=failed.
        progress_dict[bid] = _new_progress_record(bid, n_total=n_total, chunk_size=cs)

        try:
            checkpoint_vol.reload()
        except Exception as _e:
            emit("checkpoint_vol.reload_failed", error=str(_e)[:200])

        ckpt_dir = self.ckpt_root / bid
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_path = ckpt_dir / "results.jsonl"
        state_path = ckpt_dir / "state.json"

        existing_results: list[dict] = []
        if results_path.exists():
            with results_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        n_resumed = min(len(existing_results), n_total)

        prog = _new_progress_record(bid, n_total=n_total, chunk_size=cs)
        prog["n_done"] = n_resumed
        prog["status"] = "resuming" if n_resumed > 0 else "queued"
        progress_dict[bid] = prog
        # Heartbeat publisher — keeps the master alive-bit fresh during the
        # blocking Engine.generate call inside each chunk. Started here so
        # `set_chunk` / `clear_chunk` calls below land on a live publisher.
        hb = _ProgressHeartbeat(progress_dict, bid, interval_s=5.0)
        hb.start()
        self._inflight.add(bid)
        self._phase = "generating"
        self._phase_started_at = time.time()

        emit(
            "inference.start",
            batch_id=bid,
            n=n_total,
            n_resumed=n_resumed,
            chunk_size=cs,
            ckpt=str(ckpt_dir),
        )

        if n_resumed >= n_total:
            state_path.write_text(json.dumps({
                "status": "done",
                "n_done": n_total,
                "n_total": n_total,
                "updated_at": time.time(),
            }))
            checkpoint_vol.commit()
            prog["status"] = "done"
            prog["completed_at"] = time.time()
            hb.publish_full(prog)
            emit("inference.already_done", batch_id=bid, n=n_total)
            hb.stop()
            self._inflight.discard(bid)
            if not self._inflight:
                self._phase = "ready"
                self._phase_started_at = time.time()
            return existing_results[:n_total]

        try:
            prog["status"] = "generating"
            prog["started_at"] = time.time()
            hb.publish_full(prog)
            cumul_in_session = 0
            cumul_out_session = 0

            safety_margin = 256
            max_input_toks = self.cfg.context_length - safety_margin

            # Total chunk count (informational — heartbeat reports this so
            # the master can show "chunk 3/8").
            n_chunks_total = (n_total - n_resumed + cs - 1) // cs if cs > 0 else 1

            for chunk_idx, chunk_global_start in enumerate(
                range(n_resumed, n_total, cs)
            ):
                chunk_global_end = min(chunk_global_start + cs, n_total)
                chunk_records = records[chunk_global_start:chunk_global_end]
                hb.set_chunk(
                    chunk_index=chunk_idx,
                    chunk_n_chunks=n_chunks_total,
                    chunk_size_actual=chunk_global_end - chunk_global_start,
                    chunk_row_offset=chunk_global_start,
                )

                chunk_outputs: list[dict | None] = [None] * len(chunk_records)
                # (within-chunk-idx, BatchRow, rendered_text, sampling_dict, n_samples)
                generate_plan: list[tuple[int, object, str, dict, int]] = []

                for j, rec in enumerate(chunk_records):
                    try:
                        row = parse_row(rec)
                    except Exception as e:
                        rid = str(rec.get("id", f"row{chunk_global_start + j}"))
                        chunk_outputs[j] = make_error_row(
                            rid, f"parse_error: {e}", model=self.cfg.key
                        )
                        continue

                    try:
                        text = self._render_prompt(row)
                    except Exception as e:
                        chunk_outputs[j] = make_error_row(
                            row.id,
                            f"chat_template_error: {e}",
                            model=self.cfg.key,
                        )
                        continue

                    try:
                        n_input_toks = len(self.tokenizer(text)["input_ids"])
                    except Exception:
                        n_input_toks = 0
                    if n_input_toks > max_input_toks:
                        chunk_outputs[j] = make_error_row(
                            row.id,
                            (
                                f"prompt_too_long: {n_input_toks} > "
                                f"max_input_toks={max_input_toks} "
                                f"(context_length={self.cfg.context_length})"
                            ),
                            model=self.cfg.key,
                        )
                        continue

                    sp = self._build_sampling_params(row)
                    generate_plan.append((j, row, text, sp, max(1, int(row.n))))

                if generate_plan:
                    # Flatten to lists. For n>1 rows we replicate prompt
                    # entries and aggregate the per-replica outputs back.
                    prompts: list[str] = []
                    sps: list[dict] = []
                    # Reverse-map: index in `prompts` → which generate_plan row
                    replica_owner: list[int] = []
                    for plan_idx, (_, _, text, sp, n_samples) in enumerate(generate_plan):
                        for _r in range(n_samples):
                            prompts.append(text)
                            sps.append(sp)
                            replica_owner.append(plan_idx)

                    with phase(
                        "generate.chunk",
                        batch_id=bid,
                        n=len(prompts),
                        chunk_start=chunk_global_start,
                    ):
                        outs = self.llm.generate(prompts, sps)

                    # Group raw outputs by plan-row.
                    grouped: dict[int, list[dict]] = {}
                    for out_idx, out in enumerate(outs or []):
                        owner = replica_owner[out_idx]
                        grouped.setdefault(owner, []).append(out or {})

                    for plan_idx, (j, row, _text, _sp, _n) in enumerate(generate_plan):
                        outs_for_row = grouped.get(plan_idx) or []
                        if not outs_for_row:
                            chunk_outputs[j] = make_error_row(
                                row.id,
                                "sglang_no_output: engine returned None",
                                model=self.cfg.key,
                            )
                            continue
                        # SGLang dict shape:
                        #   {"text": str,
                        #    "meta_info": {
                        #       "prompt_tokens": int,
                        #       "completion_tokens": int,
                        #       "finish_reason": {"type": "stop"|"length", ...} | str,
                        #       ...
                        #    }}
                        first_meta = (outs_for_row[0] or {}).get("meta_info") or {}
                        in_toks = int(first_meta.get("prompt_tokens", 0) or 0)
                        cands: list[dict] = []
                        for o in outs_for_row:
                            meta = o.get("meta_info") or {}
                            ct = int(meta.get("completion_tokens", 0) or 0)
                            cumul_out_session += ct
                            fr = meta.get("finish_reason")
                            if isinstance(fr, dict):
                                fr_str = fr.get("type") or fr.get("matched", "stop")
                            else:
                                fr_str = fr or "stop"
                            cands.append(
                                {
                                    "text": o.get("text", ""),
                                    "finish_reason": fr_str,
                                    "completion_tokens": ct,
                                }
                            )
                        cumul_in_session += in_toks
                        chunk_outputs[j] = make_output_row(
                            row.id, in_toks, cands, model=self.cfg.key
                        )

                with results_path.open("a") as f:
                    for output_row in chunk_outputs:
                        f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                state_path.write_text(json.dumps({
                    "status": "generating",
                    "n_done": chunk_global_end,
                    "n_total": n_total,
                    "updated_at": time.time(),
                }))
                checkpoint_vol.commit()

                prog["n_done"] = chunk_global_end
                last_id = chunk_outputs[-1].get("id") if chunk_outputs else None
                prog["current_row_id"] = last_id
                prog["total_prompt_tokens"] = cumul_in_session
                prog["total_completion_tokens"] = cumul_out_session
                if prog["started_at"]:
                    elapsed = time.time() - prog["started_at"]
                    prog["elapsed_s"] = round(elapsed, 2)
                    if elapsed > 0:
                        prog["tps_decode"] = round(cumul_out_session / elapsed, 1)
                hb.publish_full(prog)
                # Clear "in-chunk" heartbeat fields between chunks so the
                # master can distinguish "between chunks (committing)" from
                # "active in chunk".
                hb.clear_chunk()
                emit(
                    "inference.chunk_done",
                    batch_id=bid,
                    chunk_index=chunk_idx,
                    chunk_n_chunks=n_chunks_total,
                    n_done=chunk_global_end,
                    n_total=n_total,
                    tps_decode=prog["tps_decode"],
                    ckpt="committed",
                )

            state_path.write_text(json.dumps({
                "status": "done",
                "n_done": n_total,
                "n_total": n_total,
                "updated_at": time.time(),
            }))
            checkpoint_vol.commit()
            prog["status"] = "done"
            prog["completed_at"] = time.time()
            hb.publish_full(prog)
            emit(
                "inference.done",
                batch_id=bid,
                n=n_total,
                n_resumed=n_resumed,
                duration_s=round(
                    prog["completed_at"] - (prog["started_at"] or prog["completed_at"]),
                    2,
                ),
                total_prompt_tokens=cumul_in_session,
                total_completion_tokens=cumul_out_session,
                tps_decode=prog["tps_decode"],
            )

            with results_path.open() as f:
                return [json.loads(line) for line in f if line.strip()]

        except BaseException as e:
            err = repr(e)[:500]
            try:
                state_path.write_text(json.dumps({
                    "status": "failed",
                    "n_done": prog.get("n_done", n_resumed),
                    "n_total": n_total,
                    "error": err,
                    "updated_at": time.time(),
                }))
                checkpoint_vol.commit()
            except Exception:
                pass
            prog["status"] = "failed"
            prog["error"] = err
            prog["completed_at"] = time.time()
            hb.publish_full(prog)
            self._last_error = err
            emit("inference.fail", batch_id=bid, error=err, n_done=prog.get("n_done"))
            raise
        finally:
            try:
                hb.stop()
            except Exception:
                pass
            self._inflight.discard(bid)
            if not self._inflight:
                self._phase = "ready"
                self._phase_started_at = time.time()

    @modal.method()
    def get_results(self, batch_id: str) -> list[dict]:
        try:
            checkpoint_vol.reload()
        except Exception:
            pass
        ckpt_dir = self.ckpt_root / batch_id
        results_path = ckpt_dir / "results.jsonl"
        if not results_path.exists():
            return []
        rows: list[dict] = []
        with results_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        return rows

    # ============================================================================
    # Public HTTP API (FastAPI ASGI app)
    # ============================================================================

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, HTTPException

        api = FastAPI(title=f"sglang slave {self.cfg.key}", version="1")

        @api.get("/")
        def _root() -> dict:
            return self._wrap(
                {
                    "slave": self.cfg.key,
                    "hf_id": self.cfg.hf_id,
                    "gpu": self.cfg.gpu,
                    "tp": self.cfg.tp,
                    "context_length": self.cfg.context_length,
                    "default_max_tokens": self.cfg.default_max_tokens,
                    "attention_backend": self.cfg.attention_backend,
                    "engine": "sglang",
                    "routes": [
                        "GET  /ping",
                        "GET  /status",
                        "POST /warmup",
                        "POST /cold_down",
                        "POST /inference",
                        "GET  /progress/{batch_id}",
                        "GET  /result/{batch_id}",
                    ],
                }
            )

        @api.get("/ping")
        def _ping() -> dict:
            return self.ping.local()

        @api.get("/status")
        def _status() -> dict:
            return self.status.local()

        @api.post("/warmup")
        def _warmup() -> dict:
            return self.warmup.local()

        @api.post("/cold_down")
        def _cold_down() -> dict:
            return self.cold_down.local()

        @api.post("/inference")
        def _http_inference(body: dict) -> dict:
            from observability import emit  # noqa: E402

            records = body.get("records")
            if not isinstance(records, list):
                raise HTTPException(
                    status_code=400,
                    detail="body.records must be a list of slave-format rows",
                )
            bid = body.get("batch_id") or str(uuid.uuid4())
            cs = body.get("chunk_size")
            if cs is not None:
                try:
                    cs = int(cs)
                except (TypeError, ValueError) as e:
                    raise HTTPException(400, f"chunk_size must be int: {e}")

            fc = self.inference.spawn(records=records, batch_id=bid, chunk_size=cs)
            fc_dict[bid] = fc.object_id
            emit(
                "http.inference_spawned",
                batch_id=bid,
                fc_object_id=fc.object_id,
                n=len(records),
            )
            return self._wrap(
                {
                    "ok": True,
                    "batch_id": bid,
                    "fc_object_id": fc.object_id,
                    "n_records": len(records),
                    "chunk_size": cs or self.cfg.max_running_requests,
                    "track_url": f"/progress/{bid}",
                    "result_url": f"/result/{bid}",
                }
            )

        @api.get("/progress/{batch_id}")
        def _http_progress(batch_id: str) -> dict:
            return self.progress.local(batch_id)

        @api.get("/result/{batch_id}")
        def _http_result(batch_id: str) -> dict:
            results = self.get_results.local(batch_id)
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail=f"no checkpointed results for batch_id {batch_id!r}",
                )
            return self._wrap(
                {
                    "ok": True,
                    "batch_id": batch_id,
                    "results": results,
                    "n": len(results),
                }
            )

        return api


# ============================================================================
# Local entrypoints (CLI)
# ============================================================================


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"batch file not found: {path}")
    records: list[dict] = []
    with path.open() as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln} invalid json: {e}") from e
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@app.local_entrypoint()
def ping() -> None:
    print(f"[client] pinging {APP_NAME} ...")
    info = SGLangSlave().ping.remote()
    print(f"[client] {info}")


@app.local_entrypoint()
def warmup() -> None:
    print(f"[client] warming up {APP_NAME} ...")
    info = SGLangSlave().warmup.remote()
    print(f"[client] {info}")


@app.local_entrypoint()
def cold_down() -> None:
    info = SGLangSlave().cold_down.remote()
    print(f"[client] {info}")


@app.local_entrypoint()
def status() -> None:
    info = SGLangSlave().status.remote()
    print(json.dumps(info, indent=2, default=str))


@app.local_entrypoint()
def submit(batch: str, batch_id: str = "", chunk_size: int = 0) -> None:
    bid = batch_id or str(uuid.uuid4())
    cs = chunk_size if chunk_size > 0 else None
    records = _read_jsonl(Path(batch))
    print(
        f"[client] {APP_NAME}: submitting {len(records)} rows "
        f"(batch_id={bid}, chunk_size={cs or 'default'})"
    )
    fc = SGLangSlave().inference.spawn(
        records=records, batch_id=bid, chunk_size=cs
    )
    fc_dict[bid] = fc.object_id
    print(f"[client] batch_id  = {bid}")
    print(f"[client] fc_object_id = {fc.object_id}")
    print(
        f"[client] track with: MODEL_KEY={_MODEL_KEY} "
        f"modal run inference-farm/slave/app.py::watch --batch-id {bid}"
    )


@app.local_entrypoint()
def watch(batch_id: str) -> None:
    rec = SGLangSlave().progress.remote(batch_id)
    print(json.dumps(rec, indent=2, default=str))


@app.local_entrypoint()
def fetch(batch_id: str, output: str) -> None:
    out_path = Path(output)
    try:
        fc_object_id = fc_dict[batch_id]
    except KeyError as e:
        raise SystemExit(
            f"no FunctionCall id stored for batch_id={batch_id!r}; was it "
            "submitted via `submit` from this slave?"
        ) from e

    print(f"[client] retrieving FunctionCall {fc_object_id} ...")
    fc = modal.FunctionCall.from_id(fc_object_id)
    results: list[dict] = fc.get()
    _write_jsonl(out_path, results)
    n_err = sum(1 for r in results if r.get("error"))
    n_ok = len(results) - n_err
    print(f"[client] wrote {len(results)} rows to {out_path}  (ok={n_ok}, err={n_err})")


@app.local_entrypoint()
def run_batch(batch: str, output: str = "slave_output.jsonl") -> None:
    out_path = Path(output)
    records = _read_jsonl(Path(batch))
    print(
        f"[client] {APP_NAME}: loaded {len(records)} rows from {batch}, "
        "running synchronously"
    )
    bid = str(uuid.uuid4())
    results = SGLangSlave().inference.remote(records=records, batch_id=bid)
    _write_jsonl(out_path, results)
    n_err = sum(1 for r in results if r.get("error"))
    n_ok = len(results) - n_err
    print(
        f"[client] wrote {len(results)} rows to {out_path}  "
        f"(ok={n_ok}, err={n_err})  batch_id={bid}"
    )
