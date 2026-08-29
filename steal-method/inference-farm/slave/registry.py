"""Per-model SGLang config registry — one entry per model, batch-tuned for
**stability + throughput** on H200 with YaRN-extended 131k context.

Targeted workload:
    * single-turn prompts with very long shared prefix (~100K tokens of
      in-context-learning exemplars; final ~1K tokens differ per row)
    * decode-heavy: max_new_tokens up to 20K
    * RadixAttention amortizes the prefill across the batch — only the
      first row of a fresh container pays the full 100K prefill cost.

Hard requirements that every entry honours:
    * `context_length=131072` (YaRN factor 4.0 from native 32768)
    * `default_max_tokens=8192`  (per-row floor; rows can override)
    * `attention_backend="fa3"`  (FA3 wins long-context decode on H200;
                                  FA4 regresses ~49% past 16k tokens)
    * RadixAttention prefix cache always on (SGLang's main moat)
    * `schedule_policy="lpm"`    (longest-prefix-match — required for
                                  shared-prefix workloads per SGLang docs)
    * `page_size=1`              (byte-identical prefix-cache match)

# Why SGLang
    SGLang ≥ 0.5.10 ships static YaRN + Hopper FA3 + RadixAttention all in
    one image. RadixAttention compounds throughput on shared prefixes; for
    unique-prompt workloads expect ≈ vLLM ±20%; for shared-prefix RAG, up
    to several×.

# KV-budget intuition (H200, 141 GB HBM)
    KV_per_token = 2 × layers × kv_heads × head_dim × 2  (bytes, bf16)
    KV_budget    = HBM × mem_fraction_static − weights − overhead
    full_len_seqs = KV_budget / context_length

    SGLang uses `mem_fraction_static` for the same role as vLLM's
    `gpu_memory_utilization`. We pin to 0.80 across the board for
    long-running batch jobs — pushing higher (0.85+) hits a known
    KV-cache leak / retract-deadlock class on Qwen3 + reasoning workloads
    (Issue #6778, #15840).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — slave runs on Modal images with Py 3.11+
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    gpu: str
    tp: int

    # Context / KV
    context_length: int                  # SGLang's --context-length (analog of max_model_len)
    mem_fraction_static: float           # 0.0..1.0, analog of vLLM gpu_memory_utilization

    # Throughput knobs (0 = sentinel "let SGLang decide")
    max_running_requests: int = 0        # max concurrent in-flight (analog of max_num_seqs)
    chunked_prefill_size: int = 0        # SGLang default ≈ 8192 on ≥80GB
    cuda_graph_max_bs: int = 0           # cap CUDA-graph capture batch size

    # Output budget
    default_max_tokens: int = 8192

    # Backends
    attention_backend: str = "fa3"       # fa3 / fa4 / triton / flashinfer / torch_native
    kv_cache_dtype: str = "auto"         # auto / fp8_e5m2 / fp8_e4m3 — fp8 doubles KV
                                         # but degrades reasoning quality (Issue #6778);
                                         # KEEP "auto" (bf16) for this workload
    disable_radix_cache: bool = False    # leave RadixAttention ON (the SGLang moat)
    disable_cuda_graph: bool = False     # set True only if a model crashes during graph capture
    enable_torch_compile: bool = False   # off; long startup, marginal win, CUDA-graph conflicts

    # Scheduling (SGLang ≥ 0.5)
    schedule_policy: str = "lpm"         # longest-prefix-match: REQUIRED for shared-prefix
                                         # workloads per docs.sglang.io/.../hyperparameter_tuning
    schedule_conservativeness: float = 1.0  # raise to 1.3 only on retract warnings
    page_size: int = 1                   # 1 = max prefix-cache reuse on FA3
    watchdog_timeout: int = 1800         # 30 min, covers cold-start + slow chunks

    # Stability flags — keep all OFF for this workload
    disable_overlap_schedule: bool = False  # default; flip if alloc-fail past hour 6
    enable_dp_attention: bool = False       # MoE-only path; do NOT enable on dense Qwen3
    enable_two_batch_overlap: bool = False  # multi-node DeepEP only; regresses on single node
    enable_mixed_chunk: bool = False        # nothing to mix when prefill is radix-cached

    # CUDA-graph bucket list. None = SGLang infers (~58 buckets on long-ctx
    # Qwen3, costing ~12 min cold-start). An explicit list dramatically
    # shrinks cold-start by only capturing the buckets we'll actually hit.
    cuda_graph_bs: list[int] | None = None

    # MoE (kept for future MoE entries; defaults to 1 = TP-only routing)
    ep_size: int = 1

    # Reasoning parser (auto-extracts <think>...</think>; offline path applies
    # template manually so this is informational unless you switch to serve mode)
    reasoning_parser: str = "qwen3"

    # Long-context (YaRN). Set rope_yarn_factor=None to keep native ctx.
    rope_yarn_factor: float | None = None
    rope_orig_max: int = 32768
    # Qwen3 ships rope_theta=1000000 across every published size. SGLang
    # 0.5.10.post1 reads ``config.rope_parameters["rope_theta"]`` inside
    # qwen3.py and KeyErrors if our override didn't carry it through. We
    # include it explicitly in json_model_override_args so the override
    # path doesn't drop it.
    rope_theta: float = 1_000_000.0

    # Override the default sglang docker image tag.
    # When None, the slave uses ``_DEFAULT_SGLANG_IMAGE`` from app.py.
    sglang_image: str | None = None

    # Modal lifecycle
    concurrent_max_inputs: int = 1       # 1 = single in-flight inference per container.
                                         # Multiple in-flight inference() calls on the same
                                         # SGLang Engine is the shared cause of leak class
                                         # in Issues #6778, #14972, #15840 — keep at 1.
    timeout_s: int = 12 * 3600
    scaledown_s: int = 600
    startup_timeout_s: int = 30 * 60     # cold start = weight load + cuda-graph capture


_REGISTRY: dict[str, ModelConfig] = {
    # ============================================================
    # Production recipe: YaRN-131k for ALL Qwen3 sizes.
    # Image pin: lmsysorg/sglang:v0.5.10.post1-cu130-runtime
    # (set in app.py as _DEFAULT_SGLANG_IMAGE; per-entry override via
    #  sglang_image only if a model needs a different tag).
    # ============================================================

    # 28 layers × 8 kv_heads × 128 head_dim → KV ≈ 28 KB/token; weights ~3.4 GB
    "qwen3-1p7b": ModelConfig(
        key="qwen3-1p7b",
        hf_id="Qwen/Qwen3-1.7B",
        gpu="H200",
        tp=1,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=64,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=64,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32, 48, 64],
        rope_yarn_factor=4.0,
        concurrent_max_inputs=1,
    ),

    # 36 × 8 × 128 → KV ≈ 144 KB/token; weights ~8 GB
    # 4B fits on a single H200 with room to spare; TP=1 saves the NCCL tax.
    "qwen3-4b": ModelConfig(
        key="qwen3-4b",
        hf_id="Qwen/Qwen3-4B",
        gpu="H200",
        tp=1,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=64,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=64,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32, 48, 64],
        rope_yarn_factor=4.0,
        concurrent_max_inputs=1,
    ),

    # 36 × 8 × 128 → KV ≈ 144 KB/token; weights ~16 GB
    # 8B at TP=1: weights 16 GB; KV budget ≈ 113 GB → ~785K KV-tok → ~6 full
    # 131K seqs. concurrent_max_inputs=1 means we never need more than 1
    # in-flight inference call per container, so KV headroom is plenty.
    # Switched from H200×2 TP=2: saves 1 GPU/replica, doubles fanout per
    # Modal account, drops the 8% NVLink all-reduce tax.
    "qwen3-8b": ModelConfig(
        key="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        gpu="H200",
        tp=1,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=64,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=64,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32, 48, 64],
        rope_yarn_factor=4.0,
        concurrent_max_inputs=1,
    ),

    # 40 × 8 × 128 → KV ≈ 160 KB/token; weights ~28 GB
    # 14B at TP=1: weights 28 GB; KV budget ≈ 92 GB → ~575K KV-tok → ~4.4
    # full 131K seqs. Switched from H200×2 TP=2 for the same reasons as 8B.
    "qwen3-14b": ModelConfig(
        key="qwen3-14b",
        hf_id="Qwen/Qwen3-14B",
        gpu="H200",
        tp=1,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=48,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=48,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32, 48],
        rope_yarn_factor=4.0,
        concurrent_max_inputs=1,
    ),

    # 64 × 8 × 128 → KV ≈ 256 KB/token; weights ~64 GB
    # TP=4 on H200×4. Qwen3-32B has num_kv_heads=8, so legal TP ∈ {1,2,4,8}.
    # TP=4 is the sweet spot per community benchmarks: TP=8 puts kv_heads=1
    # per GPU and decode regresses ~25% vs TP=4.
    # Aggregate ~564 GB HBM at TP=4, mem=0.80 → ~451 GB available; weights
    # split 16 GB/GPU leaves ~435 GB for KV → ~1.7M token KV budget →
    # ~13 full-len 131K seqs.
    "qwen3-32b": ModelConfig(
        key="qwen3-32b",
        hf_id="Qwen/Qwen3-32B",
        gpu="H200:4",
        tp=4,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=32,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=32,
        # Tighter bucket list: cold-start was ~12 min on the 58-bucket
        # default; capping to 6 buckets cuts capture cost by ~10×.
        cuda_graph_bs=[1, 2, 4, 8, 16, 32],
        rope_yarn_factor=4.0,
        concurrent_max_inputs=1,
        timeout_s=12 * 3600,
    ),

    # ============================================================
    # Cross-model victims
    # ============================================================

    # Gemma 3 12B-it (dense). Architecture: 48 layers, num_attention_heads=16,
    # num_kv_heads=8, head_dim=256. KV is dominated by sliding-window layers
    # (5:1 SWA:full ratio in Gemma 3) — effective KV/token is much smaller
    # than the naive 48*8*256 calc. Weights ~24 GB bf16. Native ctx 128k
    # via local+global attention; no YaRN needed.
    # Workload: OpenThoughts-500 — input ≤ 44k (V3 attack), output cap 80k →
    # need context_length ≥ 124k. Use 131072 (native max).

    # Qwen3.6-27B (released 2026; dense hybrid). Architecture: 64 layers in a
    # 4-block pattern of 3×GatedDeltaNet + 1×GatedAttention. Only the 16
    # gated-attention layers store traditional KV (Q=24, KV=4, head_dim=256);
    # the 48 DeltaNet layers carry recurrent state, no KV cache. Effective
    # KV_per_token = 2 × 16 × 4 × 256 × 2 = 65,536 B ≈ 64 KB/token — roughly
    # 1/4 of Qwen3-32B per token despite ~85% of the params.
    # Weights ~54 GB bf16 → fits H200×1 (141 GB) at mem_fraction=0.80.
    # KV budget at 131k ctx: ~8 full-length seqs.
    # Native context 262144 → no YaRN; we cap at 131072 to match the rest
    # of the farm (OT500 V3 attack tops out well under 50k input + 32k output).
    # `enable_thinking` kwarg in tokenizer.apply_chat_template is supported
    # (Qwen3-compatible). reasoning_parser="qwen3" matches the <think>…</think>
    # emit format. Requires SGLang ≥ 0.5.10 (already baked in default image).
    "qwen3p6-27b": ModelConfig(
        key="qwen3p6-27b",
        hf_id="Qwen/Qwen3.6-27B",
        gpu="H200",
        tp=1,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=16,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=16,
        cuda_graph_bs=[1, 2, 4, 8, 16],
        rope_yarn_factor=None,      # native 262k; no YaRN needed
        rope_orig_max=131072,
        default_max_tokens=32768,   # reasoning model — long outputs expected
        concurrent_max_inputs=1,
        reasoning_parser="qwen3",
    ),

    # OpenAI gpt-oss-20b (sparse MoE: 20B total / 3.6B active). Architecture:
    # 24 layers, num_attention_heads=64, num_kv_heads=8, head_dim=64 → KV
    # tiny per token. Weights ship in MXFP4 quant (~13 GB on disk); SGLang
    # handles the on-the-fly dequant and Harmony chat template.
    # Native 128k ctx; no YaRN needed. Set reasoning_parser="gpt-oss" so the
    # serve-mode auto-extracts the `analysis` channel — informational only
    # for our offline path (we parse ourselves at scoring time).
    "gpt-oss-20b": ModelConfig(
        key="gpt-oss-20b",
        hf_id="openai/gpt-oss-20b",
        gpu="H200",
        tp=1,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=32,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=32,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32],
        rope_yarn_factor=None,
        rope_orig_max=131072,
        concurrent_max_inputs=1,
        reasoning_parser="gpt-oss",
    ),

    # Cross-model victim: Google Gemma-4-31B-it (dense, released 2026-04-02).
    # Architecture: 30.7B params, native ctx 256k; reasoning channels via
    # <|channel>thought ... <channel|>. SGLang reasoning_parser="gemma4"
    # (PR #21952). May require an updated image; the default
    # lmsysorg/sglang:v0.5.10.post1-cu130-runtime might not include Gemma-4
    # arch — if boot fails on weight load with "unknown architecture",
    # override sglang_image in experiment.toml to a newer tag.
    "gemma-4-31b": ModelConfig(
        key="gemma-4-31b",
        hf_id="google/gemma-4-31B-it",
        gpu="B200",
        tp=1,
        context_length=65536,
        mem_fraction_static=0.80,
        max_running_requests=16,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=16,
        cuda_graph_bs=[1, 2, 4, 8, 16],
        rope_yarn_factor=None,   # native 256k
        rope_orig_max=131072,
        default_max_tokens=20000,
        concurrent_max_inputs=1,
        reasoning_parser="gemma4",
        attention_backend="flashinfer",
        timeout_s=12 * 3600,
    ),

    # Trace summarizer: Qwen2.5-7B-Instruct (added 2026-05-24).
    # Used to compress long ideal reasoning traces into distillation-quality
    # summaries (no <think> channel; clean ChatML output).
    # Architecture: 28 layers, num_attention_heads=28, num_kv_heads=4 (GQA),
    # head_dim=128 → KV ≈ 2 × 28 × 4 × 128 × 2 = 56 KB/token. Weights ~15 GB
    # bf16. Native max_position_embeddings=32768; no YaRN — source r1 traces
    # are p99 ≈ 17K tokens and output cap is 8K, so 32K ctx covers everything.
    # B200 TP=1: weights ~15 GB; mem_fraction_static=0.80 × 192 GB − 15 GB ≈
    # 138 GB for KV → ~75 full-length 32K seqs. Plenty for max_running_requests=64.
    # attention_backend=flashinfer per feedback_sglang_b200_attention_backend.md
    # (fa3 crash-loops on B200 SM 100).
    "qwen2p5-7b-instruct": ModelConfig(
        key="qwen2p5-7b-instruct",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        gpu="B200",
        tp=1,
        context_length=32768,
        mem_fraction_static=0.80,
        max_running_requests=64,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=64,
        cuda_graph_bs=[1, 2, 4, 8, 16, 32, 48, 64],
        rope_yarn_factor=None,    # native 32768 is enough; no YaRN needed
        rope_orig_max=32768,
        default_max_tokens=8192,
        concurrent_max_inputs=1,
        attention_backend="flashinfer",
        timeout_s=12 * 3600,
    ),

    # Cross-model transferability target (added 2026-05-21)
    # Qwen3-235B-A22B MoE (235B total / 22B active). Architecture per HF card:
    # 94 layers, num_attention_heads=64, num_kv_heads=4, head_dim=128 → KV
    # ≈ 2 × 4 × 128 × 2 = 2,048 B/token/layer; per 94 layers ≈ 192 KB/token.
    # Weights ~470 GB bf16 → 4×B200 (180GB each = 720GB aggregate) with
    # mem_fraction_static=0.80 leaves ~106 GB shared for KV — comfortable.
    # Native ctx 32768 → enable YaRN to reach 131k like the rest of the
    # Qwen3 family. reasoning_parser="qwen3" — same <think>...</think> emit.
    # GPU set to B200:4 here; if no B200 quota, override via experiment.toml
    # to gpu="H200:8" tp=8 (last-resort; H200:8 may not be available either).
    "qwen3-235b-a22b": ModelConfig(
        key="qwen3-235b-a22b",
        hf_id="Qwen/Qwen3-235B-A22B",
        gpu="B200:4",
        tp=4,
        context_length=131072,
        mem_fraction_static=0.80,
        max_running_requests=16,
        chunked_prefill_size=16384,
        cuda_graph_max_bs=16,
        cuda_graph_bs=[1, 2, 4, 8, 16],
        rope_yarn_factor=4.0,
        rope_orig_max=32768,
        concurrent_max_inputs=1,
        reasoning_parser="qwen3",
        default_max_tokens=20000,
        attention_backend="flashinfer",  # B200 SM 100: fa3 crash-loops; flashinfer is the supported path
        timeout_s=12 * 3600,
    ),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


# ============================================================================
# Per-experiment overrides — read from `<server-root>/experiment.toml`.
#
# When duplicating this repo as an experiment template, prefer editing
# experiment.toml over patching this file: the registry stays the same
# stable baseline, and per-experiment tweaks live in one obvious place.
#
# Example experiment.toml:
#     [models.qwen3-8b]
#     default_max_tokens = 4096
#     context_length = 65536
#     max_running_requests = 32
#
#     [models.qwen3-14b]
#     gpu = "H200:2"
#     tp = 2
# ============================================================================

_OVERRIDES_FILENAME = "experiment.toml"


def _server_root() -> Path:
    """Return the inference-farm root (parent of slave/, master/)."""
    return Path(__file__).resolve().parent.parent


def _load_overrides_from_path(toml_path: Path) -> dict[str, dict]:
    if not toml_path.exists():
        return {}
    try:
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise RuntimeError(f"failed to parse {toml_path}: {e}") from e
    models = data.get("models", {})
    if not isinstance(models, dict):
        return {}
    out: dict[str, dict] = {}
    for key, overrides in models.items():
        if not isinstance(overrides, dict):
            raise RuntimeError(
                f"{toml_path}: [models.{key}] must be a table"
            )
        out[key] = dict(overrides)
    return out


def load_overrides() -> dict[str, dict]:
    """Public entry — read overrides.

    Resolution order:
    1. `EXPERIMENT_OVERRIDES_JSON` env var (set by the slave's container env;
       lets the container see the same overrides the host loaded at deploy
       time without shipping the TOML file).
    2. `<server-root>/experiment.toml` (host path).
    3. Empty dict.
    """
    env_json = os.environ.get("EXPERIMENT_OVERRIDES_JSON", "").strip()
    if env_json:
        try:
            data = json.loads(env_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"EXPERIMENT_OVERRIDES_JSON is not valid JSON: {e}"
            ) from e
        if not isinstance(data, dict):
            raise RuntimeError(
                "EXPERIMENT_OVERRIDES_JSON must decode to a JSON object"
            )
        return {k: dict(v) for k, v in data.items() if isinstance(v, dict)}
    return _load_overrides_from_path(_server_root() / _OVERRIDES_FILENAME)


def _apply_overrides(base: ModelConfig, overrides: dict) -> ModelConfig:
    if not overrides:
        return base
    valid = {f.name for f in fields(base)}
    extra = set(overrides) - valid
    if extra:
        raise ValueError(
            f"experiment.toml [models.{base.key}] has unknown fields: "
            f"{sorted(extra)}. Valid fields: {sorted(valid)}"
        )
    return replace(base, **overrides)


def get_config(model_name: str) -> ModelConfig:
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Unknown model_name {model_name!r}. "
            f"Known: {', '.join(list_models())}"
        )
    base = _REGISTRY[model_name]
    overrides = load_overrides().get(model_name, {})
    return _apply_overrides(base, overrides)
