"""rep_q3_14b_to_qwen25_14b - Qwen3-14B REP traces -> Qwen2.5-14B-Instruct student

Downstream-distillation config for the EMNLP paper "Hidden Thoughts Are Not
Secret". One supplementary table row = the canonical engine driven by this file.

Recipe:   s1/train/sft.py full-parameter FSDP full_shard (8xB200, adamw_torch_fused)
Epochs:   5  (save_strategy=epoch; Reported cell = best epoch of 5 by Delta-sum.)
Teacher:  Qwen3-14B (REP-exposed, clean {problem, solution})
Student:  Qwen/Qwen2.5-14B-Instruct (FSDP, 8xB200)
Source:   the 10k pure-Qwen3-14B REP set (see SOURCE_REPO).
Supplementary row: 14B student, 100% Qwen3-14B REP traces. MATH500 0.813 (n=3, T=0.5).

Run:  DISTILL_CONFIG=configs/rep_q3_14b_to_qwen25_14b.py  (loaded by ../engine/*.py scripts)
"""
from __future__ import annotations

# --- Which engine drives this row (doc marker; scripts live in that folder) --
ENGINE = "engine"

# --- Modal account (scrub: fill in your own B200-quota workspace) ------------
PROFILE = "<your-modal-profile>"

# --- Run identity ------------------------------------------------------------
RUN_ID = "rep-q3pure-qwen25-14b-v1"
MODEL_FAMILY = "qwen25"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# --- Teacher dataset (the per-row knob) --------------------------------------
# Build it: flatten `Chia-Mu-Lab/REP-datasets` config `distill_q3_14b_clean`
# to `{problem: question, solution: r2 + \\boxed{answer}}` (no <think> block,
# no V3 wrapper) and push it to your own Hub repo. The engine's dataset_prep
# auto-detects the {problem, solution} schema: problem -> user, solution ->
# assistant, no strip / no correctness filter, drop-if-too-long.
SOURCE_REPO = "your-org/rep-q3_14b-problem-solution"
DATASET_SUBDIR = "rep_q3pure_qwen25_14b_text"

# --- Modal volumes + apps ----------------------------------------------------
CKPTS_VOL = "rep-q3pure-14b-ckpts"
RESULTS_VOL = "rep-q3pure-14b-results"
HF_CACHE_VOL = "rep-q3pure-14b-hfcache"
TRAIN_APP = "rep-q3pure-14b-train"
EVAL_APP = "rep-q3pure-14b-eval-multi"
EVAL_APP_SUFFIXES = ['', '-r1', '-r2', '-r3', '-r4', '-r5']          # length == EPOCHS + 1 (base + per-epoch)
EVAL_APP_POOL = [EVAL_APP + s for s in EVAL_APP_SUFFIXES]

# --- Training recipe ---------------------------------------------------------
# Same frozen s1 recipe as the 7B rows; only the sharding knobs change for the
# larger student: FSDP full_shard on 8xB200 with GRAD_ACCUM=2 so the effective
# batch stays 8 * 2 * 1 = 16 (LR 1e-5 is tuned for eff-batch 16). The engine
# passes --optim=adamw_torch_fused (torch-2.7.1 FSDP epoch-save workaround).
EPOCHS = 5
SAVE_STRATEGY = "epoch"
BLOCK_SIZE = 32768
GRAD_ACCUM = 2
MICRO_BATCH = 1
LEARNING_RATE = 1e-5
GPU_TYPE = "B200"
GPU_COUNT = 8
USE_FSDP = True

# --- Eval suite (frozen; n-sampled averaging protocol) -----------------------------
EVAL_BENCH_SUBSET = ["aime24", "aime25", "math500", "jeebench", "lcb_v5"]
JEE_MATH_ONLY = True
JEE_N_SAMPLES = 6
AIME_N_SAMPLES = 3
MATH500_N_SAMPLES = 3

# --- HF upload target (optional; "" skips upload) ----------------------------
HF_REPO = ""
HF_VARIANT_NOTE = "Student = Qwen2.5-14B-Instruct. 10k Qwen3-14B REP traces (distill_q3_14b_clean), clean {problem, solution}; solution is the plain SFT target (reasoning + boxed answer). FSDP full_shard."
