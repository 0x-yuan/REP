"""summary_32b - Qwen2.5-7B summary of the trace + answer

Downstream-distillation config for the EMNLP paper "Hidden Thoughts Are Not
Secret". One paper table row = the canonical engine driven by this file.

Recipe:   s1/train/sft.py full-parameter FSDP->DDP
Epochs:   6  (save_strategy=epoch; Reported cell = best epoch of 6 by Delta-sum. Control: summary.)
Teacher:  Qwen3-32B (ideal), trace summarized
Source:   engine template + a dataset built by ../data_builders/ (see SOURCE_REPO).

Run:  DISTILL_CONFIG=configs/summary_32b.py  (loaded by ../engine/*.py scripts)
"""
from __future__ import annotations

# --- Which engine drives this row (doc marker; scripts live in that folder) --
ENGINE = "engine"

# --- Modal account (scrub: fill in your own B200-quota workspace) ------------
PROFILE = "<your-modal-profile>"

# --- Run identity ------------------------------------------------------------
RUN_ID = "summary-q3_32b-v1"
MODEL_FAMILY = "qwen25"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- Teacher dataset (the per-row knob) --------------------------------------
SOURCE_REPO = "your-org/summary_32b"  # build with ../data_builders/, push, then point here
DATASET_SUBDIR = "sum_q3_32b_qwen25_7b"

# --- Modal volumes + apps ----------------------------------------------------
CKPTS_VOL = "summary-q3_32b-ckpts"
RESULTS_VOL = "summary-q3_32b-results"
HF_CACHE_VOL = "summary-q3_32b-hfcache"
TRAIN_APP = "summary-q3_32b-train"
EVAL_APP = "summary-q3_32b-eval-multi"
EVAL_APP_SUFFIXES = ['', '-r1', '-r2', '-r3', '-r4', '-r5', '-r6']          # length == EPOCHS + 1 (base + per-epoch)
EVAL_APP_POOL = [EVAL_APP + s for s in EVAL_APP_SUFFIXES]

# --- Training recipe ---------------------------------------------------------
EPOCHS = 6
SAVE_STRATEGY = "epoch"
BLOCK_SIZE = 32768
GRAD_ACCUM = 4
MICRO_BATCH = 1
LEARNING_RATE = 1e-5
GPU_TYPE = "B200"
GPU_COUNT = 4
USE_FSDP = False

# --- Eval suite (frozen; n-sampled averaging protocol) -----------------------------
EVAL_BENCH_SUBSET = ["aime24", "aime25", "math500", "jeebench", "lcb_v5"]
JEE_MATH_ONLY = True
JEE_N_SAMPLES = 6
AIME_N_SAMPLES = 3
MATH500_N_SAMPLES = 3

# --- HF upload target (optional; "" skips upload) ----------------------------
HF_REPO = ""
HF_VARIANT_NOTE = "qwen3_32b teacher / variant=summary_answer"
