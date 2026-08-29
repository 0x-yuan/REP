"""rep_14b_orig - full <think> trace (r1) + answer

Downstream-distillation config for the EMNLP paper "Hidden Thoughts Are Not
Secret". One paper table row = the canonical engine driven by this file.

Recipe:   s1/train/sft.py full-parameter FSDP->DDP
Epochs:   5  (save_strategy=epoch; Reported cell = best epoch of 5 by Delta-sum. Original (V3-wrapped) teacher traces; the engine dataset_prep strips the attack wrapper.)
Teacher:  Qwen3-14B (original harvest; wrapper stripped at prep time)
Source:   training recipe + the published 14B original teacher dataset (see
          SOURCE_REPO). Parameters mirror the 32B-original counterpart, differing
          only in SOURCE_REPO / RUN_ID / DATASET_SUBDIR.

Run:  DISTILL_CONFIG=configs/rep_14b_orig.py  (loaded by ../engine/*.py scripts)
"""
from __future__ import annotations

# --- Which engine drives this row (doc marker; scripts live in that folder) --
ENGINE = "engine"

# --- Modal account (scrub: fill in your own B200-quota workspace) ------------
PROFILE = "<your-modal-profile>"

# --- Run identity ------------------------------------------------------------
RUN_ID = "rep-distill-14borig-v1"
MODEL_FAMILY = "qwen25"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- Teacher dataset (the per-row knob) --------------------------------------
SOURCE_REPO = "Chia-Mu-Lab/REP-datasets"
SOURCE_CONFIG = "distill_q3_14b_original"
DATASET_SUBDIR = "orig10k_qwen14b_qwen25_7b_text"

# --- Modal volumes + apps ----------------------------------------------------
CKPTS_VOL = "rep-distill-14borig-ckpts"
RESULTS_VOL = "rep-distill-14borig-results"
HF_CACHE_VOL = "rep-distill-14borig-hfcache"
TRAIN_APP = "rep-distill-14borig-train"
EVAL_APP = "rep-distill-14borig-eval-multi"
EVAL_APP_SUFFIXES = ['', '-r1', '-r2', '-r3', '-r4', '-r5']          # length == EPOCHS + 1 (base + per-epoch)
EVAL_APP_POOL = [EVAL_APP + s for s in EVAL_APP_SUFFIXES]

# --- Training recipe ---------------------------------------------------------
EPOCHS = 5
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
HF_VARIANT_NOTE = "full <think> trace (r1) + answer"
