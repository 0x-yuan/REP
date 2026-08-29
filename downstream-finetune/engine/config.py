"""Distillation engine config: single source of truth for ONE distill run.

This is the ONLY file you should need to edit when starting a new
experiment. The training recipe, eval suite, and engine code in
`_common/` and `s1/` stay frozen across experiments — by design.

Usage:
    1. cp -r engine my-new-run
    2. Edit THIS file (config.py) — set DATASET_REPO, RUN_ID, PROFILE,
       and the matching volume / app names.
    3. Follow README.md from there (deploy → spawn → orchestrate).

Multi-variant tip:
    Scripts read `DISTILL_CONFIG=path/to/other_config.py` from the env
    if you want to keep several configs in one folder. The default is
    `config.py` next to the script.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# 1. Modal account / workspace
# -----------------------------------------------------------------------------
# B200 quota is required for both training (4×B200 for 7B, 8×B200 for 14B/32B)
# AND eval (B200:1). Use
# `modal profile current` to confirm the active profile matches PROFILE.
PROFILE = "your-modal-profile"

# -----------------------------------------------------------------------------
# 2. Run identity
# -----------------------------------------------------------------------------
# RUN_ID controls the on-volume path: /ckpts/<RUN_ID>/ + /results/<RUN_ID>/.
# Bump the trailing version (-v1 → -v2) for a clean re-run on the same dataset.
RUN_ID = "distill-default-v1"

# -----------------------------------------------------------------------------
# 3. Student model (FROZEN — do not change per-experiment)
# -----------------------------------------------------------------------------
MODEL_FAMILY = "qwen25"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# -----------------------------------------------------------------------------
# 4. Teacher dataset (★ this is the main per-experiment change)
# -----------------------------------------------------------------------------
# Two schemas are auto-detected by `_common/dataset_prep.py` (column presence):
#   * raw-harvest-style teacher dump: question, completion, structural, victim, ...
#     -> the V3-attack wrapper is stripped at prep time and the
#        structural=True + has-\\boxed{} filter is applied.
#   * Clean {problem, solution}: problem -> user turn, solution -> assistant
#     turn (already reasoning + \\boxed{}); no strip, no correctness filter,
#     rows longer than BLOCK_SIZE are dropped.
# Any other schema means editing dataset_prep.py — that's a CHANGE TO THE
# ENGINE, not a per-experiment knob.
SOURCE_REPO = "<your-hf-org>/<your-dataset>"

# Subdir under `/ckpts/datasets/` where the prepared HF dataset is cached.
# Make this UNIQUE per (dataset, tokenizer) combo so prep doesn't reuse stale
# cache when you swap datasets.
DATASET_SUBDIR = "<your-dataset-subdir>"

# -----------------------------------------------------------------------------
# 5. Modal volume + app names
# -----------------------------------------------------------------------------
# Convention: prefix with RUN_ID's family (here `distill-default-`). Keeping
# names symmetric makes the cleanup script trivial.
CKPTS_VOL = "distill-default-ckpts"
RESULTS_VOL = "distill-default-results"
HF_CACHE_VOL = "distill-hfcache"

TRAIN_APP = "distill-default-train"
EVAL_APP = "distill-default-eval-multi"

# Pool of eval apps (one per ckpt + base) so per-ckpt evals run in their own
# Modal App / task queue / GPU and can be canceled individually. The base
# EVAL_APP (no suffix) handles the base model; -r1..-rN handle epoch ckpts.
# Length should be `EPOCHS + 1`.
EVAL_APP_SUFFIXES = ["", "-r1", "-r2", "-r3", "-r4", "-r5", "-r6"]
EVAL_APP_POOL = [EVAL_APP + s for s in EVAL_APP_SUFFIXES]

# -----------------------------------------------------------------------------
# 6. Training recipe (FROZEN by default; edit only if the recipe is what
#    you're actually trying to study)
# -----------------------------------------------------------------------------
EPOCHS = 6
SAVE_STRATEGY = "epoch"        # one ckpt at the end of every epoch
BLOCK_SIZE = 32768
GRAD_ACCUM = 4
MICRO_BATCH = 1                # micro × grad_accum × GPUs = effective batch
LEARNING_RATE = 1e-5           # cosine schedule, warmup_ratio=0.05
GPU_COUNT = 4
GPU_TYPE = "B200"              # Blackwell SM 100; needs CUDA 12.8 wheels

# Student scale decides the sharding mode:
#   * 7B student  -> USE_FSDP=False, GPU_COUNT=4, GRAD_ACCUM=4 (plain DDP; a 7B
#     model + AdamW state fits one 192GB B200 at ~70-100GB per rank).
#   * 14B / 32B student -> USE_FSDP=True, GPU_COUNT=8, GRAD_ACCUM=2 (FSDP
#     `full_shard auto_wrap` over Qwen2DecoderLayer; DDP would need the full
#     model + fp32 AdamW master/moment states per rank, far over one B200).
#     eff-batch stays 8*2*1 = 16.
# torch-2.7.1 + FSDP + default `adamw_torch` hits a
# `_group_tensors_by_device_and_dtype` device-mismatch RuntimeError on the
# first optimizer step after an end-of-epoch checkpoint save; the engine always
# passes `--optim=adamw_torch_fused`, which routes around that path and is what
# the 14B/32B FSDP rows ran with (the epoch1->2 boundary is the point to watch).
USE_FSDP = False

# -----------------------------------------------------------------------------
# 7. Eval suite (FROZEN by default)
# -----------------------------------------------------------------------------
EVAL_BENCH_SUBSET = ["aime24", "aime25", "math500", "jeebench", "lcb_v5"]
JEE_MATH_ONLY = True           # filter JEEbench to subject=="math" (~236/515)
JEE_N_SAMPLES = 6              # n-sampled averaging protocol
AIME_N_SAMPLES = 3
MATH500_N_SAMPLES = 3

# -----------------------------------------------------------------------------
# 8. Optional: HF upload target (only used by run_uploads.py)
# -----------------------------------------------------------------------------
# Leave HF_REPO as "" to skip uploads.
HF_REPO = ""
# Short description string for the model card; tweak per dataset.
HF_VARIANT_NOTE = ""
