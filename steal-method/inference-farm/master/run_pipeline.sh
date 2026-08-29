#!/usr/bin/env bash
# Per-model SGLang pipeline.
#
# This script is self-locating — it resolves the server root from
# `${BASH_SOURCE[0]}`, so it works wherever the repo is dropped:
#   inference-farm/master/run_pipeline.sh
#   exp-01/inference-farm/master/run_pipeline.sh
#
# Two modes:
#   * REPLICAS unset or =1  (default, single-replica path):
#       stop sglang-slave-<MODEL> → redeploy → run queue_runner.py
#   * REPLICAS=N (N >= 2, multi-replica fanout under one Modal account):
#       for r in r0..r{N-1}: stop sglang-slave-<MODEL>-<r> → redeploy
#       then run multi_replica_runner.py with --replicas N
#
# Required env: PROFILE, MODEL
# Optional env (also picked up automatically from <server-root>/experiment.env):
#   EXP_ID                     duplication-namespace prefix; empty = legacy
#                              names (`sglang-slave-<m>`). Set to e.g.
#                              "exp-01" to fully isolate this copy of the
#                              repo from any sibling deployment.
#   LAUNCH_ID                  default = 8-char hex auto-generated. Identifies
#                              this single invocation of run_pipeline.sh.
#                              Every line of the main pipeline log is
#                              prefixed with [L:<id>] so N parallel launches
#                              (even with the same EXP_ID + MODEL) can be
#                              tailed without their output mixing. The launch
#                              is registered at <server-root>/launches/<id>.json
#                              and observable via master/launches.py.
#                              Override only if you want a memorable tag.
#                              EXP_ID and LAUNCH_ID are orthogonal — EXP_ID
#                              is per-experiment-folder, LAUNCH_ID is
#                              per-invocation.
#   REPLICAS                   default 1
#   POLL_INTERVAL              default 10.0
#   MAX_RETRIES                default 3
#   SGLANG_STALL_TIMEOUT_S     default 1200 (queue_runner.py default; bump
#                              to 2400-3600 for chunks with max_new_tokens
#                              >= 16K so the runner doesn't kill an in-flight
#                              decode that's just slow)
#   SGLANG_STATUS_PRINT_INTERVAL_S default 30 — how often to reprint a
#                              progress line per replica even when n_done
#                              hasn't ticked. Lower = chattier logs but
#                              tighter mid-chunk visibility.
#   SHARD_FILES_MIN_ROWS       default 0 (multi-replica only). When >0, any
#                              inbox file with at least this many rows is
#                              pre-sharded into REPLICAS parts before being
#                              queued. Use this to fan out one giant inbox
#                              file across all replicas.
#   WATCH                      default 0. Set to 1 to keep the runner alive
#                              after the inbox drains, polling for new files.
#                              Without it, the runner exits when inbox is
#                              empty + workers idle for DRAIN_QUIET_S seconds.
#   DRAIN_QUIET_S              default 15. How long the inbox + workers must
#                              be idle before the runner exits in drain mode.
#   INBOX_SCAN_INTERVAL_S      default 5. How often to rescan inbox/ for
#                              newly-added files.
#   PIPELINE_INNER_OVERRIDE    test-only hook. If set, the script runs the
#                              given command instead of the real
#                              stop/deploy/runner block. Lets the smoke test
#                              exercise the LAUNCH_ID wrapper without modal.
set -u
set -o pipefail

# ---- self-locate ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- experiment.env (optional) ----------------------------------------------
# Source the per-experiment env file if present. Existing env values win over
# the file (so `EXP_ID=foo bash run_pipeline.sh` still overrides the file).
if [ -f "$SERVER_ROOT/experiment.env" ]; then
  while IFS='=' read -r key val; do
    case "$key" in
      ''|\#*) continue ;;
    esac
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    if [ -z "${!key+x}" ]; then
      export "$key=$val"
    fi
  done < "$SERVER_ROOT/experiment.env"
fi

PROFILE="${PROFILE:?}"
MODEL="${MODEL:?}"
EXP_ID="${EXP_ID:-}"
REPLICAS="${REPLICAS:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-10.0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SHARD_FILES_MIN_ROWS="${SHARD_FILES_MIN_ROWS:-0}"
WATCH="${WATCH:-0}"
DRAIN_QUIET_S="${DRAIN_QUIET_S:-15}"
INBOX_SCAN_INTERVAL_S="${INBOX_SCAN_INTERVAL_S:-5}"

# Validate EXP_ID early so a typo doesn't leak into Modal app names.
if [ -n "$EXP_ID" ] && ! [[ "$EXP_ID" =~ ^[a-zA-Z0-9._-]{1,40}$ ]]; then
  echo "EXP_ID must match [a-zA-Z0-9._-]{1,40}; got '$EXP_ID'" >&2
  exit 1
fi

# Generate or accept LAUNCH_ID (8-char hex by default). Validated to a small
# alphanumeric set so it's safe to splice into log filenames and Modal tags.
if [ -z "${LAUNCH_ID:-}" ]; then
  LAUNCH_ID=$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')
fi
if ! [[ "$LAUNCH_ID" =~ ^[A-Za-z0-9_-]{1,32}$ ]]; then
  echo "LAUNCH_ID must be 1-32 chars of [A-Za-z0-9_-]; got '$LAUNCH_ID'" >&2
  exit 2
fi
export LAUNCH_ID

if [ -n "$EXP_ID" ]; then
  APP_PREFIX="${EXP_ID}-sglang-slave-${MODEL}"
  TAG_NS="[${EXP_ID}/${MODEL}@${PROFILE}]"
else
  APP_PREFIX="sglang-slave-${MODEL}"
  TAG_NS="[${MODEL}@${PROFILE}]"
fi

LAUNCHES_DIR="$SERVER_ROOT/launches"
REGISTRY="$LAUNCHES_DIR/$LAUNCH_ID.json"
LOG="/tmp/sglang_pipeline_${LAUNCH_ID}.log"
mkdir -p "$LAUNCHES_DIR"
cd "$SERVER_ROOT"
: > "$LOG"

if ! [[ "$REPLICAS" =~ ^[0-9]+$ ]] || [ "$REPLICAS" -lt 1 ]; then
  echo "[L:$LAUNCH_ID] $TAG_NS REPLICAS must be a positive integer; got '$REPLICAS'" | tee -a "$LOG" >&2
  exit 1
fi

# --- registry ----------------------------------------------------------------
# Register the launch so master/launches.py can find / monitor / stop it.
START_TS=$(date -u +%FT%TZ)
python3 - "$REGISTRY" "$LAUNCH_ID" "$EXP_ID" "$MODEL" "$PROFILE" "$REPLICAS" "$WATCH" "$LOG" "$START_TS" "$$" <<'PY'
import json, sys
out, lid, exp, model, profile, replicas, watch, log, started_at, pid = sys.argv[1:11]
record = {
    "launch_id": lid,
    "exp_id": exp or None,
    "model": model,
    "profile": profile,
    "replicas": int(replicas),
    "watch": watch == "1",
    "log_path": log,
    "started_at": started_at,
    "pid": int(pid),
    "status": "running",
    "exit_code": None,
    "ended_at": None,
}
with open(out, "w") as f:
    json.dump(record, f, indent=2)
PY

_finalize_registry() {
  local exit_code=$?
  local end_ts
  end_ts=$(date -u +%FT%TZ)
  python3 - "$REGISTRY" "$end_ts" "$exit_code" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
end_ts = sys.argv[2]
exit_code = int(sys.argv[3])
if not p.exists():
    sys.exit(0)
d = json.loads(p.read_text())
d["status"] = "exited"
d["exit_code"] = exit_code
d["ended_at"] = end_ts
p.write_text(json.dumps(d, indent=2))
PY
}
trap _finalize_registry EXIT

# --- banner (stderr; not awk-prefixed so the operator can grab the LAUNCH_ID
# instantly without having to wait for the first inner line) ---------------
{
  echo "============================================================"
  echo "  sglang pipeline LAUNCH_ID=${LAUNCH_ID}"
  echo "  exp_id=${EXP_ID:-<unset>} model=${MODEL} profile=${PROFILE}"
  echo "  replicas=${REPLICAS} watch=${WATCH} server_root=${SERVER_ROOT}"
  echo "  log=${LOG}"
  echo "  registry=${REGISTRY}"
  echo "  tail:    tail -F ${LOG}"
  echo "  monitor: uv run python ${SERVER_ROOT}/master/launches.py status ${LAUNCH_ID}"
  echo "============================================================"
} >&2

# --- main pipeline body, every line wrapped with [L:<launch_id>] -----------
# `awk fflush()` keeps the prefixed output line-buffered so `tail -F` sees
# updates immediately. The pipefail at the top of the script propagates the
# inner exit code through the awk pipe.
{
  echo "$TAG_NS === SGLANG PIPELINE start $START_TS (REPLICAS=$REPLICAS, EXP_ID=${EXP_ID:-<unset>}, server_root=$SERVER_ROOT) ==="

  if [ -n "${PIPELINE_INNER_OVERRIDE:-}" ]; then
    # Test hook — exercises LAUNCH_ID wiring without touching Modal.
    echo "$TAG_NS === inner override: $PIPELINE_INNER_OVERRIDE ==="
    bash -c "$PIPELINE_INNER_OVERRIDE"
    QUEUE_RC=$?
  elif [ "$REPLICAS" -eq 1 ]; then
    APP_NAME="${APP_PREFIX}"

    echo ""
    echo "$TAG_NS === stop ${APP_NAME} ==="
    MODAL_PROFILE="$PROFILE" \
      uv run modal app stop "$APP_NAME" -y 2>&1 | tail -3 || true

    echo ""
    echo "$TAG_NS === deploy ${APP_NAME} ==="
    MODAL_PROFILE="$PROFILE" MODEL_KEY="$MODEL" EXP_ID="$EXP_ID" \
      uv run modal deploy slave/app.py 2>&1 | tail -10
    DEPLOY_RC=${PIPESTATUS[0]}
    if [ "$DEPLOY_RC" -ne 0 ]; then
      echo "$TAG_NS DEPLOY FAILED (exit=$DEPLOY_RC); abort pipeline"
      exit 1
    fi

    echo ""
    echo "$TAG_NS === inbox queue (single replica) ==="
    QR_ARGS=(--model "$MODEL"
             --poll-interval "$POLL_INTERVAL"
             --max-retries "$MAX_RETRIES"
             --inbox-scan-interval-s "$INBOX_SCAN_INTERVAL_S"
             --drain-quiet-s "$DRAIN_QUIET_S")
    if [ "$WATCH" = "1" ]; then
      QR_ARGS+=(--watch)
    fi
    MODAL_PROFILE="$PROFILE" EXP_ID="$EXP_ID" \
      uv run python -u master/queue_runner.py \
      "${QR_ARGS[@]}" 2>&1
    QUEUE_RC=$?

  else
    echo ""
    echo "$TAG_NS === stop replicas r0..r$((REPLICAS - 1)) ==="
    for i in $(seq 0 $((REPLICAS - 1))); do
      APP_NAME="${APP_PREFIX}-r${i}"
      MODAL_PROFILE="$PROFILE" \
        uv run modal app stop "$APP_NAME" -y 2>&1 | tail -1 || true
    done

    echo ""
    echo "$TAG_NS === deploy replicas r0..r$((REPLICAS - 1)) (live-streamed, per-replica prefix) ==="
    DEPLOY_PIDS=()
    DEPLOY_RIDS=()
    for i in $(seq 0 $((REPLICAS - 1))); do
      RID="r${i}"
      DEPLOY_RIDS+=("$RID")
      (
        MODAL_PROFILE="$PROFILE" MODEL_KEY="$MODEL" REPLICA_ID="$RID" EXP_ID="$EXP_ID" \
          uv run modal deploy slave/app.py 2>&1 \
          | awk -v p="[${RID}-deploy] " '{print p $0; fflush()}'
        exit ${PIPESTATUS[0]}
      ) &
      DEPLOY_PIDS+=($!)
    done
    DEPLOY_FAIL=0
    for idx in "${!DEPLOY_PIDS[@]}"; do
      RID="${DEPLOY_RIDS[$idx]}"
      if ! wait "${DEPLOY_PIDS[$idx]}"; then
        DEPLOY_FAIL=1
        echo "$TAG_NS   deploy ${RID} FAILED"
      else
        echo "$TAG_NS   deploy ${RID} ok"
      fi
    done
    if [ "$DEPLOY_FAIL" -ne 0 ]; then
      echo "$TAG_NS one or more deploys failed; abort pipeline"
      exit 1
    fi

    echo ""
    echo "$TAG_NS === inbox queue (multi-replica, N=$REPLICAS) ==="
    MR_ARGS=(--model "$MODEL" --replicas "$REPLICAS"
             --poll-interval "$POLL_INTERVAL"
             --max-retries "$MAX_RETRIES"
             --stall-timeout-s "${SGLANG_STALL_TIMEOUT_S:-1200}"
             --inbox-scan-interval-s "$INBOX_SCAN_INTERVAL_S"
             --drain-quiet-s "$DRAIN_QUIET_S")
    if [ "$SHARD_FILES_MIN_ROWS" -gt 0 ]; then
      MR_ARGS+=(--shard-files-min-rows "$SHARD_FILES_MIN_ROWS")
    fi
    if [ "$WATCH" = "1" ]; then
      MR_ARGS+=(--watch)
    fi
    MODAL_PROFILE="$PROFILE" EXP_ID="$EXP_ID" \
      uv run python -u master/multi_replica_runner.py \
      "${MR_ARGS[@]}" 2>&1
    QUEUE_RC=$?
  fi

  echo ""
  echo "$TAG_NS === DONE (queue exit=$QUEUE_RC) $(date -u +%FT%TZ) ==="
  exit "$QUEUE_RC"
} 2>&1 | awk -v p="[L:$LAUNCH_ID] " '{print p $0; fflush()}' >> "$LOG"
