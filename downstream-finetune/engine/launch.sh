#!/usr/bin/env bash
# One-button launch for ONE distill run.
#
#   1. Deploy train + eval (base + pool) apps + upload app.
#   2. Spawn the training FC (idempotent at the FC layer — re-running creates
#      a second FC, so don't re-run unless you mean to.)
#   3. Launch the per-ckpt eval orchestrator as a nohup daemon.
#
# Pre-req: $MODAL_PROFILE matches cfg.PROFILE (the script warns if not).
#
# Usage:
#   ./launch.sh              # full pipeline
#   ./launch.sh deploy-only  # just the deploys (skip spawn + orchestrate)
#   ./launch.sh spawn-only   # skip deploys; spawn + orchestrate
#   ./launch.sh orch-only    # skip deploys + spawn; just (re)launch the daemon

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-full}"

# --- Sanity: check MODAL_PROFILE matches cfg.PROFILE ------------------------
CFG_PROFILE="$(python -c 'import _config_loader as L; print(L.cfg.PROFILE)' 2>/dev/null || echo "")"
if [[ -n "$CFG_PROFILE" && "${MODAL_PROFILE:-}" != "$CFG_PROFILE" ]]; then
    echo "WARN: MODAL_PROFILE='${MODAL_PROFILE:-<unset>}' but config.py PROFILE='$CFG_PROFILE'."
    echo "      export MODAL_PROFILE=$CFG_PROFILE  # before running this script"
fi

mkdir -p "$ROOT/logs"

# --- 1. Deploys -------------------------------------------------------------
if [[ "$MODE" == "full" || "$MODE" == "deploy-only" ]]; then
    echo "==> deploy train.py"
    uv run --with modal modal deploy train.py 2>&1 | tail -3

    echo "==> deploy eval_multi.py (base + pool)"
    uv run --with modal modal deploy eval_multi.py 2>&1 | tail -3
    bash "$ROOT/deploy_eval_pool.sh"

    echo "==> deploy upload_ckpts.py"
    uv run --with modal modal deploy upload_ckpts.py 2>&1 | tail -3
fi

if [[ "$MODE" == "deploy-only" ]]; then
    echo "DONE (deploy-only)."
    exit 0
fi

# --- 2. Spawn training FC ---------------------------------------------------
if [[ "$MODE" == "full" || "$MODE" == "spawn-only" ]]; then
    echo "==> spawn training FC"
    uv run --with modal python spawn_train.py
fi

# --- 3. Launch orchestrator daemon ------------------------------------------
echo "==> launch per-ckpt eval orchestrator (nohup daemon)"
nohup uv run --with modal python orchestrate.py --include-base \
    >> "$ROOT/logs/orchestrator.stdout.log" 2>&1 < /dev/null &
disown $! || true

sleep 2
echo "==> running orchestrator processes:"
ps -A -o pid,ppid,etime,command | grep "$ROOT/orchestrate.py" | grep -v grep || \
    ps -A -o pid,ppid,etime,command | grep "orchestrate.py" | grep -v grep || true

echo "DONE."
