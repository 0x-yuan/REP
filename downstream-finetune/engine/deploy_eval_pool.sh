#!/usr/bin/env bash
# Deploy ONE eval app replica per suffix in cfg.EVAL_APP_SUFFIXES (skipping
# the empty "" suffix — that's the base app already deployed by launch.sh).
#
# Modal caches image layers by content hash, so subsequent deploys of the same
# image just register a new app + function bindings.
#
# Usage:
#     bash deploy_eval_pool.sh

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Pull EVAL_APP_SUFFIXES from config.py so the pool size stays in lockstep.
mapfile -t SUFFIXES < <(python -c '
import _config_loader as L
for s in L.cfg.EVAL_APP_SUFFIXES:
    if s:
        print(s)
')

for s in "${SUFFIXES[@]}"; do
    echo "==> deploy eval pool replica ${s}"
    EVAL_APP_SUFFIX="$s" uv run --with modal modal deploy eval_multi.py 2>&1 | tail -3
done

echo "DONE."
