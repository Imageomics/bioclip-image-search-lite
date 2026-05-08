#!/usr/bin/env bash
# Run the full index-build pipeline sequentially on the current machine.
# Suitable for development, smoke tests, and small datasets.
#
# Usage:
#   bash scripts/run_local.sh path/to/config.yaml
#
# Each phase prints "outputs already exist, skipping" if its outputs are
# present. Pass FORCE=1 to rebuild everything from scratch.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <config.yaml>" >&2
    exit 2
fi
CONFIG="$1"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

# Pick the sample script based on config.sample.engine
SAMPLE_ENGINE=$(python -c "
from scripts.lib.config import load_config
print(load_config('$CONFIG').sample.engine)
")
SAMPLE_SCRIPT="scripts/pipeline/02_stratified_sample_${SAMPLE_ENGINE}.py"

PHASES=(
    "scripts/pipeline/01_build_manifest.py"
    "$SAMPLE_SCRIPT"
    "scripts/pipeline/03_train_leader.py"
    "scripts/pipeline/04_build_shards.py"
    "scripts/pipeline/05_merge_shards.py"
    "scripts/pipeline/06_build_duckdb.py"
    "scripts/pipeline/07_verify_alignment.py"
)

for phase in "${PHASES[@]}"; do
    echo "==> $phase"
    python "$phase" --config "$CONFIG" $FORCE_FLAG
done

echo "==> pipeline complete"
