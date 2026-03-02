#!/bin/bash
#SBATCH --job-name=duckdb_lite_convert
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --account=<YOUR_ACCOUNT>   # TODO: set your SLURM account

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${BIOCLIP_VENV:?Set BIOCLIP_VENV to your virtualenv path}"
DATA_DIR="${BIOCLIP_DATA_DIR:?Set BIOCLIP_DATA_DIR to your data directory}"

echo "=== DuckDB Lite Conversion ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

source "$VENV/bin/activate"

python "$REPO_ROOT/scripts/data/convert_duckdb_lite.py" \
    --from-duckdb "${1:?Usage: sbatch convert_duckdb_lite.slurm <source.duckdb>}" \
    --output "$DATA_DIR/metadata.duckdb"

echo ""
echo "End: $(date)"
