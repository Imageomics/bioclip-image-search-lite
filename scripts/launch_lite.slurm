#!/bin/bash
#SBATCH --job-name=bioclip_lite
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=PAS2136

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
# Set these environment variables before submitting, or edit the defaults below.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${BIOCLIP_VENV:?Set BIOCLIP_VENV to your virtualenv path}"
DATA_DIR="${BIOCLIP_DATA_DIR:?Set BIOCLIP_DATA_DIR to your data directory}"
LOG_DIR="${DATA_DIR}/logs"
FAISS_INDEX="${DATA_DIR}/index.faiss"
DUCKDB_PATH="${DATA_DIR}/metadata.duckdb"
SESSION="bioclip-lite"

# ── Setup ───────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
APP_LOG="$LOG_DIR/app_${SLURM_JOB_ID}.log"

echo "=== BioCLIP Lite Image Search ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Start:    $(date)"
echo "App log:  $APP_LOG"
echo ""
echo "To attach:  ssh $(hostname) -t 'tmux attach -t $SESSION'"
echo ""

source "$VENV/bin/activate"

# ── Launch in tmux ──────────────────────────────────────────────────
# Kill any stale session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
    "source $VENV/bin/activate && python $REPO_ROOT/app.py \
        --faiss-index $FAISS_INDEX \
        --duckdb-path $DUCKDB_PATH \
        --device cpu \
        --scope all \
        --host 0.0.0.0 \
        --port 7860 \
        --log-dir $LOG_DIR \
    2>&1 | tee $APP_LOG"

echo "tmux session '$SESSION' started."
echo "Waiting for session to exit (Ctrl-C the job to stop)..."

# Keep the SLURM job alive while tmux is running
while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 30
done

echo ""
echo "Session ended at $(date)"
