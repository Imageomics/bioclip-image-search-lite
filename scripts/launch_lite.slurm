#!/bin/bash
#SBATCH --job-name=bioclip_lite
#SBATCH --output=/fs/scratch/PAS2136/netzissou/bioclip-lite/logs/lite_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=PAS2136

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
VENV="/fs/scratch/PAS2136/netzissou/venv/bioclip-lite"
APP="/users/PZS1008/netzhang/bioclip-image-search-lite/app.py"
LOG_DIR="/fs/scratch/PAS2136/netzissou/bioclip-lite/logs"
FAISS_INDEX="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index"
DUCKDB_PATH="/fs/scratch/PAS2136/netzissou/bioclip-lite/data/metadata.duckdb"
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
    "source $VENV/bin/activate && python $APP \
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
