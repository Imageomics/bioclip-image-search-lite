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

echo "=== BioCLIP Lite Image Search ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate /fs/scratch/PAS2136/netzissou/conda-envs/faiss_env

FAISS_INDEX="/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/index_200M_normalized_stratified_merged.index"
DUCKDB_PATH="/fs/scratch/PAS2136/netzissou/bioclip-lite/data/metadata.duckdb"

python /users/PZS1008/netzhang/bioclip-image-search-lite/app.py \
    --faiss-index "$FAISS_INDEX" \
    --duckdb-path "$DUCKDB_PATH" \
    --device cpu \
    --scope all \
    --host 0.0.0.0 \
    --port 7860
