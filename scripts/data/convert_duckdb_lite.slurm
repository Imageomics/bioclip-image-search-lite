#!/bin/bash
#SBATCH --job-name=duckdb_lite_convert
#SBATCH --output=/fs/scratch/PAS2136/netzissou/bioclip-lite/logs/convert_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --account=PAS2136

set -euo pipefail

echo "=== DuckDB Lite Conversion ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

module load miniconda3/24.1.2-py310
eval "$(conda shell.bash hook)"
conda activate /fs/scratch/PAS2136/netzissou/conda-envs/faiss_env

python /users/PZS1008/netzhang/bioclip-image-search-lite/scripts/data/convert_duckdb_lite.py \
    --from-duckdb /fs/scratch/PAS2136/netzissou/deployment_optimization_research/2a_duckdb_benchmark/metadata.duckdb \
    --output /fs/scratch/PAS2136/netzissou/bioclip-lite/data/metadata.duckdb

echo ""
echo "End: $(date)"
