#!/usr/bin/env bash
#SBATCH --job-name=lite_06_build_duckdb
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Resource directives (cpu, mem, time, gpu, array) are injected by
# scripts/slurm/submit_pipeline.sh from the config's resources section.
# Do not hardcode them here.
set -euo pipefail
[[ -n "${ENV_SETUP:-}" ]] && source "$ENV_SETUP"
cd "$SLURM_SUBMIT_DIR"
python scripts/pipeline/06_build_duckdb.py --config "$CONFIG"
