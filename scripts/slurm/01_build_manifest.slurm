#!/usr/bin/env bash
#SBATCH --job-name=lite_01_manifest
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Resource directives (cpu, mem, time, gpu, array) are injected by
# scripts/slurm/submit_pipeline.sh from the config's `resources.manifest`
# section. Do not hardcode them here.
set -euo pipefail
[[ -n "${ENV_SETUP:-}" ]] && source "$ENV_SETUP"
cd "$SLURM_SUBMIT_DIR"
python scripts/pipeline/01_build_manifest.py --config "$CONFIG"
