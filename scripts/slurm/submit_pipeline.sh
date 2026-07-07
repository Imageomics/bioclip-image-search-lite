#!/usr/bin/env bash
# Submit the full index-build pipeline to SLURM with --dependency=afterok
# chaining. Reads sbatch parameters (cpu, mem, walltime, gpu, array_size)
# from the YAML's `resources` section.
#
# Usage:
#   bash scripts/slurm/submit_pipeline.sh path/to/config.yaml
#
# Each phase's .slurm template lives next to this script. The template body
# calls the corresponding pipeline/NN_*.py with --config $CONFIG.

# Exit on any error, undefined var or pipe failure
set -euo pipefail         


# Convert config path to absolute
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <config.yaml>" >&2
    exit 2
fi
CONFIG="$(realpath "$1")"

# Always run from repo root
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Activate the build env in the current shell too, since submit_pipeline.sh
# itself runs Python (for YAML reads) before submitting. The same ENV_SETUP
# is propagated to each compute node via sbatch --export=ALL.
if [[ -n "${ENV_SETUP:-}" ]]; then
    source "$ENV_SETUP"
fi

# Helper: read one resource field from YAML via Pydantic-resolved config.
read_resource() {
    local phase="$1"; local field="$2"
    python -c "
from scripts.lib.config import load_config
r = getattr(load_config('$CONFIG').resources, '$phase')
v = getattr(r, '$field')
print('' if v is None else v)
"
}

# Pick the sample slurm based on config.sample.engine
SAMPLE_ENGINE=$(python -c "
from scripts.lib.config import load_config
print(load_config('$CONFIG').sample.engine)
")

PHASES=(
    "manifest:scripts/slurm/01_build_manifest.slurm"
    "sample:scripts/slurm/02_stratified_sample_${SAMPLE_ENGINE}.slurm"
    "train_leader:scripts/slurm/03_train_leader.slurm"
    "shards:scripts/slurm/04_build_shards.slurm"
    "merge:scripts/slurm/05_merge_shards.slurm"
    "duckdb:scripts/slurm/06_build_duckdb.slurm"
    "verify:scripts/slurm/07_verify_alignment.slurm"
)

prev_jobid=""
for entry in "${PHASES[@]}"; do
    phase="${entry%%:*}"; tpl="${entry##*:}"
    cpu=$(read_resource "$phase" cpu)
    mem=$(read_resource "$phase" mem)
    wall=$(read_resource "$phase" walltime)
    gpu=$(read_resource "$phase" gpu)
    array=$(read_resource "$phase" array_size)
    partition=$(read_resource "$phase" partition)
    account=$(read_resource "$phase" account)
    nodes=$(read_resource "$phase" nodes)
    tasks_per_node=$(read_resource "$phase" tasks_per_node)

    sbatch_args=( --time="$wall" )
    if [[ -n "$cpu" ]]; then
        sbatch_args+=( --cpus-per-task="$cpu" )
    fi
    if [[ -n "$mem" ]]; then
        sbatch_args+=( --mem="$mem" )
    fi
    if [[ -n "$partition" ]]; then
        sbatch_args+=( --partition="$partition" )
    fi
    if [[ -n "$account" ]]; then
        sbatch_args+=( --account="$account" )
    fi
    if [[ -n "$nodes" ]]; then
        sbatch_args+=( --nodes="$nodes" )
    fi
    if [[ -n "$tasks_per_node" ]]; then
        sbatch_args+=( --ntasks-per-node="$tasks_per_node" )
    fi
    if [[ "$gpu" -gt 0 ]]; then
        sbatch_args+=( --gpus="$gpu" )
    fi
    if [[ -n "$array" ]]; then
        sbatch_args+=( --array=0-$((array - 1)) )
    fi
    if [[ -n "$prev_jobid" ]]; then
        sbatch_args+=( --dependency=afterok:"$prev_jobid" )
    fi

    out=$(sbatch --parsable "${sbatch_args[@]}" --export=ALL,CONFIG="$CONFIG" "$tpl")
    prev_jobid="$out"
    echo "submitted $phase as job $prev_jobid (template: $tpl)"
done

echo "==> all 7 phases queued; tail -F slurm-<jobid>.out to follow"
