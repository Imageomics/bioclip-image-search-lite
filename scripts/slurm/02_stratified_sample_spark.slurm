#!/usr/bin/env bash
#SBATCH --job-name=lite_02_stratified_sample_spark
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Resource directives (cpu, mem, time, partition, nodes, tasks_per_node) are
# injected by scripts/slurm/submit_pipeline.sh from the config's
# resources.sample section. For multi-node Spark, set nodes > 1 and
# tasks_per_node in the YAML; the slurm-spark-submit launcher picks up the
# allocation automatically.
set -euo pipefail
[[ -n "${ENV_SETUP:-}" ]] && source "$ENV_SETUP"
cd "$SLURM_SUBMIT_DIR"

# Use slurm-spark-submit (OSC-style multi-node Spark cluster launcher) when
# available; fall back to plain python (single-node Spark in the driver JVM)
# otherwise. The fallback works for small datasets (~few M rows); 100M+
# requires multi-node and slurm-spark-submit.
if command -v slurm-spark-submit >/dev/null 2>&1; then
    module load spark/3.5.1 2>/dev/null || true
    slurm-spark-submit \
        --driver-memory   "${SPARK_DRIVER_MEMORY:-64G}" \
        --executor-memory "${SPARK_EXECUTOR_MEMORY:-75G}" \
        scripts/pipeline/02_stratified_sample_spark.py --config "$CONFIG"
else
    python scripts/pipeline/02_stratified_sample_spark.py --config "$CONFIG"
fi
