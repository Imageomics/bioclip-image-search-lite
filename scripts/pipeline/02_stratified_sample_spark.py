"""Phase 02 (spark engine): stratified sample on sample.stratify_col via Spark.

Outputs:
  ${output.workdir}/leader_sample.parquet/   (parquet directory, ZSTD)

Output schema: uuid:string, emb:list<float32>[D]

Strategy: water-filling fair-share allocation.
  Walk groups by ascending size. 
  At each step, fair_share = remaining_budget / remaining_groups. 
  Groups at or below fair_share are taken whole (fraction = 1).
  Groups above it share a single cap (= fair_share). 
  NULL values in stratify_col are bucketed under the literal "Unknown".

Use this script when sample.engine == "spark". For sample.engine == "duckdb"
see 02_stratified_sample_duckdb.py. Both produce the same output schema and
target row count; the exact uuids sampled may differ because the two engines
have different RNGs and partition layouts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    return [Path(cfg.output.workdir) / "leader_sample.parquet"]


def stratified_sample_spark(cfg: PipelineConfig) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, coalesce, lit

    out_path = Path(cfg.output.workdir) / "leader_sample.parquet"
    Path(cfg.output.workdir).mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("lite_stratified_sample")
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.io.compression.zstd.level", "3")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    stratify = cfg.sample.stratify_col
    uuid_col = cfg.input.uuid_col
    emb_col  = cfg.input.embedding_col

    df = spark.read.parquet(cfg.input.embeddings_parquet).select(
        col(uuid_col).cast("string").alias("uuid"),
        col(emb_col).alias("emb"),
        coalesce(col(stratify), lit("Unknown")).alias("__stratify__"),
    )

    # Per-group counts (no embedding read; Spark prunes columns).
    counts = (
        df.groupBy("__stratify__").count()
        .orderBy(col("count").asc())
        .collect()
    )
    n_groups = len(counts)
    total_rows = sum(r["count"] for r in counts)
    logger.info("input rows: %d  distinct '%s' values: %d", total_rows, stratify, n_groups)

    # Water-filling fair-share allocation.
    pairs = [(r["__stratify__"], r["count"]) for r in counts]  # already asc by count
    remaining_budget = cfg.sample.n_total
    remaining = n_groups
    min_floor = cfg.sample.min_per_class

    fractions: dict[str, float] = {}
    cap_for_large = 0
    small_total = 0
    n_small = n_large = 0
    for group, cnt in pairs:
        fair_share = max(min_floor, int(remaining_budget / max(remaining, 1)))
        if cnt <= fair_share:
            fractions[group] = 1.0
            remaining_budget -= cnt
            small_total += cnt
            n_small += 1
        else:
            cap_for_large = fair_share
            fractions[group] = fair_share / cnt
            remaining_budget -= fair_share
            n_large += 1
        remaining -= 1

    expected = small_total + n_large * cap_for_large
    logger.info(
        "water-fill: cap_for_large=%d  small_groups=%d (whole, total=%d)  large_groups=%d",
        cap_for_large, n_small, small_total, n_large,
    )
    logger.info("expected sample size: %d", expected)

    sampled = df.sampleBy("__stratify__", fractions, seed=cfg.sample.seed)
    sampled = sampled.select("uuid", "emb")

    # Repartition so output file count is sane (~250k rows per file).
    target_parts = max(1, expected // 250_000)
    logger.info("writing %s (target %d partitions)", out_path, target_parts)
    (sampled
        .repartition(target_parts)
        .write.mode("overwrite")
        .option("compression", "zstd")
        .parquet(str(out_path)))

    # Quick verification read-back (count only, no full materialization).
    back = spark.read.parquet(str(out_path))
    n_written = back.count()
    logger.info("written sample rows: %d", n_written)

    spark.stop()


if __name__ == "__main__":
    raise SystemExit(run_phase(
        "02_stratified_sample_spark", outputs, stratified_sample_spark,
    ))
