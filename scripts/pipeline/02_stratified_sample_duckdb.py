"""Phase 02 (duckdb engine): stratified sample on sample.stratify_col via DuckDB.

Outputs:
  ${output.workdir}/leader_sample.parquet/   (parquet directory, ZSTD)

Output schema: uuid:string, emb:list<float32>[D]

Strategy: same water-filling fair-share allocation as the Spark variant.
NULL values in stratify_col are bucketed under the literal "Unknown".

Reproducibility: rows within a group are ranked by
hash(uuid || '|' || seed::VARCHAR) and the top-take_n per group are kept.
The same uuid+seed always produces the same rank, so reruns are stable.

Use this script when sample.engine == "duckdb". Single-node, works to
~100M rows on a hugemem-class machine (peak memory scales with the union
of grouped rows held during ranking; for the canonical 233M-vector build,
prefer the Spark variant).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    return [Path(cfg.output.workdir) / "leader_sample.parquet"]


def _water_fill(
    pairs: List[Tuple[str, int]], target_total: int, min_per_class: int
) -> Tuple[dict[str, int], dict]:
    """Per-group take_n via fair-share water-filling.

    pairs: [(group, count)] sorted by count ascending.
    Returns (take_n_by_group, info_dict).
    """
    remaining = len(pairs)
    budget = target_total
    take_n: dict[str, int] = {}
    cap_for_large = 0
    small_total = 0
    n_small = n_large = 0

    for group, cnt in pairs:
        fair = max(min_per_class, int(budget / max(remaining, 1)))
        if cnt <= fair:
            take_n[group] = cnt
            budget -= cnt
            small_total += cnt
            n_small += 1
        else:
            cap_for_large = fair
            take_n[group] = fair
            budget -= fair
            n_large += 1
        remaining -= 1

    info = {
        "n_groups": len(pairs),
        "n_small": n_small,
        "n_large": n_large,
        "cap_for_large": cap_for_large,
        "small_total": small_total,
        "expected": small_total + n_large * cap_for_large,
    }
    return take_n, info


def stratified_sample_duckdb(cfg: PipelineConfig) -> None:
    out_dir = Path(cfg.output.workdir) / "leader_sample.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    stratify = cfg.sample.stratify_col
    uuid_col = cfg.input.uuid_col
    emb_col  = cfg.input.embedding_col

    con = duckdb.connect()
    con.execute(f"SET threads={max(1, (__import__('os').cpu_count() or 8))}")

    src = cfg.input.embeddings_parquet
    # Per-group counts (column projection only; no emb scanned).
    counts = con.sql(f"""
        SELECT COALESCE(CAST({stratify} AS VARCHAR), 'Unknown') AS g, COUNT(*) AS cnt
        FROM read_parquet('{src}')
        GROUP BY 1
        ORDER BY cnt ASC
    """).fetchall()
    pairs = [(g, int(cnt)) for g, cnt in counts]
    total = sum(c for _, c in pairs)
    logger.info("input rows: %d  distinct '%s' values: %d", total, stratify, len(pairs))

    take_n, info = _water_fill(pairs, cfg.sample.n_total, cfg.sample.min_per_class)
    logger.info(
        "water-fill: cap_for_large=%d  small_groups=%d (whole, total=%d)  large_groups=%d",
        info["cap_for_large"], info["n_small"], info["small_total"], info["n_large"],
    )
    logger.info("expected sample size: %d", info["expected"])

    # Stage take_n table for the join.
    con.execute("CREATE TEMP TABLE limits (g VARCHAR, take_n BIGINT)")
    con.executemany("INSERT INTO limits VALUES (?, ?)", list(take_n.items()))

    seed = cfg.sample.seed
    out_part = out_dir / "part-00000.parquet"

    # Single-pass: rank within group by deterministic hash, keep top take_n.
    # We let DuckDB stream the result into a ZSTD parquet directly to keep
    # peak RSS bounded.
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT
                    {uuid_col}::VARCHAR AS uuid,
                    {emb_col}           AS emb,
                    COALESCE(CAST({stratify} AS VARCHAR), 'Unknown') AS g,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(CAST({stratify} AS VARCHAR), 'Unknown')
                        ORDER BY hash({uuid_col}::VARCHAR || '|' || CAST({seed} AS VARCHAR))
                    ) AS rn
                FROM read_parquet('{src}')
            )
            SELECT r.uuid, r.emb
            FROM ranked r JOIN limits l ON r.g = l.g
            WHERE r.rn <= l.take_n
        )
        TO '{out_part}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)
    """)

    n_written = con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_part}')").fetchone()[0]
    logger.info("written sample rows: %d -> %s", n_written, out_part)


if __name__ == "__main__":
    raise SystemExit(run_phase(
        "02_stratified_sample_duckdb", outputs, stratified_sample_duckdb,
    ))
