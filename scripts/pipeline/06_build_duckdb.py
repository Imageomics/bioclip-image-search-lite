"""Phase 06: build the optimized DuckDB metadata lookup.

Joins the user's catalog parquet against the uuid->id mapping from phase 01,
applies the optimizations described in scripts/README.md, and writes a
DuckDB file at output.duckdb_path with two tables:

  metadata
    id                INTEGER  FAISS vector id (joins to index.index)
    uuid              UUID     native 16-byte
    <pass-through>    ...      every other catalog column, with ENUM cast
                               applied where the column appears in
                               duckdb.enum_cardinality_caps and its actual
                               distinct-value count is at or under the cap
    url_prefix_id     USMALLINT  (only if duckdb.url_column is set)
    identifier_suffix VARCHAR    (only if duckdb.url_column is set)
    has_url           BOOLEAN    (only if duckdb.url_column is set)

  url_prefixes  (only if duckdb.url_column is set)
    prefix_id  USMALLINT
    prefix     VARCHAR     bare scheme://domain (no trailing '/')

The metadata table is sorted by duckdb.sort_by and has a single
UNIQUE INDEX `idx_metadata_id` on id (the only hot path; scope filtering
is applied in Python in the app, not as a SQL WHERE clause).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    return [Path(cfg.output.duckdb_path)]


def _q(name: str) -> str:
    """DuckDB-safe identifier quoting."""
    return '"' + name.replace('"', '""') + '"'


def _esc(s: str) -> str:
    """Escape a SQL string literal."""
    return s.replace("'", "''")


def _discover_enums(
    conn: duckdb.DuckDBPyConnection, caps: Dict[str, int]
) -> Dict[str, str]:
    """Return {col -> enum_type_name} for cols whose distinct-value count
    fits under their cap."""
    enum_types: Dict[str, str] = {}
    for col, cap in caps.items():
        n_distinct = conn.execute(
            f"SELECT COUNT(DISTINCT {_q(col)}) FROM catalog WHERE {_q(col)} IS NOT NULL"
        ).fetchone()[0]
        if n_distinct == 0:
            logger.info("  SKIP ENUM for %s: 0 distinct values", col)
            continue
        if n_distinct > cap:
            logger.info("  SKIP ENUM for %s: %d distinct > cap %d", col, n_distinct, cap)
            continue
        values = [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT {_q(col)} FROM catalog "
                f"WHERE {_q(col)} IS NOT NULL ORDER BY {_q(col)}"
            ).fetchall()
        ]
        type_name = f"enum_{col}"
        value_list = ", ".join("'" + _esc(v) + "'" for v in values)
        conn.execute(f"CREATE TYPE {type_name} AS ENUM ({value_list})")
        enum_types[col] = type_name
        logger.info("  ENUM %s: %d distinct values", type_name, n_distinct)
    return enum_types


def _build_url_prefixes(
    conn: duckdb.DuckDBPyConnection, url_col: str
) -> List[Tuple[int, str]]:
    """Discover distinct URL prefixes (scheme://domain) from the catalog
    and return [(prefix_id, prefix)] ordered by frequency descending."""
    rows = conn.execute(f"""
        SELECT
            regexp_extract({_q(url_col)}, '^([a-z]+://[^/]+)', 1) AS prefix,
            COUNT(*) AS n
        FROM catalog
        WHERE {_q(url_col)} IS NOT NULL AND {_q(url_col)} != ''
        GROUP BY prefix
        HAVING prefix IS NOT NULL AND prefix != ''
        ORDER BY n DESC
    """).fetchall()
    prefixes = [(i, p) for i, (p, _) in enumerate(rows)]
    logger.info("  %d distinct URL prefixes", len(prefixes))
    for p, n in rows[:5]:
        logger.info("    %s: %d", p, n)
    return prefixes


def build_duckdb(cfg: PipelineConfig) -> None:
    workdir = Path(cfg.output.workdir)
    u2i_dir = workdir / "uuid_to_id.parquet"
    out_path = Path(cfg.output.duckdb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not u2i_dir.exists():
        raise SystemExit(f"missing input: {u2i_dir} (run phase 01 first)")

    if out_path.exists():
        out_path.unlink()
    wal = out_path.with_suffix(out_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()

    conn = duckdb.connect(str(out_path))
    conn.execute(f"SET threads = {max(1, os.cpu_count() or 8)}")

    t_all = time.time()

    # ── Stage inputs as views ────────────────────────────────────────
    logger.info("=== Stage inputs ===")
    conn.execute(
        f"CREATE VIEW catalog AS SELECT * FROM read_parquet('{cfg.input.catalog_parquet}')"
    )
    conn.execute(
        f"CREATE VIEW u2i AS SELECT * FROM read_parquet('{u2i_dir}/*.parquet')"
    )
    n_cat = conn.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
    n_u2i = conn.execute("SELECT COUNT(*) FROM u2i").fetchone()[0]
    logger.info("catalog rows: %d", n_cat)
    logger.info("uuid->id rows: %d", n_u2i)

    # Catalog schema (we'll emit every column other than the uuid column,
    # because uuid is replaced by the joined id+uuid pair).
    cat_cols = [
        r[0] for r in conn.execute("DESCRIBE SELECT * FROM catalog LIMIT 0").fetchall()
    ]
    if cfg.input.uuid_col not in cat_cols:
        raise SystemExit(
            f"catalog has no '{cfg.input.uuid_col}' column; cols: {cat_cols}"
        )

    # Inner join coverage check.
    n_join = conn.execute(
        "SELECT COUNT(*) FROM catalog c JOIN u2i i ON c.uuid::VARCHAR = i.uuid"
    ).fetchone()[0]
    logger.info("inner-join (catalog ⨝ uuid_to_id): %d rows", n_join)
    if n_join == 0:
        raise SystemExit("catalog and uuid_to_id share no uuids; check input paths")

    # ── Step 1: ENUM types ───────────────────────────────────────────
    logger.info("=== Step 1: ENUM types ===")
    t0 = time.time()
    enum_types = _discover_enums(conn, cfg.duckdb.enum_cardinality_caps)
    logger.info("  (%.0fs)", time.time() - t0)

    # ── Step 2: url_prefixes table ───────────────────────────────────
    prefixes: List[Tuple[int, str]] = []
    if cfg.duckdb.url_column:
        if cfg.duckdb.url_column not in cat_cols:
            raise SystemExit(
                f"duckdb.url_column='{cfg.duckdb.url_column}' not in catalog cols"
            )
        logger.info("=== Step 2: url_prefixes table ===")
        t0 = time.time()
        prefixes = _build_url_prefixes(conn, cfg.duckdb.url_column)
        conn.execute("CREATE TABLE url_prefixes (prefix_id USMALLINT, prefix VARCHAR)")
        conn.executemany("INSERT INTO url_prefixes VALUES (?, ?)", prefixes)
        logger.info("  (%.0fs)", time.time() - t0)

    # ── Step 3: build metadata table ─────────────────────────────────
    logger.info("=== Step 3: build metadata table ===")
    t0 = time.time()

    # Build SELECT clause:
    #   id, uuid as UUID, then every other catalog column (with ENUM cast
    #   if applicable). If duckdb.url_column is set, that column is replaced
    #   by url_prefix_id + identifier_suffix + has_url derived via longest-
    #   prefix LIKE match.
    select_parts: List[str] = [
        "CAST(i.id AS INTEGER) AS id",
        f"CAST(c.{_q(cfg.input.uuid_col)} AS UUID) AS uuid",
    ]

    excluded = set(cfg.duckdb.exclude_cols)
    if excluded:
        # Warn on any excluded names that aren't actually in the catalog,
        # so config typos don't silently no-op.
        unknown = excluded - set(cat_cols)
        if unknown:
            logger.warning("duckdb.exclude_cols not in catalog (no-op): %s", sorted(unknown))
    pass_through_cols = [
        c for c in cat_cols
        if c != cfg.input.uuid_col
        and c != cfg.duckdb.url_column
        and c not in excluded
    ]
    for col in pass_through_cols:
        if col in enum_types:
            select_parts.append(f"TRY_CAST(c.{_q(col)} AS {enum_types[col]}) AS {_q(col)}")
        else:
            select_parts.append(f"c.{_q(col)}")

    if cfg.duckdb.url_column:
        # Longest-prefix-first LIKE match for correct disambiguation when
        # one prefix is a substring of another (rare but real).
        prefix_sorted = sorted(prefixes, key=lambda p: -len(p[1]))
        url = _q(cfg.duckdb.url_column)
        case_pid = "CASE " + " ".join(
            f"WHEN c.{url} LIKE '{_esc(p)}%' THEN {pid}"
            for pid, p in prefix_sorted
        ) + " ELSE NULL END"
        case_suffix = "CASE " + " ".join(
            f"WHEN c.{url} LIKE '{_esc(p)}%' THEN substr(c.{url}, {len(p) + 1})"
            for pid, p in prefix_sorted
        ) + " ELSE NULL END"
        select_parts.extend([
            f"{case_pid} AS url_prefix_id",
            f"{case_suffix} AS identifier_suffix",
            f"(c.{url} IS NOT NULL AND c.{url} != '') AS has_url",
        ])

    select_clause = ",\n    ".join(select_parts)
    sort_clause = ", ".join(_q(c) for c in cfg.duckdb.sort_by)

    conn.execute(f"""
        CREATE TABLE metadata AS
        SELECT
            {select_clause}
        FROM catalog c
        JOIN u2i i ON c.{_q(cfg.input.uuid_col)}::VARCHAR = i.uuid
        ORDER BY {sort_clause}
    """)
    logger.info("  table created (%.0fs)", time.time() - t0)

    # ── Step 4: id index ─────────────────────────────────────────────
    logger.info("=== Step 4: id index ===")
    t0 = time.time()
    conn.execute("CREATE UNIQUE INDEX idx_metadata_id ON metadata (id)")
    logger.info("  idx_metadata_id (%.0fs)", time.time() - t0)

    # ── Step 5: validation ───────────────────────────────────────────
    logger.info("=== Step 5: validation ===")
    n_meta = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    n_distinct_id = conn.execute("SELECT COUNT(DISTINCT id) FROM metadata").fetchone()[0]
    logger.info("metadata rows:   %d", n_meta)
    logger.info("distinct ids:    %d", n_distinct_id)
    if n_meta != n_join:
        raise SystemExit(f"metadata row count {n_meta} != inner-join {n_join}")
    if n_distinct_id != n_meta:
        raise SystemExit("non-unique ids in metadata table")
    lo, hi = conn.execute("SELECT MIN(id), MAX(id) FROM metadata").fetchone()
    logger.info("id range:        [%d, %d]", lo, hi)
    if cfg.duckdb.url_column:
        n_url = conn.execute("SELECT COUNT(*) FROM metadata WHERE has_url").fetchone()[0]
        logger.info("has_url=True:    %d", n_url)
    size_gb = out_path.stat().st_size / 1024**3
    logger.info("output size:     %.2f GB", size_gb)

    conn.close()
    logger.info("done in %.1f min", (time.time() - t_all) / 60)


if __name__ == "__main__":
    raise SystemExit(run_phase("06_build_duckdb", outputs, build_duckdb))
