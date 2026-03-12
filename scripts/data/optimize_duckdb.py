"""Experiment: rebuild DuckDB with size optimizations.

Optimizations applied:
  1. Drop unused columns (scientific_name, basisOfRecord, resolution_status)
  2. Cast id BIGINT → INTEGER, uuid VARCHAR → UUID (native 16-byte)
  3. Sort rows by source_dataset, taxonomy (kingdom→species), scientific_name, common_name
     for better compression via long runs of identical values
  4. Split identifier URLs into prefix (domain) + suffix for dictionary compression
  5. Cast low-cardinality VARCHAR columns to ENUM types

Usage:
    python scripts/data/optimize_duckdb.py \
        --source /path/to/metadata.duckdb \
        --output /path/to/metadata_optimized.duckdb
"""

import argparse
import os
import re
import time

import duckdb


EXPECTED_ROW_COUNT = 234_391_308

# Columns to drop (not used by the app — can be re-added later from source)
DROP_COLUMNS = {"scientific_name", "resolution_status"}

# Low-cardinality columns to convert to ENUM (column → max distinct values observed)
ENUM_CANDIDATES = {
    "source_dataset": 5,       # 2 + NULL
    "kingdom": 50,             # 42 (some dirty data)
    "phylum": 200,             # 135
    "class": 500,              # 383
    "order": 2000,             # 1,531
    "family": 15000,           # 13,088
    "publisher": 600,          # 472
    "img_type": 20,            # 13
    "basisOfRecord": 15,       # 8
}


# Valid biological kingdom values
VALID_KINGDOMS = {
    'Animalia', 'Plantae', 'Fungi', 'Chromista', 'Protozoa',
    'Bacteria', 'Archaea', 'Viruses', 'Metazoa',
    'Archaeplastida', 'incertae sedis',
}


def find_corrupted_ids(conn: duckdb.DuckDBPyConnection) -> set[int]:
    """Find rows with column-shift metadata corruption.

    These are GBIF records where taxonomy columns contain timestamps, UUIDs,
    country names, boolean strings, or scientific names with authority citations
    due to column misalignment during original ingestion.
    """
    placeholders = ",".join(f"'{k}'" for k in VALID_KINGDOMS)

    # Rows with invalid kingdom values
    kingdom_rows = conn.execute(f"""
        SELECT id FROM metadata
        WHERE kingdom IS NOT NULL AND kingdom NOT IN ({placeholders})
    """).fetchall()
    ids = {r[0] for r in kingdom_rows}

    # Rows with valid kingdom but corrupted phylum
    phylum_rows = conn.execute(f"""
        SELECT id FROM metadata
        WHERE (kingdom IS NULL OR kingdom IN ({placeholders}))
          AND phylum IS NOT NULL
          AND (phylum LIKE '2024-%%'
               OR phylum IN ('true', 'false', 'US', 'bracteatum')
               OR phylum LIKE '%%Wall.%%' OR phylum LIKE '%%Pers.%%'
               OR phylum LIKE '%% L.' OR phylum LIKE '%%Makino%%'
               OR phylum LIKE '%%subsp.%%' OR phylum LIKE '%%var.%%'
               OR phylum LIKE '%%Stokes%%' OR phylum LIKE '%%Reveal%%'
               OR phylum LIKE '%%E.Wolf%%')
    """).fetchall()
    ids |= {r[0] for r in phylum_rows}

    # Rows with valid kingdom+phylum but corrupted class
    class_rows = conn.execute(f"""
        SELECT id FROM metadata
        WHERE (kingdom IS NULL OR kingdom IN ({placeholders}))
          AND (phylum NOT LIKE '2024-%%' OR phylum IS NULL)
          AND class IS NOT NULL
          AND (class LIKE '2024-%%'
               OR class LIKE '%%INVALID%%'
               OR class LIKE '%%MATCH%%'
               OR (class LIKE '%% var. %%' AND class LIKE '%%.%%'))
    """).fetchall()
    ids |= {r[0] for r in class_rows}

    return ids


def build_enum_types(source_conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Query source DB to discover distinct values and build ENUM type DDL.

    Returns a dict of column_name → enum_type_name.
    """
    enum_types = {}
    for col, max_card in ENUM_CANDIDATES.items():
        quoted = f'"{col}"' if col in ("order", "class") else col
        rows = source_conn.execute(
            f"SELECT DISTINCT {quoted} FROM metadata "
            f"WHERE {quoted} IS NOT NULL "
            f"ORDER BY {quoted}"
        ).fetchall()
        values = [r[0] for r in rows]

        if len(values) > max_card:
            print(f"  SKIP ENUM for {col}: {len(values)} distinct > {max_card} limit")
            continue

        type_name = f"enum_{col}"
        enum_types[col] = type_name
        print(f"  ENUM {type_name}: {len(values)} distinct values")

    return enum_types


def build_url_prefix_table(source_conn: duckdb.DuckDBPyConnection) -> list[tuple[int, str]]:
    """Extract top URL domain prefixes from identifier column.

    Returns list of (prefix_id, prefix_string) tuples.
    """
    print("  Extracting URL domain prefixes...")
    rows = source_conn.execute("""
        SELECT
            regexp_extract(identifier, '^(https?://[^/]+)', 1) AS domain,
            COUNT(*) AS cnt
        FROM metadata
        WHERE identifier IS NOT NULL AND identifier != ''
        GROUP BY domain
        ORDER BY cnt DESC
    """).fetchall()

    prefixes = [(i, row[0]) for i, row in enumerate(rows) if row[0]]
    print(f"  Found {len(prefixes)} distinct URL domains")
    for domain, cnt in rows[:10]:
        print(f"    {domain}: {cnt:,}")
    return prefixes


def create_optimized_db(source_path: str, output_path: str):
    """Rebuild the DuckDB with all optimizations."""
    print(f"Source: {source_path} ({os.path.getsize(source_path) / 1024**3:.1f} GB)")
    print(f"Output: {output_path}")

    if os.path.exists(output_path):
        os.remove(output_path)
    # Also remove WAL file if present
    wal_path = output_path + ".wal"
    if os.path.exists(wal_path):
        os.remove(wal_path)

    # Open source read-only
    src = duckdb.connect(source_path, read_only=True)
    src_count = src.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    print(f"Source rows: {src_count:,}")

    # Open destination
    dst = duckdb.connect(output_path)
    # Allow more memory for sorting 234M rows
    dst.execute("SET memory_limit = '100GB'")
    dst.execute("SET threads = 8")
    # Attach source
    dst.execute(f"ATTACH '{source_path}' AS src (READ_ONLY)")

    # ── Step 0: Identify corrupted rows ────────────────────────────
    print("\n=== Step 0: Identifying corrupted rows ===")
    corrupted_ids = find_corrupted_ids(src)
    print(f"  Found {len(corrupted_ids)} rows with column-shift corruption")
    if corrupted_ids:
        for cid in sorted(corrupted_ids):
            print(f"    id={cid}")
        # Register as a temp table so we can use it in the CREATE TABLE query
        id_list = ",".join(str(i) for i in corrupted_ids)
        dst.execute(f"CREATE TEMP TABLE corrupted_ids AS SELECT unnest([{id_list}]) AS id")

    # ── Step 1: Build ENUM types ─────────────────────────────────────
    print("\n=== Step 1: Building ENUM types ===")
    # Exclude corrupted rows from ENUM value discovery
    exclude_clause = ""
    if corrupted_ids:
        exclude_clause = f" AND id NOT IN ({id_list})"

    enum_types = build_enum_types(src)

    for col, type_name in enum_types.items():
        quoted = f'"{col}"' if col in ("order", "class") else col
        values = src.execute(
            f"SELECT DISTINCT {quoted} FROM metadata "
            f"WHERE {quoted} IS NOT NULL{exclude_clause} ORDER BY {quoted}"
        ).fetchall()
        value_list = ", ".join(f"'{v[0].replace(chr(39), chr(39)+chr(39))}'" for v in values)
        dst.execute(f"CREATE TYPE {type_name} AS ENUM ({value_list})")

    # ── Step 2: Build URL prefix lookup ──────────────────────────────
    print("\n=== Step 2: Building URL prefix table ===")
    prefixes = build_url_prefix_table(src)

    dst.execute("""
        CREATE TABLE url_prefixes (
            prefix_id USMALLINT,
            prefix VARCHAR
        )
    """)
    dst.executemany(
        "INSERT INTO url_prefixes VALUES (?, ?)",
        prefixes
    )
    # Build a lookup for the SQL CASE expression
    prefix_map = {prefix: pid for pid, prefix in prefixes}

    # ── Step 3: Create optimized metadata table ──────────────────────
    print("\n=== Step 3: Creating optimized metadata table ===")
    print("  Sorting by source_dataset, taxonomy, common_name...")
    print("  Splitting identifier into prefix_id + suffix...")

    # Build column expressions
    col_exprs = []

    # id: BIGINT → INTEGER
    col_exprs.append("CAST(s.id AS INTEGER) AS id")

    # uuid: VARCHAR → UUID native type
    col_exprs.append("CAST(s.uuid AS UUID) AS uuid")

    # Taxonomy columns — NULL out corrupted rows, ENUM cast the rest
    # For corrupted rows, all taxonomy + common_name are garbage from column shift
    has_corrupt = len(corrupted_ids) > 0
    for col in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
        quoted_src = f's."{col}"' if col in ("order", "class") else f"s.{col}"
        if has_corrupt:
            clean_expr = (
                f"CASE WHEN s.id IN (SELECT id FROM corrupted_ids) "
                f"THEN NULL ELSE {quoted_src} END"
            )
        else:
            clean_expr = quoted_src
        if col in enum_types:
            col_exprs.append(
                f"TRY_CAST({clean_expr} AS {enum_types[col]}) AS \"{col}\""
            )
        else:
            col_exprs.append(f"{clean_expr} AS \"{col}\"")

    # common_name stays VARCHAR (177K distinct — too high for ENUM)
    if has_corrupt:
        col_exprs.append(
            "CASE WHEN s.id IN (SELECT id FROM corrupted_ids) "
            "THEN NULL ELSE s.common_name END AS common_name"
        )
    else:
        col_exprs.append("s.common_name")

    # source_dataset, publisher, img_type, basisOfRecord → ENUM
    for col in ["source_dataset", "publisher", "img_type", "basisOfRecord"]:
        if col in enum_types:
            col_exprs.append(
                f"TRY_CAST(s.{col} AS {enum_types[col]}) AS {col}"
            )
        else:
            col_exprs.append(f"s.{col}")

    col_exprs.append("s.source_id")

    # identifier → split into prefix_id + identifier_suffix
    # Build a CASE expression to map domain → prefix_id
    case_parts = []
    for prefix, pid in sorted(prefix_map.items(), key=lambda x: -len(x[0])):
        escaped = prefix.replace("'", "''")
        case_parts.append(
            f"WHEN s.identifier LIKE '{escaped}%' THEN {pid}"
        )
    case_expr = "CASE " + " ".join(case_parts) + " ELSE NULL END"

    col_exprs.append(f"{case_expr} AS url_prefix_id")

    # suffix: strip the matched domain prefix
    suffix_parts = []
    for prefix, pid in sorted(prefix_map.items(), key=lambda x: -len(x[0])):
        escaped = prefix.replace("'", "''")
        suffix_parts.append(
            f"WHEN s.identifier LIKE '{escaped}%' "
            f"THEN substr(s.identifier, {len(prefix) + 1})"
        )
    suffix_expr = "CASE " + " ".join(suffix_parts) + " ELSE s.identifier END"
    col_exprs.append(f"{suffix_expr} AS identifier_suffix")

    col_exprs.append("s.has_url")

    # in_bioclip2_training: carry through if present in source
    src_cols = [r[0] for r in src.execute("DESCRIBE metadata").fetchall()]
    has_training_col = "in_bioclip2_training" in src_cols
    if has_training_col:
        col_exprs.append("s.in_bioclip2_training")
        print("  Including in_bioclip2_training column")

    select_clause = ",\n    ".join(col_exprs)

    # Sort order: source_dataset, taxonomy hierarchy, common_name
    sort_order = (
        'source_dataset, kingdom, phylum, class, "order", family, genus, species, '
        "common_name"
    )

    t0 = time.time()
    create_sql = f"""
        CREATE TABLE metadata AS
        SELECT
            {select_clause}
        FROM src.metadata s
        ORDER BY {sort_order}
    """

    print("  Executing CREATE TABLE ... ORDER BY (this will take a while)...")
    dst.execute(create_sql)
    elapsed = time.time() - t0
    print(f"  Table created in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Step 4: Create indexes ───────────────────────────────────────
    print("\n=== Step 4: Creating indexes ===")
    t0 = time.time()
    dst.execute("CREATE INDEX idx_id ON metadata (id)")
    print(f"  idx_id created in {time.time() - t0:.0f}s")

    t0 = time.time()
    if has_training_col:
        dst.execute(
            "CREATE INDEX idx_scope ON metadata (source_dataset, has_url, in_bioclip2_training)"
        )
    else:
        dst.execute("CREATE INDEX idx_scope ON metadata (source_dataset, has_url)")
    print(f"  idx_scope created in {time.time() - t0:.0f}s")

    # ── Step 5: Validate ─────────────────────────────────────────────
    print("\n=== Step 5: Validation ===")
    validate(dst, src, output_path)

    src.close()
    dst.close()


def validate(dst: duckdb.DuckDBPyConnection, src: duckdb.DuckDBPyConnection, output_path: str):
    """Validate the optimized DB against the source."""
    dst_count = dst.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    src_count = src.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]

    print(f"  Source rows:    {src_count:>15,}")
    print(f"  Output rows:    {dst_count:>15,}")
    if dst_count != src_count:
        print(f"  ERROR: Row count mismatch!")

    # Check a few random IDs match
    sample_ids = src.execute(
        "SELECT id FROM metadata ORDER BY random() LIMIT 20"
    ).fetchall()
    id_list = ",".join(str(r[0]) for r in sample_ids)

    # Compare key fields
    src_rows = src.execute(
        f"SELECT id, uuid, kingdom, species, has_url FROM metadata "
        f"WHERE id IN ({id_list}) ORDER BY id"
    ).fetchall()

    dst_rows = dst.execute(
        f"SELECT id, uuid, kingdom, species, has_url FROM metadata "
        f"WHERE id IN ({id_list}) ORDER BY id"
    ).fetchall()

    # Cast for comparison (uuid type differs in format: no hyphens vs hyphens)
    mismatches = 0
    for s, d in zip(src_rows, dst_rows):
        s_uuid = str(s[1]).replace("-", "")
        d_uuid = str(d[1]).replace("-", "")
        if str(s[0]) != str(d[0]) or s_uuid != d_uuid or \
           str(s[2]) != str(d[2]) or str(s[3]) != str(d[3]) or \
           s[4] != d[4]:
            print(f"  MISMATCH: src={s} dst={d}")
            mismatches += 1

    if mismatches == 0:
        print(f"  Spot check: {len(src_rows)} random rows OK")
    else:
        print(f"  ERROR: {mismatches} mismatches in spot check!")

    # URL reconstruction check
    print("  Checking URL reconstruction...")
    sample_urls = src.execute(
        f"SELECT id, identifier FROM metadata "
        f"WHERE id IN ({id_list}) AND identifier IS NOT NULL "
        f"ORDER BY id"
    ).fetchall()

    dst_urls = dst.execute(
        f"SELECT m.id, COALESCE(p.prefix, '') || COALESCE(m.identifier_suffix, '') "
        f"FROM metadata m "
        f"LEFT JOIN url_prefixes p ON m.url_prefix_id = p.prefix_id "
        f"WHERE m.id IN ({id_list}) AND m.identifier_suffix IS NOT NULL "
        f"ORDER BY m.id"
    ).fetchall()

    url_mismatches = 0
    dst_url_map = {r[0]: r[1] for r in dst_urls}
    for sid, surl in sample_urls:
        durl = dst_url_map.get(sid)
        if durl != surl:
            print(f"  URL MISMATCH id={sid}: src={surl[:80]} dst={durl[:80] if durl else None}")
            url_mismatches += 1

    if url_mismatches == 0:
        print(f"  URL reconstruction: {len(sample_urls)} URLs OK")
    else:
        print(f"  ERROR: {url_mismatches} URL mismatches!")

    # Size report
    size_gb = os.path.getsize(output_path) / 1024**3
    print(f"\n  Output size: {size_gb:.2f} GB")

    # Per-column storage estimate (count distinct blocks × 256 KB block size)
    print("\n  Column storage breakdown:")
    storage = dst.execute("""
        SELECT column_name,
               COUNT(DISTINCT block_id) * 256.0 / 1024 AS mb
        FROM pragma_storage_info('metadata')
        WHERE block_id IS NOT NULL
        GROUP BY column_name
        ORDER BY mb DESC
    """).fetchall()
    for col, mb in storage:
        print(f"    {col:<25s} {mb:>8.1f} MB")

    # Query performance sanity check
    print("\n  Query performance check:")
    test_ids = ",".join(str(r[0]) for r in sample_ids[:10])

    t0 = time.time()
    for _ in range(100):
        dst.execute(
            f"SELECT id, uuid, kingdom, phylum, class, \"order\", family, genus, species, "
            f"common_name, source_dataset, source_id, publisher, img_type, "
            f"COALESCE(p.prefix, '') || COALESCE(m.identifier_suffix, '') AS identifier, "
            f"has_url "
            f"FROM metadata m "
            f"LEFT JOIN url_prefixes p ON m.url_prefix_id = p.prefix_id "
            f"WHERE m.id IN ({test_ids})"
        ).fetchall()
    avg_ms = (time.time() - t0) / 100 * 1000
    print(f"  Avg query time (10 IDs, 100 runs): {avg_ms:.2f} ms")

    t0 = time.time()
    for _ in range(100):
        src.execute(
            f"SELECT id, uuid, kingdom, phylum, class, \"order\", family, genus, species, "
            f"common_name, source_dataset, source_id, publisher, img_type, identifier, has_url "
            f"FROM metadata WHERE id IN ({test_ids})"
        ).fetchall()
    avg_ms_src = (time.time() - t0) / 100 * 1000
    print(f"  Avg query time ORIGINAL (10 IDs, 100 runs): {avg_ms_src:.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize DuckDB: drop columns, ENUM types, sort, split URLs"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to source metadata.duckdb"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for optimized output .duckdb"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_optimized_db(args.source, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
