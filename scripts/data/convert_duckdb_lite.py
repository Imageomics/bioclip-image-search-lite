"""Convert SQLite metadata to optimized DuckDB for BioCLIP Lite.

Copies the existing research DuckDB and adds Lite-specific enhancements:
  1. Materialized has_url BOOLEAN column
  2. Compound index on (source_dataset, has_url) for scope filtering
  3. URL coverage validation

Usage:
    python scripts/data/convert_duckdb_lite.py [--from-sqlite | --from-duckdb PATH]
"""

import argparse
import os
import shutil
import time

import duckdb

SQLITE_DB = (
    "/fs/scratch/PAS2136/TreeOfLife/embeddings/vector_db_sample/"
    "flight_plan/lookup.db"
)
EXISTING_DUCKDB = (
    "/fs/scratch/PAS2136/netzissou/deployment_optimization_research/"
    "2a_duckdb_benchmark/metadata.duckdb"
)
OUTPUT_PATH = "/fs/scratch/PAS2136/netzissou/bioclip-lite/data/metadata.duckdb"
EXPECTED_ROW_COUNT = 234_391_308


def convert_from_sqlite(output_path: str):
    """Full conversion from the 80 GB SQLite source."""
    print(f"Converting from SQLite: {SQLITE_DB}")
    print(f"Output: {output_path}")

    if os.path.exists(output_path):
        os.remove(output_path)

    t0 = time.time()
    conn = duckdb.connect(output_path)
    conn.execute("INSTALL sqlite; LOAD sqlite;")

    print("Transferring 234M rows via sqlite_scan (this takes a while)...")
    conn.execute(f"""
        CREATE TABLE metadata AS
        SELECT * FROM sqlite_scan('{SQLITE_DB}', 'metadata')
    """)
    print(f"Transfer completed in {time.time() - t0:.0f}s")

    # Create primary index
    print("Creating index on id...")
    conn.execute("CREATE INDEX idx_id ON metadata (id)")

    _add_lite_enhancements(conn)
    conn.close()


def convert_from_existing_duckdb(source_path: str, output_path: str):
    """Copy existing research DuckDB and add Lite-specific enhancements."""
    print(f"Copying from: {source_path}")
    print(f"         to: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.copy2(source_path, output_path)
    print(f"Copy complete ({os.path.getsize(output_path) / 1024**3:.1f} GB)")

    conn = duckdb.connect(output_path)
    _add_lite_enhancements(conn)
    conn.close()


def _add_lite_enhancements(conn: duckdb.DuckDBPyConnection):
    """Add has_url column and compound index for scope filtering."""
    # Check if has_url already exists
    cols = [r[0] for r in conn.execute("DESCRIBE metadata").fetchall()]
    if "has_url" in cols:
        print("has_url column already exists, skipping ALTER")
    else:
        print("Adding has_url column...")
        t0 = time.time()
        conn.execute("ALTER TABLE metadata ADD COLUMN has_url BOOLEAN")
        conn.execute(
            "UPDATE metadata SET has_url = "
            "(identifier IS NOT NULL AND identifier != '')"
        )
        print(f"has_url column populated in {time.time() - t0:.0f}s")

    # Compound index for scope queries
    existing_indexes = [
        r[0] for r in conn.execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()
    ]
    if "idx_scope" not in existing_indexes:
        print("Creating compound index idx_scope(source_dataset, has_url)...")
        t0 = time.time()
        conn.execute(
            "CREATE INDEX idx_scope ON metadata (source_dataset, has_url)"
        )
        print(f"Index created in {time.time() - t0:.0f}s")
    else:
        print("idx_scope already exists, skipping")

    # Validate
    _validate(conn)


def _validate(conn: duckdb.DuckDBPyConnection):
    """Print validation stats."""
    total, with_url = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN has_url THEN 1 ELSE 0 END) AS with_url "
        "FROM metadata"
    ).fetchone()

    inat_count = conn.execute(
        "SELECT COUNT(*) FROM metadata "
        "WHERE has_url = true AND source_dataset = 'gbif' "
        "AND publisher LIKE '%iNaturalist%'"
    ).fetchone()[0]

    print(f"\n=== Validation ===")
    print(f"Total rows:     {total:>15,}")
    print(f"With URL:       {with_url:>15,}  ({with_url/total*100:.1f}%)")
    print(f"iNaturalist:    {inat_count:>15,}  ({inat_count/total*100:.1f}%)")
    print(f"Without URL:    {total - with_url:>15,}  ({(total-with_url)/total*100:.1f}%)")

    if total != EXPECTED_ROW_COUNT:
        print(f"WARNING: Expected {EXPECTED_ROW_COUNT:,} rows, got {total:,}")

    size_gb = os.path.getsize(OUTPUT_PATH) / 1024**3
    print(f"DuckDB size:    {size_gb:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="DuckDB Lite conversion")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from-sqlite", action="store_true",
        help="Convert from SQLite source (slow, ~1-2 hours)"
    )
    group.add_argument(
        "--from-duckdb", type=str, default=EXISTING_DUCKDB,
        help=f"Copy from existing DuckDB (default: {EXISTING_DUCKDB})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.from_sqlite:
        convert_from_sqlite(args.output)
    else:
        convert_from_existing_duckdb(args.from_duckdb, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
