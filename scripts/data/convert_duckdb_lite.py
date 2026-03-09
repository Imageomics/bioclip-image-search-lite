"""Convert source metadata to optimized DuckDB for BioCLIP Lite.

Two-stage pipeline:
  Stage 1 (this script): Import raw metadata from SQLite or DuckDB source,
           add has_url column, and create a base DuckDB.
  Stage 2 (optimize_duckdb.py): Apply size optimizations (ENUM types, taxonomy
           sort, URL prefix split, type downcasting, corruption cleanup).

Usage:
    # From SQLite source (slow, ~1-2 hours):
    python scripts/data/convert_duckdb_lite.py --from-sqlite SOURCE --output OUT

    # From existing research DuckDB:
    python scripts/data/convert_duckdb_lite.py --from-duckdb SOURCE --output OUT

    # Then optimize:
    python scripts/data/optimize_duckdb.py --source OUT --output OPTIMIZED
"""

import argparse
import os
import shutil
import time

import duckdb

EXPECTED_ROW_COUNT = 234_391_308


def convert_from_sqlite(sqlite_path: str, output_path: str):
    """Full conversion from the 80 GB SQLite source."""
    print(f"Converting from SQLite: {sqlite_path}")
    print(f"Output: {output_path}")

    if os.path.exists(output_path):
        os.remove(output_path)

    t0 = time.time()
    conn = duckdb.connect(output_path)
    conn.execute("INSTALL sqlite; LOAD sqlite;")

    print("Transferring 234M rows via sqlite_scan (this takes a while)...")
    conn.execute(f"""
        CREATE TABLE metadata AS
        SELECT * FROM sqlite_scan('{sqlite_path}', 'metadata')
    """)
    print(f"Transfer completed in {time.time() - t0:.0f}s")

    # Create primary index
    print("Creating index on id...")
    conn.execute("CREATE INDEX idx_id ON metadata (id)")

    _add_has_url(conn)
    _validate(conn, output_path)
    conn.close()


def convert_from_existing_duckdb(source_path: str, output_path: str):
    """Copy existing research DuckDB and add has_url if missing."""
    print(f"Copying from: {source_path}")
    print(f"         to: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.copy2(source_path, output_path)
    print(f"Copy complete ({os.path.getsize(output_path) / 1024**3:.1f} GB)")

    conn = duckdb.connect(output_path)
    _add_has_url(conn)
    _validate(conn, output_path)
    conn.close()


def _add_has_url(conn: duckdb.DuckDBPyConnection):
    """Add has_url BOOLEAN column if not present."""
    cols = [r[0] for r in conn.execute("DESCRIBE metadata").fetchall()]
    if "has_url" in cols:
        print("has_url column already exists, skipping")
    else:
        print("Adding has_url column...")
        t0 = time.time()
        conn.execute("ALTER TABLE metadata ADD COLUMN has_url BOOLEAN")
        conn.execute(
            "UPDATE metadata SET has_url = "
            "(identifier IS NOT NULL AND identifier != '')"
        )
        print(f"has_url column populated in {time.time() - t0:.0f}s")


def _validate(conn: duckdb.DuckDBPyConnection, output_path: str):
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

    size_gb = os.path.getsize(output_path) / 1024**3
    print(f"DuckDB size:    {size_gb:.1f} GB")
    print(f"\nNext step: run optimize_duckdb.py --source {output_path} --output <optimized.duckdb>")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Import metadata into base DuckDB. "
        "Run optimize_duckdb.py afterward for size optimization."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-sqlite", type=str, metavar="PATH",
        help="Convert from SQLite source (slow, ~1-2 hours)"
    )
    group.add_argument(
        "--from-duckdb", type=str,
        help="Copy from existing DuckDB and add has_url column"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output DuckDB path (base DB, not yet optimized)"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.from_sqlite:
        convert_from_sqlite(args.from_sqlite, args.output)
    else:
        convert_from_existing_duckdb(args.from_duckdb, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
