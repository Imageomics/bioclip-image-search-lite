"""Validate optimized DuckDB against the original source.

Checks:
  1. Row count matches
  2. Random row spot-checks (id, uuid, taxonomy, has_url)
  3. URL reconstruction (prefix table + suffix == original identifier)
  4. Corrupted rows have NULLed taxonomy
  5. Per-column storage breakdown
  6. Query performance comparison (optimized vs original)
  7. Schema and index verification

Usage:
    python scripts/data/validate_optimized_duckdb.py \
        --source /path/to/metadata.duckdb \
        --optimized /path/to/metadata_optimized.duckdb
"""

import argparse
import os
import time

import duckdb


VALID_KINGDOMS = {
    'Animalia', 'Plantae', 'Fungi', 'Chromista', 'Protozoa',
    'Bacteria', 'Archaea', 'Viruses', 'Metazoa',
    'Archaeplastida', 'incertae sedis',
}

TAXONOMY_COLS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Columns the app selects (from config.py METADATA_COLUMNS)
APP_COLUMNS = [
    "id", "uuid", "kingdom", "phylum", "class", '"order"', "family", "genus",
    "species", "common_name", "source_dataset", "source_id", "publisher",
    "img_type", "identifier", "has_url",
]


def validate(source_path: str, optimized_path: str):
    passed = 0
    failed = 0

    src = duckdb.connect(source_path, read_only=True)
    opt = duckdb.connect(optimized_path, read_only=True)

    # ── 1. Row count ─────────────────────────────────────────────────
    print("=== 1. Row Count ===")
    src_count = src.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    opt_count = opt.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    print(f"  Source:    {src_count:>15,}")
    print(f"  Optimized: {opt_count:>15,}")
    if src_count == opt_count:
        print("  PASS")
        passed += 1
    else:
        print("  FAIL: row count mismatch")
        failed += 1

    # ── 2. Random spot-check ─────────────────────────────────────────
    print("\n=== 2. Random Spot-Check (100 rows) ===")
    sample_ids = src.execute(
        "SELECT id FROM metadata ORDER BY random() LIMIT 100"
    ).fetchall()
    id_list = ",".join(str(r[0]) for r in sample_ids)

    src_rows = src.execute(
        f"SELECT id, uuid, kingdom, species, has_url, source_dataset "
        f"FROM metadata WHERE id IN ({id_list}) ORDER BY id"
    ).fetchall()
    opt_rows = opt.execute(
        f"SELECT id, uuid, kingdom, species, has_url, source_dataset "
        f"FROM metadata WHERE id IN ({id_list}) ORDER BY id"
    ).fetchall()

    mismatches = 0
    for s, o in zip(src_rows, opt_rows):
        s_uuid = str(s[1]).replace("-", "")
        o_uuid = str(o[1]).replace("-", "")
        # kingdom/species may be NULL in optimized if row was corrupted
        s_kingdom = str(s[2]) if s[2] else None
        o_kingdom = str(o[2]) if o[2] else None
        s_species = str(s[3]) if s[3] else None
        o_species = str(o[3]) if o[3] else None

        id_ok = s[0] == o[0]
        uuid_ok = s_uuid == o_uuid
        has_url_ok = s[4] == o[4]
        source_ok = str(s[5]) == str(o[5])
        # Taxonomy may differ if row was corrupted (NULLed in optimized)
        taxonomy_ok = (o_kingdom == s_kingdom and o_species == s_species) or \
                      (o_kingdom is None and s_kingdom not in VALID_KINGDOMS)

        if not (id_ok and uuid_ok and has_url_ok and source_ok and taxonomy_ok):
            print(f"  MISMATCH id={s[0]}:")
            print(f"    src: uuid={s[1]}, kingdom={s[2]}, species={s[3]}, has_url={s[4]}")
            print(f"    opt: uuid={o[1]}, kingdom={o[2]}, species={o[3]}, has_url={o[4]}")
            mismatches += 1

    if mismatches == 0:
        print(f"  PASS ({len(src_rows)} rows checked)")
        passed += 1
    else:
        print(f"  FAIL: {mismatches} mismatches")
        failed += 1

    # ── 3. URL reconstruction ────────────────────────────────────────
    print("\n=== 3. URL Reconstruction ===")
    has_prefix_table = opt.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'url_prefixes'"
    ).fetchone()[0] > 0

    if has_prefix_table:
        # Sample 200 rows with URLs
        url_sample = src.execute(
            f"SELECT id, identifier FROM metadata "
            f"WHERE id IN ({id_list}) AND identifier IS NOT NULL "
            f"ORDER BY id"
        ).fetchall()

        opt_urls = opt.execute(
            f"SELECT m.id, "
            f"  COALESCE(p.prefix, '') || COALESCE(m.identifier_suffix, '') "
            f"FROM metadata m "
            f"LEFT JOIN url_prefixes p ON m.url_prefix_id = p.prefix_id "
            f"WHERE m.id IN ({id_list}) AND "
            f"  (m.identifier_suffix IS NOT NULL OR m.url_prefix_id IS NOT NULL) "
            f"ORDER BY m.id"
        ).fetchall()

        opt_url_map = {r[0]: r[1] for r in opt_urls}
        url_mismatches = 0
        for sid, surl in url_sample:
            ourl = opt_url_map.get(sid)
            if ourl != surl:
                print(f"  MISMATCH id={sid}:")
                print(f"    src: {surl[:100]}")
                print(f"    opt: {ourl[:100] if ourl else None}")
                url_mismatches += 1

        if url_mismatches == 0:
            print(f"  PASS ({len(url_sample)} URLs checked)")
            passed += 1
        else:
            print(f"  FAIL: {url_mismatches} URL mismatches")
            failed += 1
    else:
        print("  SKIP: no url_prefixes table found")

    # ── 4. Corrupted row cleanup ─────────────────────────────────────
    print("\n=== 4. Corrupted Row Cleanup ===")
    placeholders_str = ",".join(f"'{k}'" for k in VALID_KINGDOMS)

    # Find corrupted IDs from source
    corrupt_src = src.execute(f"""
        SELECT id FROM metadata
        WHERE kingdom IS NOT NULL AND kingdom NOT IN ({placeholders_str})
    """).fetchall()
    corrupt_ids = [r[0] for r in corrupt_src]

    if corrupt_ids:
        corrupt_id_list = ",".join(str(i) for i in corrupt_ids)
        # Check that these rows have NULL taxonomy in optimized
        opt_corrupt = opt.execute(f"""
            SELECT id, kingdom, phylum, class, "order", family, genus, species, common_name
            FROM metadata
            WHERE id IN ({corrupt_id_list})
        """).fetchall()

        not_cleaned = 0
        for row in opt_corrupt:
            # All taxonomy cols (index 1-8) should be NULL
            for i, col in enumerate(TAXONOMY_COLS + ["common_name"], 1):
                if row[i] is not None:
                    print(f"  NOT CLEANED id={row[0]}: {col}={row[i]}")
                    not_cleaned += 1
                    break

        if not_cleaned == 0:
            print(f"  PASS ({len(corrupt_ids)} corrupted rows have NULLed taxonomy)")
            passed += 1
        else:
            print(f"  FAIL: {not_cleaned} rows still have non-NULL taxonomy")
            failed += 1
    else:
        print("  SKIP: no corrupted rows found in source")

    # ── 5. No new corruption introduced ──────────────────────────────
    print("\n=== 5. No New Corruption ===")
    # Check that all non-NULL kingdom values in optimized are valid
    opt_kingdoms = opt.execute("""
        SELECT DISTINCT kingdom FROM metadata WHERE kingdom IS NOT NULL
    """).fetchall()
    invalid = [r[0] for r in opt_kingdoms if str(r[0]) not in VALID_KINGDOMS]
    if not invalid:
        print(f"  PASS (all {len(opt_kingdoms)} distinct kingdoms are valid)")
        passed += 1
    else:
        print(f"  FAIL: invalid kingdoms found: {invalid[:10]}")
        failed += 1

    # ── 6. Schema and indexes ────────────────────────────────────────
    print("\n=== 6. Schema & Indexes ===")
    schema = opt.execute("DESCRIBE metadata").fetchall()
    col_types = {r[0]: r[1] for r in schema}
    print("  Columns:")
    for name, dtype in col_types.items():
        # Truncate long ENUM type strings
        dtype_str = str(dtype)
        if len(dtype_str) > 60:
            dtype_str = dtype_str[:57] + "..."
        print(f"    {name:<25s} {dtype_str}")

    indexes = opt.execute(
        "SELECT index_name FROM duckdb_indexes()"
    ).fetchall()
    idx_names = {r[0] for r in indexes}
    print(f"\n  Indexes: {', '.join(sorted(idx_names))}")

    required_indexes = {"idx_id", "idx_scope"}
    if required_indexes.issubset(idx_names):
        print("  PASS (required indexes present)")
        passed += 1
    else:
        missing = required_indexes - idx_names
        print(f"  FAIL: missing indexes: {missing}")
        failed += 1

    # Check id type is INTEGER (not BIGINT)
    if "INTEGER" in str(col_types.get("id", "")):
        print("  PASS (id is INTEGER)")
        passed += 1
    else:
        print(f"  FAIL: id type is {col_types.get('id')}, expected INTEGER")
        failed += 1

    # Check uuid type is UUID (not VARCHAR)
    if "UUID" in str(col_types.get("uuid", "")):
        print("  PASS (uuid is native UUID)")
        passed += 1
    else:
        print(f"  FAIL: uuid type is {col_types.get('uuid')}, expected UUID")
        failed += 1

    # ── 7. Column storage breakdown ──────────────────────────────────
    print("\n=== 7. Storage Breakdown ===")
    src_size = os.path.getsize(source_path) / 1024**3
    opt_size = os.path.getsize(optimized_path) / 1024**3
    print(f"  Source:    {src_size:.2f} GB")
    print(f"  Optimized: {opt_size:.2f} GB")
    print(f"  Reduction: {(1 - opt_size/src_size)*100:.1f}%")

    storage = opt.execute("""
        SELECT column_name,
               COUNT(DISTINCT block_id) * 256.0 / 1024 AS mb
        FROM pragma_storage_info('metadata')
        WHERE block_id IS NOT NULL
        GROUP BY column_name
        ORDER BY mb DESC
    """).fetchall()
    total = 0
    print(f"\n  {'Column':<25s} {'Size (MB)':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for col, mb in storage:
        print(f"  {col:<25s} {mb:>10.1f}")
        total += mb
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'TOTAL':<25s} {total:>10.1f}")

    # ── 8. Query performance ─────────────────────────────────────────
    print("\n=== 8. Query Performance ===")
    test_ids = ",".join(str(r[0]) for r in sample_ids[:10])

    # Optimized query (with URL join)
    opt_query = (
        f"SELECT m.id, m.uuid, m.kingdom, m.phylum, m.class, m.\"order\", "
        f"m.family, m.genus, m.species, m.common_name, m.source_dataset, "
        f"m.source_id, m.publisher, m.img_type, "
        f"COALESCE(p.prefix, '') || COALESCE(m.identifier_suffix, '') AS identifier, "
        f"m.has_url "
        f"FROM metadata m "
        f"LEFT JOIN url_prefixes p ON m.url_prefix_id = p.prefix_id "
        f"WHERE m.id IN ({test_ids})"
    )

    # Source query (direct)
    src_query = (
        f'SELECT id, uuid, kingdom, phylum, class, "order", family, genus, '
        f"species, common_name, source_dataset, source_id, publisher, "
        f"img_type, identifier, has_url "
        f"FROM metadata WHERE id IN ({test_ids})"
    )

    # Warmup
    opt.execute(opt_query).fetchall()
    src.execute(src_query).fetchall()

    iterations = 500
    t0 = time.time()
    for _ in range(iterations):
        opt.execute(opt_query).fetchall()
    opt_ms = (time.time() - t0) / iterations * 1000

    t0 = time.time()
    for _ in range(iterations):
        src.execute(src_query).fetchall()
    src_ms = (time.time() - t0) / iterations * 1000

    print(f"  Optimized (10 IDs, {iterations} runs): {opt_ms:.2f} ms avg")
    print(f"  Original  (10 IDs, {iterations} runs): {src_ms:.2f} ms avg")
    ratio = opt_ms / src_ms if src_ms > 0 else float('inf')
    if ratio < 2.0:
        print(f"  PASS (ratio: {ratio:.2f}x)")
        passed += 1
    else:
        print(f"  WARN: optimized is {ratio:.1f}x slower than original")
        failed += 1

    # Also test scope-filtered queries
    t0 = time.time()
    for _ in range(iterations):
        opt.execute(
            f"{opt_query} AND m.has_url = true"
        ).fetchall()
    opt_scope_ms = (time.time() - t0) / iterations * 1000

    t0 = time.time()
    for _ in range(iterations):
        src.execute(
            f"{src_query} AND has_url = true"
        ).fetchall()
    src_scope_ms = (time.time() - t0) / iterations * 1000

    print(f"  Optimized scoped (url_only): {opt_scope_ms:.2f} ms avg")
    print(f"  Original  scoped (url_only): {src_scope_ms:.2f} ms avg")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"PASSED: {passed}  FAILED: {failed}")
    if failed == 0:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review above")

    src.close()
    opt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate optimized DuckDB")
    parser.add_argument("--source", required=True, help="Original metadata.duckdb")
    parser.add_argument("--optimized", required=True, help="Optimized metadata.duckdb")
    args = parser.parse_args()

    validate(args.source, args.optimized)


if __name__ == "__main__":
    main()
