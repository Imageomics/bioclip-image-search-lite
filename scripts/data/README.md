# Data Pipeline

Two-stage pipeline to build the optimized DuckDB metadata database from source.

## Pipeline

```
Source (SQLite or DuckDB)
  → convert_duckdb_lite.py   # Stage 1: import, add has_url + in_bioclip2_training
  → optimize_duckdb.py       # Stage 2: ENUM types, URL split, sort, index
  → validate_optimized_duckdb.py  # Verify correctness
```

## Optimized Schema

**Table:** `metadata` — 234,391,308 rows

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER` | FAISS vector index (downcast from BIGINT) |
| `uuid` | `UUID` | Native 16-byte UUID (normalized hyphenated format) |
| `kingdom`..`family` | `ENUM` | Low-cardinality taxonomy columns as ENUM types |
| `genus`, `species` | `VARCHAR` | Too many distinct values for ENUM |
| `common_name` | `VARCHAR` | |
| `source_dataset` | `ENUM` | `gbif`, `eol`, `bioscan`, `fathomnet` |
| `publisher` | `ENUM` | GBIF publisher (e.g., `iNaturalist`, `observation.org`) |
| `img_type` | `ENUM` | Image type category |
| `basisOfRecord` | `ENUM` | GBIF basis of record |
| `source_id` | `VARCHAR` | Source-specific identifier |
| `url_prefix_id` | `USMALLINT` | FK to `url_prefixes` table |
| `identifier_suffix` | `VARCHAR` | URL path after domain prefix |
| `has_url` | `BOOLEAN` | `TRUE` if image URL available |
| `in_bioclip2_training` | `BOOLEAN` | `TRUE` if UUID in BioCLIP 2 training catalog |

**Indexes:** `idx_id(id)`, `idx_scope(source_dataset, has_url, in_bioclip2_training)`

## Optimizations Applied

1. **ENUM types** — Low-cardinality columns (`kingdom`, `phylum`, `class`, `order`, `family`, `source_dataset`, `publisher`, `img_type`, `basisOfRecord`) stored as ENUM for ~10x compression.
2. **URL prefix deduplication** — `identifier` split into a shared prefix table (`url_prefixes`) + per-row suffix, eliminating repeated domain strings.
3. **Taxonomy sort** — Rows sorted by `source_dataset, kingdom, ..., species, common_name` for long runs of identical values and better compression.
4. **Type downcasting** — `id` BIGINT→INTEGER, `uuid` VARCHAR→native UUID (16 bytes).
5. **Corruption cleanup** — 44 rows with column-shift metadata corruption have taxonomy NULLed.

Result: **80 GB (SQLite) → 14 GB (optimized DuckDB)**, 57% smaller than the unoptimized DuckDB.

## Usage

```bash
# Stage 1: Import + add boolean columns
python scripts/data/convert_duckdb_lite.py \
    --from-duckdb /path/to/source.duckdb \
    --output /path/to/base.duckdb \
    --catalog-parquet /path/to/training/catalog.parquet

# Stage 2: Optimize
python scripts/data/optimize_duckdb.py \
    --source /path/to/base.duckdb \
    --output /path/to/metadata_optimized.duckdb

# Validate
python scripts/data/validate_optimized_duckdb.py \
    --source /path/to/base.duckdb \
    --optimized /path/to/metadata_optimized.duckdb
```
