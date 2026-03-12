"""FAISS vector search + DuckDB metadata lookup.

Adapted from bioclip-vector-db query_monolithic/neighborhood_server.py.
Replaces SQLite with DuckDB and adds scope filtering.
"""

import logging
import time
import functools
from typing import List, Dict, Any, Optional

import duckdb
import faiss
import numpy as np

logger = logging.getLogger(__name__)

SCOPE_MAP = {
    "All Sources": "all",
    "URL-Available Only": "url_only",
    "iNaturalist Only": "inaturalist",
    "BioCLIP 2 Training": "bioclip2_training",
}


def _timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.info(f"{func.__name__} completed in {dt:.4f}s")
        return result
    return wrapper


class SearchService:
    """In-process FAISS search with DuckDB metadata."""

    def __init__(
        self,
        faiss_index_path: str,
        duckdb_path: str,
        nprobe: int = 16,
        over_fetch_factor: int = 3,
        metadata_columns: str = "*",
    ):
        self.over_fetch_factor = over_fetch_factor
        self.metadata_columns = metadata_columns

        # Load FAISS index
        logger.info(f"Loading FAISS index from {faiss_index_path}")
        self.index = faiss.read_index(faiss_index_path)
        self.index.nprobe = nprobe
        logger.info(
            f"FAISS index loaded: {self.index.ntotal:,} vectors, "
            f"{self.index.d} dims, nprobe={nprobe}"
        )

        # Persistent DuckDB connection (read-only)
        logger.info(f"Connecting to DuckDB at {duckdb_path}")
        self.conn = duckdb.connect(duckdb_path, read_only=True)
        row_count = self.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        logger.info(f"DuckDB connected: {row_count:,} rows")

        # Load URL prefix lookup (410 entries, ~50 KB in memory).
        # Reconstructs full URLs in Python instead of a SQL JOIN.
        self._url_prefixes = self._load_url_prefixes()

    @_timer
    def search(
        self,
        query_vector: np.ndarray,
        top_n: int = 10,
        nprobe: int = 16,
        scope: str = "all",
    ) -> List[Dict[str, Any]]:
        """Run FAISS search and return metadata-enriched results.

        Args:
            query_vector: 1-D embedding vector (768-dim for BioCLIP-2).
            top_n: Number of results to return after scope filtering.
            nprobe: Number of IVF partitions to search.
            scope: "all", "url_only", "inaturalist", or "bioclip2_training".

        Returns:
            List of result dicts ordered by distance, each containing
            metadata fields plus 'distance'.
        """
        # Resolve scope label to key
        scope = SCOPE_MAP.get(scope, scope)

        if nprobe != self.index.nprobe:
            self.index.nprobe = nprobe

        # Prepare query
        queries = np.asarray(query_vector, dtype="float32")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self.index.d:
            raise ValueError(
                f"Dimension mismatch: query has {queries.shape[1]}, "
                f"index expects {self.index.d}"
            )

        faiss.normalize_L2(queries)

        # Over-fetch to compensate for scope filtering
        fetch_n = top_n * self.over_fetch_factor
        distances, indices = self.index.search(queries, fetch_n)

        # Collect valid IDs
        ids = []
        dists = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                ids.append(int(idx))
                dists.append(float(dist))

        if not ids:
            return []

        # DuckDB metadata lookup with scope filter
        results = self._query_metadata(ids, dists, scope)
        return results[:top_n]

    def _query_metadata(
        self,
        ids: List[int],
        distances: List[float],
        scope: str,
    ) -> List[Dict[str, Any]]:
        """Query DuckDB for metadata, filtering by scope in Python.

        Scope filtering via SQL WHERE clauses causes ~370x slowdown on
        ID-based lookups (4ms → 1600ms) because DuckDB scans the full
        column even when nearly all rows match. Since has_url and
        in_bioclip2_training are true for >87% of rows, post-filtering
        in Python is far more efficient.
        """
        id_list = ",".join(str(i) for i in ids)

        query = (
            f"SELECT {self.metadata_columns} FROM metadata "
            f"WHERE id IN ({id_list})"
        )
        rows = self.conn.execute(query).fetchall()
        col_names = [desc[0] for desc in self.conn.description]

        # Build lookup keyed by id, reconstructing full URL from prefix + suffix
        meta_map: Dict[int, Dict] = {}
        for row in rows:
            d = dict(zip(col_names, row))
            if self._url_prefixes and "url_prefix_id" in d:
                prefix = self._url_prefixes.get(d.pop("url_prefix_id"), "")
                suffix = d.pop("identifier_suffix", "") or ""
                d["identifier"] = prefix + suffix if (prefix or suffix) else None
            meta_map[d["id"]] = d

        # Merge with distances, preserving FAISS ranking
        results = []
        for fid, dist in zip(ids, distances):
            if fid in meta_map:
                results.append({"distance": dist, **meta_map[fid]})

        # Apply scope filter in Python (much faster than SQL WHERE)
        if scope == "url_only":
            results = [r for r in results if r.get("has_url")]
        elif scope == "inaturalist":
            results = [
                r for r in results
                if r.get("has_url")
                and r.get("source_dataset") == "gbif"
                and "iNaturalist" in (r.get("publisher") or "")
            ]
        elif scope == "bioclip2_training":
            results = [r for r in results if r.get("in_bioclip2_training")]

        return results

    @property
    def dimensions(self) -> int:
        return self.index.d

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def _load_url_prefixes(self) -> Dict[int, str]:
        """Load url_prefixes table into a dict for fast in-Python URL reconstruction."""
        try:
            rows = self.conn.execute(
                "SELECT prefix_id, prefix FROM url_prefixes"
            ).fetchall()
            prefixes = {row[0]: row[1] for row in rows}
            logger.info(f"Loaded {len(prefixes)} URL prefixes")
            return prefixes
        except duckdb.CatalogException:
            # Legacy DB without url_prefixes table — identifier is a direct column
            logger.info("No url_prefixes table found, using direct identifier column")
            return {}

    def close(self):
        self.conn.close()
