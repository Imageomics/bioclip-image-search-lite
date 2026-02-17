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
            scope: "all", "url_only", or "inaturalist".

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
        """Query DuckDB for metadata, applying scope filter."""
        id_list = ",".join(str(i) for i in ids)

        where = [f"id IN ({id_list})"]
        if scope == "url_only":
            where.append("has_url = true")
        elif scope == "inaturalist":
            where.append("has_url = true")
            where.append("source_dataset = 'gbif'")
            where.append("publisher LIKE '%iNaturalist%'")

        query = (
            f"SELECT {self.metadata_columns} FROM metadata "
            f"WHERE {' AND '.join(where)}"
        )
        rows = self.conn.execute(query).fetchall()
        col_names = [desc[0] for desc in self.conn.description]

        # Build lookup keyed by id
        meta_map: Dict[int, Dict] = {}
        for row in rows:
            d = dict(zip(col_names, row))
            meta_map[d["id"]] = d

        # Merge with distances, preserving FAISS ranking
        results = []
        for fid, dist in zip(ids, distances):
            if fid in meta_map:
                results.append({"distance": dist, **meta_map[fid]})
        return results

    @property
    def dimensions(self) -> int:
        return self.index.d

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def close(self):
        self.conn.close()
