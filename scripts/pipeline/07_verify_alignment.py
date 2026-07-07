"""Phase 07: verify the FAISS index and DuckDB lookup are aligned.

Hard-fails on:
  - faiss.ntotal != duckdb_row_count
  - any of verify.n_roundtrip_samples random uuids fails its round-trip

Round-trip: pick a random uuid from the catalog, fetch its embedding from
the input parquets, L2-normalize (if cfg.index.l2_normalize), search FAISS
for top-K (K=verify.roundtrip_topk), and check the queried uuid appears in
the returned top-K. Top-K (rather than strict top-1) is used because IVFPQ
is approximate: a near-duplicate at very high cosine similarity can win
rank-1 by a hair under PQ-quantized distance, which is benign. A real
alignment bug (broken uuid->id mapping) would never put the queried uuid
in any of the top-K positions.

Output:
  ${output.workdir}/verify.done.json  small JSON with timestamp + counts
                                      for downstream auditing
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import duckdb
import numpy as np
import pyarrow.parquet as pq

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    return [Path(cfg.output.workdir) / "verify.done.json"]


def verify_alignment(cfg: PipelineConfig) -> None:
    import faiss

    workdir = Path(cfg.output.workdir)
    index_path = Path(cfg.output.index_path)
    duckdb_path = Path(cfg.output.duckdb_path)

    # ── Counts ────────────────────────────────────────────────────────
    logger.info("loading FAISS index: %s", index_path)
    index = faiss.read_index(str(index_path))
    n_faiss = index.ntotal
    logger.info("faiss ntotal: %d", n_faiss)

    con = duckdb.connect(str(duckdb_path), read_only=True)
    n_duckdb = con.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    logger.info("duckdb rows: %d", n_duckdb)

    if n_faiss != n_duckdb:
        raise SystemExit(f"COUNT MISMATCH: faiss={n_faiss} duckdb={n_duckdb}")

    # ── Round-trip ────────────────────────────────────────────────────
    n_samples = cfg.verify.n_roundtrip_samples
    logger.info("round-trip on %d random uuids", n_samples)

    sample_uuids = [
        r[0] for r in con.execute(
            "SELECT CAST(uuid AS VARCHAR) FROM metadata "
            f"USING SAMPLE {n_samples} ROWS"
        ).fetchall()
    ]
    if len(sample_uuids) < n_samples:
        logger.warning(
            "only %d uuids available (asked for %d)", len(sample_uuids), n_samples,
        )

    expected_id_by_uuid = {
        r[0]: r[1] for r in con.execute(
            f"SELECT CAST(uuid AS VARCHAR), id FROM metadata "
            f"WHERE CAST(uuid AS VARCHAR) IN "
            f"({','.join(repr(u) for u in sample_uuids)})"
        ).fetchall()
    }

    # Fetch the embeddings for those uuids from the input parquets.
    # Single DuckDB scan over the embedding glob, projecting only
    # uuid + emb for the sample set.
    uuid_in = ",".join(repr(u) for u in sample_uuids)
    rows = con.execute(f"""
        SELECT {cfg.input.uuid_col}::VARCHAR AS uuid, {cfg.input.embedding_col} AS emb
        FROM read_parquet('{cfg.input.embeddings_parquet}')
        WHERE {cfg.input.uuid_col}::VARCHAR IN ({uuid_in})
    """).fetchall()

    if len(rows) != len(sample_uuids):
        raise SystemExit(
            f"could not fetch all sampled embeddings: "
            f"asked {len(sample_uuids)}, got {len(rows)}"
        )

    sample_emb = np.array([r[1] for r in rows], dtype=np.float32)
    if cfg.index.l2_normalize:
        faiss.normalize_L2(sample_emb)

    # Search top-K rather than top-1 to tolerate PQ-quantized rank shuffles.
    # The queried vector IS in the index, but with PQ16 a near-duplicate at
    # cosine ~0.96 can win rank-1 by a hair. Self-membership in top-K is the
    # correct alignment check for an approximate index.
    K = cfg.verify.roundtrip_topk
    _, returned_ids = index.search(sample_emb, K)  # shape (n_samples, K)

    # Look up uuids for every returned id (flatten + dedup the int list).
    flat_ids = sorted({int(i) for row in returned_ids for i in row if i >= 0})
    rid_list = ",".join(str(i) for i in flat_ids)
    rid_to_uuid = {
        r[0]: r[1] for r in con.execute(
            f"SELECT id, CAST(uuid AS VARCHAR) FROM metadata WHERE id IN ({rid_list})"
        ).fetchall()
    } if flat_ids else {}

    matched = mismatched = unfound = 0
    for (queried_uuid, _), rids in zip(rows, returned_ids):
        topk_uuids = [rid_to_uuid.get(int(rid)) for rid in rids if rid >= 0]
        if not topk_uuids:
            unfound += 1
            continue
        if queried_uuid in topk_uuids:
            matched += 1
        else:
            mismatched += 1
            # The queried uuid wasn't in any of top-K results; this implies
            # a real alignment problem (a working index would put self in
            # top-K even with PQ noise).
            logger.error(
                "MISMATCH: queried uuid %s not in top-%d returned: %s",
                queried_uuid, K, topk_uuids,
            )
            if mismatched >= 5:
                break

    logger.info(
        "round-trip (top-%d self-membership): matched=%d mismatched=%d unfound=%d",
        K, matched, mismatched, unfound,
    )
    if mismatched > 0 or unfound > 0:
        raise SystemExit(f"alignment broken: {mismatched} mismatched, {unfound} unfound")

    # ── Done marker ──────────────────────────────────────────────────
    done = {
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_faiss":         n_faiss,
        "n_duckdb":        n_duckdb,
        "n_roundtrip":     matched,
        "roundtrip_topk":  K,
        "index_path":      str(index_path),
        "duckdb_path":     str(duckdb_path),
    }
    out_path = workdir / "verify.done.json"
    out_path.write_text(json.dumps(done, indent=2))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    raise SystemExit(run_phase("07_verify_alignment", outputs, verify_alignment))
