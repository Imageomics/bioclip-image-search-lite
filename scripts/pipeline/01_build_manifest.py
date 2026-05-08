"""Phase 01: scan input parquet files, assign FAISS ids, emit a manifest.

Walks input.embeddings_parquet in lexicographic order and assigns sequential
global FAISS ids starting at 0. Every row across every file ends up with a
unique id in [0, total_rows).

Outputs:
  ${output.workdir}/manifest.parquet
    schema: file_path:string, start_id:int64, row_count:int64

  ${output.workdir}/uuid_to_id.parquet/
    one parquet per input file (filename mirrors input stem)
    schema: uuid:string, id:int64

Design notes:
  PyArrow only, no Spark. Row counts are gathered from parquet footers in
  parallel (no data read). UUID extraction reads only the uuid column.
  IDs are a serial prefix sum over per-file row counts so the assignment
  is deterministic.
"""
from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def _list_input_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet files matched: {pattern}")
    return files


def _count_rows(path: str) -> Tuple[str, int]:
    """Read only the parquet footer; no data is loaded."""
    return path, pq.ParquetFile(path).metadata.num_rows


def _read_uuids(args: Tuple[str, int, str]) -> Tuple[str, int, np.ndarray]:
    path, start_id, uuid_col = args
    t = pq.read_table(path, columns=[uuid_col], use_threads=False)
    uuids = t.column(uuid_col).to_numpy(zero_copy_only=False)
    return path, start_id, uuids


def outputs(cfg: PipelineConfig) -> List[Path]:
    workdir = Path(cfg.output.workdir)
    return [workdir / "manifest.parquet", workdir / "uuid_to_id.parquet"]


def build_manifest(cfg: PipelineConfig) -> None:
    workdir = Path(cfg.output.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    files = _list_input_files(cfg.input.embeddings_parquet)
    logger.info("found %d input parquet files", len(files))

    # Workers: use available CPUs (the loader doesn't expose a worker hint;
    # parquet footer reads are I/O-bound and benefit from oversubscription).
    workers = max(1, (os.cpu_count() or 8))

    # Phase 1: gather row counts.
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (path, n) in enumerate(pool.map(_count_rows, files)):
            counts[path] = n
            if (i + 1) % 200 == 0:
                logger.info("  counted %d/%d", i + 1, len(files))
    total = sum(counts[f] for f in files)
    logger.info("total rows across all files: %d", total)

    # Phase 2: write manifest.
    manifest_rows = []
    start = 0
    for f in files:
        manifest_rows.append({"file_path": f, "start_id": start, "row_count": counts[f]})
        start += counts[f]
    assert start == total

    manifest_path = workdir / "manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            manifest_rows,
            schema=pa.schema([
                ("file_path", pa.string()),
                ("start_id", pa.int64()),
                ("row_count", pa.int64()),
            ]),
        ),
        str(manifest_path),
        compression="zstd",
    )
    logger.info("wrote %s (%d rows)", manifest_path, len(manifest_rows))

    # Phase 3: build uuid_to_id mapping per file (parallel reads), emit one
    # parquet per input file to keep memory bounded. Final output is a
    # partitioned directory.
    uuid_dir = workdir / "uuid_to_id.parquet"
    uuid_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (f, manifest_rows[i]["start_id"], cfg.input.uuid_col)
        for i, f in enumerate(files)
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_read_uuids, t): t for t in tasks}
        for fut in as_completed(futures):
            path, start_id, uuids = fut.result()
            ids = np.arange(start_id, start_id + len(uuids), dtype=np.int64)
            table = pa.Table.from_arrays(
                [pa.array(uuids, type=pa.string()), pa.array(ids, type=pa.int64())],
                names=["uuid", "id"],
            )
            stem = os.path.splitext(os.path.basename(path))[0]
            out = uuid_dir / f"part-{stem}.parquet"
            pq.write_table(table, str(out), compression="zstd")
            done += 1
            if done % 200 == 0:
                logger.info("  wrote uuid_to_id  %d/%d", done, len(files))

    logger.info("wrote %s (%d files)", uuid_dir, len(files))
    logger.info("FAISS id range: [0, %d)", total)


if __name__ == "__main__":
    raise SystemExit(run_phase("01_build_manifest", outputs, build_manifest))
