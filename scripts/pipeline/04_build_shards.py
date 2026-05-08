"""Phase 04: encode all input vectors and assign them to FAISS shards.

For each input parquet listed in the manifest:
  1. Stream the embedding column via PyArrow zero-copy.
  2. Buffer vectors until shards.batch_size is reached.
  3. Load the leader, transfer to GPU 0 (if available), L2-normalize the
     buffer (when index.l2_normalize is true), call add_with_ids using
     contiguous global ids from the manifest, and write one shard.
  4. Repeat for the next batch.

Output:
  ${output.workdir}/shards/shard_{start_id}.index   one per flushed batch

Slicing:
  - Under SLURM array execution, $SLURM_ARRAY_TASK_ID picks one slice of
    the manifest and the script processes that slice only.
  - Without SLURM (local run), the script loops over all shards.n_shards
    slices sequentially.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    # Phase 04 produces a directory with N shard files; we can't predict the
    # exact filenames (they include the start_id of each flush). The
    # _phase_stub idempotency check treats a non-empty directory as present.
    return [Path(cfg.output.workdir) / "shards"]


def _read_emb_zero_copy(path: str, emb_col: str) -> np.ndarray:
    table = pq.read_table(path, columns=[emb_col], use_threads=True)
    chunks = [c.cast(pa.large_list(pa.float32())) for c in table[emb_col].chunks]
    flat = pa.concat_arrays(chunks)
    n = len(flat)
    inner = len(flat[0])
    arr = np.frombuffer(flat.values.buffers()[1], dtype=np.float32)[:n * inner]
    return arr.reshape(n, inner)


def _process_batch(
    leader_path: Path,
    out_dir: Path,
    buffer: List[np.ndarray],
    start_id: int,
    l2_normalize: bool,
    use_gpu: bool,
) -> None:
    import faiss

    if not buffer:
        return
    t0 = time.time()
    vectors = np.vstack(buffer)
    n = vectors.shape[0]
    ids = np.arange(start_id, start_id + n, dtype=np.int64)

    cpu_leader = faiss.read_index(str(leader_path))
    if use_gpu:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_leader)
    else:
        gpu_index = cpu_leader

    if l2_normalize:
        faiss.normalize_L2(vectors)
    gpu_index.add_with_ids(vectors, ids)

    out = out_dir / f"shard_{start_id:013d}.index"
    if use_gpu:
        cpu_result = faiss.index_gpu_to_cpu(gpu_index)
    else:
        cpu_result = gpu_index
    faiss.write_index(cpu_result, str(out))
    logger.info(
        "wrote %s (%d vectors, ids %d..%d, %.1fs)",
        out, n, start_id, start_id + n - 1, time.time() - t0,
    )

    del vectors, ids, cpu_leader, gpu_index, cpu_result
    if use_gpu:
        del res
    gc.collect()


def build_shards(cfg: PipelineConfig) -> None:
    import faiss

    workdir = Path(cfg.output.workdir)
    leader_path = workdir / "leader.index"
    manifest_path = workdir / "manifest.parquet"
    shards_dir = workdir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    if not leader_path.exists():
        raise SystemExit(f"missing input: {leader_path} (run phase 03 first)")
    if not manifest_path.exists():
        raise SystemExit(f"missing input: {manifest_path} (run phase 01 first)")

    manifest = pq.read_table(str(manifest_path)).to_pylist()
    n_files = len(manifest)
    n_shards = cfg.shards.n_shards
    chunk = (n_files + n_shards - 1) // n_shards

    use_gpu = faiss.get_num_gpus() > 0
    logger.info("faiss reports %d GPU(s); use_gpu=%s", faiss.get_num_gpus(), use_gpu)

    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_ids = [int(array_task)] if array_task is not None else list(range(n_shards))
    logger.info("processing tasks: %s of %d", task_ids, n_shards)

    for task_id in task_ids:
        start_idx = task_id * chunk
        end_idx = min(start_idx + chunk, n_files)
        if start_idx >= end_idx:
            logger.info("task %d: empty slice, skipping", task_id)
            continue

        logger.info("task %d: manifest rows [%d, %d)", task_id, start_idx, end_idx)

        buffer: List[np.ndarray] = []
        buffer_count = 0
        buffer_start_id = -1

        for entry in manifest[start_idx:end_idx]:
            file_path = entry["file_path"]
            file_start = entry["start_id"]
            file_count = entry["row_count"]
            logger.info("  reading %s (%d rows)", os.path.basename(file_path), file_count)

            arr = _read_emb_zero_copy(file_path, cfg.input.embedding_col)
            assert arr.shape[0] == file_count, (
                f"row count mismatch: manifest={file_count} actual={arr.shape[0]}"
            )

            if buffer_count == 0:
                buffer_start_id = file_start
            buffer.append(arr)
            buffer_count += file_count

            if buffer_count >= cfg.shards.batch_size:
                _process_batch(
                    leader_path, shards_dir, buffer, buffer_start_id,
                    cfg.index.l2_normalize, use_gpu,
                )
                buffer = []
                buffer_count = 0
                buffer_start_id = -1

        if buffer_count > 0:
            _process_batch(
                leader_path, shards_dir, buffer, buffer_start_id,
                cfg.index.l2_normalize, use_gpu,
            )


if __name__ == "__main__":
    raise SystemExit(run_phase("04_build_shards", outputs, build_shards))
