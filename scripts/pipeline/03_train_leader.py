"""Phase 03: train IVF centroids + PQ codebooks on the stratified sample.

Output:
  ${output.workdir}/leader.index    FAISS factory index with trained
                                    centroids and codebooks but ntotal=0.

Reads ${output.workdir}/leader_sample.parquet/ via PyArrow with zero-copy
extraction of the embedding column, initializes the index from
cfg.index.factory, optionally moves it to GPU 0 if a GPU is available,
runs index.train(...), and writes the trained-but-empty index to disk.

The trained leader is reused by phase 04 (each shard worker loads the
leader, encodes its slice of vectors, and adds them).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    return [Path(cfg.output.workdir) / "leader.index"]


def _load_embeddings(sample_dir: Path, emb_col: str) -> np.ndarray:
    """Zero-copy extraction of embeddings as a (N, D) float32 array."""
    dataset = pa_ds.dataset(str(sample_dir), format="parquet")
    table = dataset.to_table(columns=[emb_col], use_threads=True)
    chunked = table[emb_col]

    # Cast to large_list to support >2GB total bytes.
    chunks = [c.cast(pa.large_list(pa.float32())) for c in chunked.chunks]
    flat = pa.concat_arrays(chunks)

    n_rows = len(flat)
    inner = len(flat[0])
    values = flat.values  # underlying float32 buffer
    arr = np.frombuffer(values.buffers()[1], dtype=np.float32)[:n_rows * inner]
    return arr.reshape(n_rows, inner)


def train_leader(cfg: PipelineConfig) -> None:
    import faiss

    sample_dir = Path(cfg.output.workdir) / "leader_sample.parquet"
    out_path = Path(cfg.output.workdir) / "leader.index"

    if not sample_dir.exists():
        raise SystemExit(f"missing input: {sample_dir} (run phase 02 first)")

    logger.info("loading sample from %s", sample_dir)
    t0 = time.time()
    embs = _load_embeddings(sample_dir, cfg.input.embedding_col)
    n, d = embs.shape
    logger.info("loaded %d vectors x %d dims in %.1fs", n, d, time.time() - t0)
    if d != cfg.input.embedding_dim:
        raise SystemExit(
            f"embedding_dim mismatch: config={cfg.input.embedding_dim} sample={d}"
        )

    logger.info("initializing index: %s", cfg.index.factory)
    index = faiss.index_factory(d, cfg.index.factory)

    n_gpus = faiss.get_num_gpus()
    use_gpu = n_gpus > 0
    if use_gpu:
        logger.info("moving index to GPU 0 (faiss reports %d GPU(s))", n_gpus)
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        logger.info("no GPU available; training on CPU")

    if cfg.index.l2_normalize:
        logger.info("L2-normalizing training set")
        faiss.normalize_L2(embs)

    logger.info("training on %d vectors...", n)
    t0 = time.time()
    index.train(embs)
    logger.info("training done in %.1fs", time.time() - t0)

    if use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, str(out_path))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    raise SystemExit(run_phase("03_train_leader", outputs, train_leader))
