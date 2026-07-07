"""Phase 05: merge per-shard FAISS indices into the final index.

Inverted-list-level merge via faiss.IndexIVF.merge_from. CPU-bound.

Inputs:
  ${output.workdir}/shards/shard_*.index    one or more shard files
                                            from phase 04

Output:
  ${output.index_path}    (default: ${output.workdir}/index.index)

Shards are merged in lexicographic filename order, which (because shard
filenames embed the zero-padded global start_id) coincides with id order.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path
from typing import List

# Allow running as `python scripts/pipeline/NN_*.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.lib.config import PipelineConfig  # noqa: E402
from scripts.pipeline._phase_stub import logger, run_phase  # noqa: E402


def outputs(cfg: PipelineConfig) -> List[Path]:
    return [Path(cfg.output.index_path)]


def merge_shards(cfg: PipelineConfig) -> None:
    import faiss

    shards_dir = Path(cfg.output.workdir) / "shards"
    out_path = Path(cfg.output.index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(str(shards_dir / "shard_*.index")))
    if not shards:
        raise SystemExit(f"no shards found in {shards_dir} (run phase 04 first)")
    logger.info("merging %d shards", len(shards))

    t0 = time.time()
    logger.info("loading master shard: %s", Path(shards[0]).name)
    master = faiss.read_index(shards[0])

    for i, sh in enumerate(shards[1:], start=1):
        logger.info("[%d/%d] merging %s", i, len(shards) - 1, Path(sh).name)
        part = faiss.read_index(sh)
        master.merge_from(part, 0)

    logger.info("merged: ntotal=%d  (%.1fs)", master.ntotal, time.time() - t0)

    t0 = time.time()
    logger.info("writing %s", out_path)
    faiss.write_index(master, str(out_path))
    logger.info("write done in %.1fs (%.2f GB)", time.time() - t0, out_path.stat().st_size / 1e9)


if __name__ == "__main__":
    raise SystemExit(run_phase("05_merge_shards", outputs, merge_shards))
