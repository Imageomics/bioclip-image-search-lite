"""Common scaffolding for phase scripts.

Phase scripts in this directory should call run_phase() with their phase id,
expected output paths (so idempotent reruns work), and a callback that does
the actual work. The callback receives the validated PipelineConfig.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, List

from scripts.lib.config import PipelineConfig, load_config

logger = logging.getLogger("pipeline")


def run_phase(
    phase_id: str,
    output_paths: Callable[[PipelineConfig], List[Path]],
    work: Callable[[PipelineConfig], None],
    argv: List[str] | None = None,
) -> int:
    """Standard entry point.

    Args:
        phase_id: e.g. "01_build_manifest". Logged + used in error messages.
        output_paths: 
            callable returning the outputs the phase produces (used for the idempotency check).
            The callable takes PipelineConfig as an argument, 
            and returns a list of Paths that the phase is expected to produce.
            example: 
            ```py
            def output_paths(cfg: PipelineConfig) -> List[Path]:
                return [cfg.manifest_path]
            ```
        work: callable that does the real work. Receives the config.
        argv: optional CLI argv override (for tests).
    """
    parser = argparse.ArgumentParser(description=phase_id)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Phase %s: loading config %s", phase_id, args.config)

    cfg = load_config(args.config)

    outs = output_paths(cfg)

    def _present(p: Path) -> bool:
        # Empty directories from crashed prior runs do not count as present.
        if p.is_dir():
            return any(p.iterdir())
        return p.exists()

    # SLURM array tasks each contribute a subset to a shared output
    # directory, so the parent-level "outputs exist" check would race and
    # cause tasks 1..N to skip after task 0 writes its first artifact.
    # Bypass the idempotency check entirely when running under an array.
    is_array_task = "SLURM_ARRAY_TASK_ID" in os.environ
    if args.force:
        # Clear stale outputs before rebuilding so per-file artifacts
        # from a prior run (e.g. shards named by start_id, uuid_to_id
        # partitions named by input filename) cannot mix with fresh ones.
        # Skip when running as an array task: tasks must not delete each
        # other's siblings in the shared output directory.
        if not is_array_task:
            for p in outs:
                if p.is_dir():
                    logger.info("Phase %s: --force, removing %s", phase_id, p)
                    shutil.rmtree(p)
                elif p.is_file():
                    logger.info("Phase %s: --force, removing %s", phase_id, p)
                    p.unlink()
    elif not is_array_task and all(_present(p) for p in outs):
        logger.info("Phase %s: outputs already exist, skipping. Use --force to rebuild.", phase_id)
        return 0

    work(cfg)
    logger.info("Phase %s: done.", phase_id)
    return 0
