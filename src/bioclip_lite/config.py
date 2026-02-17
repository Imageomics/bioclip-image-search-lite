"""Application configuration and logging setup for BioCLIP Lite."""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class LiteConfig:
    # Data paths
    faiss_index_path: str = ""
    duckdb_path: str = ""

    # Model
    model_str: str = "hf-hub:imageomics/bioclip-2"
    device: str = "cpu"

    # Search
    default_top_n: int = 10
    default_nprobe: int = 16
    over_fetch_factor: int = 3

    # Scope: "all" | "url_only" | "inaturalist"
    scope: str = "all"

    # Server
    host: str = "0.0.0.0"
    port: int = 7860

    # Feature flags
    enable_export: bool = False

    # Logging
    log_dir: Optional[str] = None
    log_level: str = "INFO"

    # Image retrieval
    image_fetch_timeout: int = 10
    image_fetch_max_workers: int = 8
    thumbnail_max_dim: int = 256

    # Metadata columns to SELECT (15 of 18 — excludes resolution_status, basisOfRecord, scientific_name)
    METADATA_COLUMNS: str = (
        'id, uuid, kingdom, phylum, class, "order", family, genus, species, '
        "common_name, source_dataset, source_id, publisher, img_type, identifier, has_url"
    )


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: LiteConfig) -> None:
    """Configure logging for the application.

    Sets up two handlers:
      - Console (stderr): Always active, shows INFO+ by default.
      - File: Written to log_dir/app_<timestamp>.log if log_dir is set.
        Captures DEBUG+ for post-hoc diagnosis.

    All modules under bioclip_lite.* inherit the root logger config, so
    every logger.info() / logger.warning() / etc. call in any service
    automatically goes to both handlers.
    """
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture everything; handlers filter.

    # ── Console handler ─────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (optional) ─────────────────────────────────────
    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(config.log_dir, f"app_{ts}.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root.addHandler(fh)
        root.info(f"File logging enabled: {log_path}")

    # Quiet down noisy third-party loggers
    for name in ("httpx", "httpcore", "urllib3", "gradio", "uvicorn"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args() -> LiteConfig:
    """Parse CLI arguments into a LiteConfig."""
    p = argparse.ArgumentParser(description="BioCLIP Lite Image Search")
    p.add_argument("--faiss-index", required=True, help="Path to FAISS index file")
    p.add_argument("--duckdb-path", required=True, help="Path to DuckDB metadata file")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--model-str", default=None, help="Model identifier")
    p.add_argument("--scope", default="all", choices=["all", "url_only", "inaturalist"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--enable-export", action="store_true")
    p.add_argument(
        "--log-dir", default=None,
        help="Directory for log files. If set, writes DEBUG-level logs to a timestamped file.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
        help="Console log level (default: INFO). File logs are always DEBUG.",
    )
    args = p.parse_args()

    cfg = LiteConfig(
        faiss_index_path=args.faiss_index,
        duckdb_path=args.duckdb_path,
        device=args.device,
        scope=args.scope,
        host=args.host,
        port=args.port,
        enable_export=args.enable_export,
        log_dir=args.log_dir,
        log_level=args.log_level,
    )
    if args.model_str:
        cfg.model_str = args.model_str
    return cfg
