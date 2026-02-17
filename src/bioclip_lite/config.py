"""Application configuration for BioCLIP Lite."""

import argparse
from dataclasses import dataclass


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

    # Image retrieval
    image_fetch_timeout: int = 10
    image_fetch_max_workers: int = 8
    thumbnail_max_dim: int = 256

    # Metadata columns to SELECT (15 of 18 — excludes resolution_status, basisOfRecord, scientific_name)
    METADATA_COLUMNS: str = (
        'id, uuid, kingdom, phylum, class, "order", family, genus, species, '
        "common_name, source_dataset, source_id, publisher, img_type, identifier, has_url"
    )


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
    args = p.parse_args()

    cfg = LiteConfig(
        faiss_index_path=args.faiss_index,
        duckdb_path=args.duckdb_path,
        device=args.device,
        scope=args.scope,
        host=args.host,
        port=args.port,
        enable_export=args.enable_export,
    )
    if args.model_str:
        cfg.model_str = args.model_str
    return cfg
