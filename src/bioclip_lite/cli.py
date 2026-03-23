"""CLI entry point for bioclip-search.

Provides a pipe-friendly command-line interface to the BioCLIP 2 image search
over 234M biological images. Reuses ModelService and SearchService from the
existing web application.

Usage:
    bioclip-search photo.jpg
    bioclip-search photo.jpg --top-n 50 --scope inaturalist --format table
    bioclip-search photo.jpg --format csv --output results.csv
    bioclip-search serve
    bioclip-search stop
    bioclip-search status
    bioclip-search config --show
    bioclip-search config --set faiss_index /path/to/index.index
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants (shared with server.py — server imports these)
# ------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".bioclip-search"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATA_DIR = CONFIG_DIR / "data"
SERVER_INFO_FILE = CONFIG_DIR / "server.json"

DEFAULT_PORT = 7863
DEFAULT_IDLE_TIMEOUT = 30  # minutes

# HuggingFace repo for auto-download
HF_DATA_REPO = "imageomics/bioclip-image-search-lite"
HF_FAISS_PATH = "faiss/index.index"
HF_DUCKDB_PATH = "duckdb/metadata.duckdb"

VALID_CONFIG_KEYS = {
    "faiss_index", "duckdb_path", "device",
    "auto_start", "idle_timeout", "port",
}

SCOPE_DESCRIPTIONS = {
    "all": "All Sources (234M images)",
    "url_only": "URL-Available Only",
    "inaturalist": "iNaturalist Only (135M images)",
    "bioclip2_training": "BioCLIP 2 Training (206M images)",
}


# ------------------------------------------------------------------
# Config file management
# ------------------------------------------------------------------


def load_config() -> Dict[str, str]:
    """Load persistent config from ~/.bioclip-search/config.json."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(config: Dict[str, str]) -> None:
    """Save config to ~/.bioclip-search/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")


def _config_bool(value) -> bool:
    """Parse a config value as boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(cli_flag: Optional[str]) -> str:
    """Resolve device with priority: CLI flag > config > auto-detect."""
    if cli_flag:
        return cli_flag
    config_device = load_config().get("device")
    if config_device:
        return config_device
    return _detect_device()


def handle_config(args: argparse.Namespace) -> None:
    """Handle the 'config' subcommand."""
    config = load_config()

    if args.config_show:
        if not config:
            _eprint("No configuration set. Run a search to trigger setup.")
            return
        for key, value in sorted(config.items()):
            _eprint(f"  {key}: {value}")
        return

    if args.config_set:
        key, value = args.config_set
        if key not in VALID_CONFIG_KEYS:
            _eprint(
                f"Unknown config key: {key}\n"
                f"Valid keys: {', '.join(sorted(VALID_CONFIG_KEYS))}"
            )
            sys.exit(1)
        config[key] = value
        save_config(config)
        _eprint(f"Set {key} = {value}")
        return

    # Default: show config
    if not config:
        _eprint("No configuration set. Run a search to trigger setup.")
    else:
        for key, value in sorted(config.items()):
            _eprint(f"  {key}: {value}")


# ------------------------------------------------------------------
# Data path resolution
# ------------------------------------------------------------------


def resolve_data_paths(
    faiss_flag: Optional[str],
    duckdb_flag: Optional[str],
) -> tuple[str, str]:
    """Resolve data file paths with priority: CLI flags > config > download.

    Returns:
        (faiss_index_path, duckdb_path)
    """
    config = load_config()

    faiss_path = faiss_flag or config.get("faiss_index") or ""
    duckdb_path = duckdb_flag or config.get("duckdb_path") or ""

    # If both paths are resolved (from flags or config), trust them
    if faiss_path and duckdb_path:
        return faiss_path, duckdb_path

    # Check default data directory for any missing paths
    default_faiss = DATA_DIR / "index.index"
    default_duckdb = DATA_DIR / "metadata.duckdb"

    if not faiss_path and default_faiss.is_file():
        faiss_path = str(default_faiss)
    if not duckdb_path and default_duckdb.is_file():
        duckdb_path = str(default_duckdb)

    if faiss_path and duckdb_path:
        return faiss_path, duckdb_path

    # Prompt user to download missing files
    missing = []
    if not faiss_path:
        missing.append("FAISS index     (~5.8 GB)  - 234M BioCLIP 2 image embeddings")
    if not duckdb_path:
        missing.append("DuckDB metadata  (~14 GB)  - taxonomy and source metadata")

    _eprint("\nbioclip-search: data files not found.\n")
    _eprint("The following files are required:")
    for m in missing:
        _eprint(f"  {m}")
    _eprint(f"\nDownload to {DATA_DIR}? [Y/n] ", end="")

    try:
        response = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        _eprint("\nAborted.")
        sys.exit(1)

    if response and response not in ("y", "yes"):
        _eprint(
            "Aborted. Provide paths manually with --faiss-index and --duckdb-path,\n"
            "or set them permanently with: bioclip-search config --set <key> <path>"
        )
        sys.exit(1)

    faiss_path, duckdb_path = _download_data(
        need_faiss=not faiss_path,
        need_duckdb=not duckdb_path,
    )

    # Save to config for future runs
    config = load_config()
    config["faiss_index"] = faiss_path
    config["duckdb_path"] = duckdb_path
    save_config(config)
    _eprint(f"Paths saved to {CONFIG_FILE}\n")

    return faiss_path, duckdb_path


def _download_data(need_faiss: bool, need_duckdb: bool) -> tuple[str, str]:
    """Download data files from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss_path = ""
    duckdb_path = ""

    if need_faiss:
        _eprint(f"\nDownloading FAISS index from {HF_DATA_REPO}...")
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=HF_FAISS_PATH,
            local_dir=str(DATA_DIR),
        )
        faiss_path = str(downloaded)
        _eprint(f"  Saved to {faiss_path}")

    if need_duckdb:
        _eprint(f"\nDownloading DuckDB metadata from {HF_DATA_REPO}...")
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=HF_DUCKDB_PATH,
            local_dir=str(DATA_DIR),
        )
        duckdb_path = str(downloaded)
        _eprint(f"  Saved to {duckdb_path}")

    return faiss_path, duckdb_path


# ------------------------------------------------------------------
# Server communication
# ------------------------------------------------------------------


def _get_server_info() -> Optional[Dict]:
    """Read server info and verify the process is alive."""
    if not SERVER_INFO_FILE.exists():
        return None
    try:
        info = json.loads(SERVER_INFO_FILE.read_text())
        pid = info.get("pid")
        if pid:
            # On Windows, os.kill(pid, 0) raises OSError; use ctypes instead
            if os.name == "nt":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
                if handle:
                    kernel32.CloseHandle(handle)
                    return info
                # Process not found — fall through to cleanup
            else:
                os.kill(pid, 0)  # check if alive (signal 0)
                return info
    except (json.JSONDecodeError, OSError, ProcessLookupError, SystemError):
        # Stale PID file — clean up
        if SERVER_INFO_FILE.exists():
            SERVER_INFO_FILE.unlink(missing_ok=True)
    return None


def _server_request(
    port: int,
    method: str,
    path: str,
    body: Optional[Dict] = None,
    timeout: int = 300,
) -> Optional[Dict]:
    """Send an HTTP request to the local search server."""
    url = f"http://127.0.0.1:{port}{path}"

    if body is not None:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return None


def _wait_for_server(port: int, timeout: int = 180) -> bool:
    """Poll /status until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        resp = _server_request(port, "GET", "/status", timeout=5)
        if resp and resp.get("status") == "running":
            return True
        time.sleep(2)
    return False


def _auto_start_server(faiss_flag: Optional[str], duckdb_flag: Optional[str]) -> int:
    """Resolve data paths, then spawn server as a background process.

    Returns the port the server will listen on.
    """
    # Ensure data paths are resolved and saved to config
    resolve_data_paths(faiss_flag, duckdb_flag)

    config = load_config()
    port = int(config.get("port", DEFAULT_PORT))

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CONFIG_DIR / "server.log"

    idle_timeout = int(config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
    _eprint(
        f"Starting server in background "
        f"(idle timeout: {idle_timeout}m)..."
    )
    with open(log_path, "a") as log_file:
        subprocess.Popen(
            [sys.executable, "-m", "bioclip_lite.server", "--port", str(port)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    if not _wait_for_server(port):
        _eprint(f"Error: server failed to start. Check {log_path}")
        sys.exit(1)

    _eprint("Server ready.")
    return port


# ------------------------------------------------------------------
# Commands: serve, stop, status
# ------------------------------------------------------------------


def handle_serve(args: argparse.Namespace) -> None:
    """Start search server in foreground."""
    from bioclip_lite.server import run_server

    faiss_path, duckdb_path = resolve_data_paths(args.faiss_index, args.duckdb_path)
    config = load_config()
    device = resolve_device(args.device)
    idle_timeout = int(config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
    port = int(config.get("port", DEFAULT_PORT))

    # Set up logging for foreground serve (overrides the WARNING-only default)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CONFIG_DIR / "server.log"
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file),
        ],
    )
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    run_server(port, faiss_path, duckdb_path, device, idle_timeout)


def handle_stop(args: argparse.Namespace) -> None:
    """Stop the running server."""
    info = _get_server_info()
    if not info:
        _eprint("Server is not running.")
        return

    port = info["port"]
    resp = _server_request(port, "POST", "/stop")
    if resp and resp.get("status") == "stopping":
        _eprint(f"Server (PID {info['pid']}) stopping.")
    else:
        _eprint("Failed to reach server. It may have already stopped.")
        # Clean up stale info file
        SERVER_INFO_FILE.unlink(missing_ok=True)


def handle_status(args: argparse.Namespace) -> None:
    """Show server status."""
    info = _get_server_info()
    if not info:
        _eprint("Server is not running.")
        return

    port = info["port"]
    resp = _server_request(port, "GET", "/status", timeout=5)
    if not resp:
        _eprint("Server is not responding.")
        return

    uptime_s = resp.get("uptime_seconds", 0)
    idle_s = resp.get("idle_seconds", 0)
    auto_shutdown = resp.get("auto_shutdown_in_seconds")

    uptime_str = _format_duration(uptime_s)
    idle_str = _format_duration(idle_s)

    _eprint(f"Server running (PID {resp['pid']}, port {resp['port']})")
    _eprint(f"  Uptime:  {uptime_str}")
    _eprint(f"  Idle:    {idle_str}")
    _eprint(f"  Device:  {resp.get('device', '?')}")
    _eprint(f"  FAISS:   {resp.get('faiss_vectors', 0):,} vectors")
    _eprint(f"  DuckDB:  {resp.get('duckdb_rows', 0):,} rows")
    if auto_shutdown is not None:
        _eprint(f"  Auto-shutdown in {_format_duration(auto_shutdown)}")


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


def run_search(args: argparse.Namespace) -> None:
    """Execute image search, using server if available."""
    image_path = args.image
    if not os.path.isfile(image_path):
        _eprint(f"Error: image file not found: {image_path}")
        sys.exit(1)

    # Resolve to absolute path (server needs it to read the file)
    image_path = os.path.abspath(image_path)

    if args.local:
        _run_search_local(args, image_path)
        return

    # Try server first
    info = _get_server_info()
    if info:
        scope_desc = SCOPE_DESCRIPTIONS.get(args.scope, args.scope)
        _eprint(f"Searching {scope_desc}...")
        results = _search_via_server(info["port"], image_path, args)
        if results is not None:
            _output_results(results, args)
            return

    # Server not running — check auto_start
    config = load_config()
    auto_start = _config_bool(config.get("auto_start", True))

    if auto_start:
        port = _auto_start_server(args.faiss_index, args.duckdb_path)
        scope_desc = SCOPE_DESCRIPTIONS.get(args.scope, args.scope)
        _eprint(f"Searching {scope_desc}...")
        results = _search_via_server(port, image_path, args)
        if results is not None:
            _output_results(results, args)
            return
        _eprint("Error: failed to communicate with server after auto-start.")
        sys.exit(1)
    else:
        _eprint(
            "Server not running. Start with: bioclip-search serve\n"
            "Or enable auto-start: bioclip-search config --set auto_start true\n"
            "Or use --local for a one-off search without the server."
        )
        sys.exit(1)


def _search_via_server(
    port: int,
    image_path: str,
    args: argparse.Namespace,
) -> Optional[List[Dict]]:
    """Send search request to server. Returns raw results or None."""
    body = {
        "image_path": image_path,
        "top_n": args.top_n,
        "scope": args.scope,
        "nprobe": args.nprobe,
    }
    resp = _server_request(port, "POST", "/search", body)
    if resp is None:
        return None
    if "error" in resp:
        _eprint(f"Server error: {resp['error']}")
        sys.exit(1)
    return resp.get("results", [])


def _run_search_local(args: argparse.Namespace, image_path: str) -> None:
    """Run search locally (load everything in-process)."""
    from PIL import Image

    from bioclip_lite.config import LiteConfig
    from bioclip_lite.services.model_service import ModelService
    from bioclip_lite.services.search_service import SearchService

    faiss_path, duckdb_path = resolve_data_paths(args.faiss_index, args.duckdb_path)
    device = resolve_device(args.device)

    _eprint(f"Loading model on {device}...")
    model = ModelService(device=device)

    _eprint("Loading FAISS index and DuckDB metadata...")
    search_service = SearchService(
        faiss_index_path=faiss_path,
        duckdb_path=duckdb_path,
        nprobe=args.nprobe,
        over_fetch_factor=3,
        metadata_columns=LiteConfig.METADATA_COLUMNS,
    )

    _eprint("Embedding query image...")
    img = Image.open(image_path)
    embedding = model.embed([img], normalize=False)[0]

    scope_desc = SCOPE_DESCRIPTIONS.get(args.scope, args.scope)
    _eprint(f"Searching {scope_desc}...")
    results = search_service.search(
        query_vector=embedding,
        top_n=args.top_n,
        nprobe=args.nprobe,
        scope=args.scope,
    )
    search_service.close()

    if not results:
        _eprint("No results found.")
        sys.exit(0)

    _output_results(results, args)


def _output_results(results: List[Dict], args: argparse.Namespace) -> None:
    """Format and output search results."""
    formatted = _format_results(results)

    if not formatted:
        _eprint("No results found.")
        sys.exit(0)

    if args.format == "json":
        _output_json(formatted, args.output)
    elif args.format == "table":
        _output_table(formatted, args.output, args.scope, args.nprobe)
    elif args.format == "csv":
        _output_csv(formatted, args.output)

    _eprint(f"{len(formatted)} results returned.")


# ------------------------------------------------------------------
# Result formatting
# ------------------------------------------------------------------

OUTPUT_FIELDS = [
    "rank",
    "distance",
    "species",
    "common_name",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "source_dataset",
    "publisher",
    "occurrence_url",
    "url",
]


def _format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw search results to clean output dicts."""
    formatted = []
    for i, r in enumerate(results, 1):
        occurrence_url = None
        source = r.get("source_dataset", "")
        source_id = r.get("source_id", "")
        if source and str(source).lower() == "gbif" and source_id:
            occurrence_url = f"https://www.gbif.org/occurrence/{source_id}"

        out = {
            "rank": i,
            "distance": round(r.get("distance", 0.0), 4),
            "species": r.get("species") or None,
            "common_name": r.get("common_name") or None,
            "kingdom": r.get("kingdom") or None,
            "phylum": r.get("phylum") or None,
            "class": r.get("class") or None,
            "order": r.get("order") or None,
            "family": r.get("family") or None,
            "genus": r.get("genus") or None,
            "source_dataset": r.get("source_dataset") or None,
            "publisher": r.get("publisher") or None,
            "occurrence_url": occurrence_url,
            "url": r.get("identifier") or r.get("url") or None,
        }
        formatted.append(out)
    return formatted


def _output_json(results: List[Dict], output_path: Optional[str]) -> None:
    """Write JSON to stdout or file."""
    text = json.dumps(results, indent=2, ensure_ascii=False)
    if output_path and output_path != "-":
        Path(output_path).write_text(text + "\n")
        _eprint(f"Results saved to {output_path}")
    else:
        print(text)


def _output_table(
    results: List[Dict],
    output_path: Optional[str],
    scope: str,
    nprobe: int,
) -> None:
    """Write a human-readable table to stdout or file."""
    columns = [
        ("Rank", "rank", 4),
        ("Distance", "distance", 8),
        ("Species", "species", 25),
        ("Common Name", "common_name", 20),
        ("Family", "family", 15),
        ("Source", "source_dataset", 6),
        ("Occurrence", "occurrence_url", 45),
        ("URL", "url", 35),
    ]

    header_parts = []
    sep_parts = []
    for label, _, width in columns:
        header_parts.append(label.ljust(width))
        sep_parts.append("─" * width)

    lines = []
    lines.append("  ".join(header_parts))
    lines.append("  ".join(sep_parts))

    for r in results:
        row_parts = []
        for _, key, width in columns:
            val = r.get(key)
            if val is None:
                cell = "-"
            elif isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            if len(cell) > width:
                cell = cell[: width - 3] + "..."
            row_parts.append(cell.ljust(width))
        lines.append("  ".join(row_parts))

    scope_desc = SCOPE_DESCRIPTIONS.get(scope, scope)
    lines.append("")
    lines.append(f"{len(results)} results from scope \"{scope_desc}\" | nprobe={nprobe}")

    text = "\n".join(lines)
    if output_path and output_path != "-":
        Path(output_path).write_text(text + "\n")
        _eprint(f"Results saved to {output_path}")
    else:
        print(text)


def _output_csv(results: List[Dict], output_path: Optional[str]) -> None:
    """Write CSV to file or stdout."""
    if not output_path or output_path == "-":
        outfile = sys.stdout
    else:
        outfile = open(output_path, "w", newline="")

    try:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    finally:
        if outfile is not sys.stdout:
            outfile.close()
            _eprint(f"Results saved to {output_path}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _eprint(*args, **kwargs) -> None:
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def _format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="bioclip-search",
        description="Search 234M biological images using BioCLIP 2 embeddings.",
    )
    p.add_argument(
        "image",
        help=(
            'Path to query image, or a command: '
            '"config" (manage settings), '
            '"serve" (start server), '
            '"stop" (stop server), '
            '"status" (server status).'
        ),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to return (default: 10)",
    )
    p.add_argument(
        "--scope",
        default="all",
        choices=["all", "url_only", "inaturalist", "bioclip2_training"],
        help="Filter result scope (default: all)",
    )
    p.add_argument(
        "--nprobe",
        type=int,
        default=16,
        metavar="N",
        help="FAISS search depth, higher=slower+better (default: 16)",
    )
    p.add_argument(
        "--format",
        choices=["json", "table", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help='Output file path; use "-" for stdout (default: stdout for json/table)',
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device (default: auto-detect CUDA > MPS > CPU)",
    )
    p.add_argument(
        "--faiss-index",
        metavar="PATH",
        default=None,
        help="FAISS index file (overrides config)",
    )
    p.add_argument(
        "--duckdb-path",
        metavar="PATH",
        default=None,
        help="DuckDB metadata file (overrides config)",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Run search locally (skip server, load everything in-process)",
    )

    # Config subcommand arguments (used when image=="config")
    p.add_argument(
        "--show",
        dest="config_show",
        action="store_true",
        help="Show current configuration (use with 'config')",
    )
    p.add_argument(
        "--set",
        dest="config_set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set a config value (use with 'config')",
    )

    return p


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Suppress library logging — we handle our own status messages
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    command = args.image
    if command == "config":
        handle_config(args)
    elif command == "serve":
        handle_serve(args)
    elif command == "stop":
        handle_stop(args)
    elif command == "status":
        handle_status(args)
    else:
        run_search(args)
