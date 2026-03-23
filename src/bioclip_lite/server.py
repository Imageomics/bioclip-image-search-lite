"""Lightweight local HTTP server for bioclip-search.

Keeps BioCLIP 2 model, FAISS index, and DuckDB metadata loaded in memory
so subsequent CLI searches skip the ~20s startup cost.

Listens on 127.0.0.1 only (not externally accessible).

Can be run directly:
    python -m bioclip_lite.server [options]

Or started via CLI:
    bioclip-search serve
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid as _uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Import shared CLI config (cli.py does NOT import server.py at module level)
from bioclip_lite.cli import (
    CONFIG_DIR,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PORT,
    SERVER_INFO_FILE,
    load_config,
    resolve_device,
)


def write_server_info(port: int, pid: int) -> None:
    """Write PID and port to server info file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "pid": pid,
        "port": port,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    SERVER_INFO_FILE.write_text(json.dumps(info, indent=2) + "\n")


def remove_server_info() -> None:
    """Remove server info file on shutdown."""
    if SERVER_INFO_FILE.exists():
        SERVER_INFO_FILE.unlink()


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles DuckDB types (UUID, etc.)."""

    def default(self, obj):
        if isinstance(obj, _uuid.UUID):
            return str(obj)
        return super().default(obj)


class SearchHandler(BaseHTTPRequestHandler):
    """Handles search, status, and stop requests."""

    def log_message(self, format, *args):
        logger.debug(format, *args)

    def do_POST(self):
        if self.path == "/search":
            self._handle_search()
        elif self.path == "/stop":
            self._handle_stop()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/status":
            self._handle_status()
        else:
            self.send_error(404)

    def _handle_search(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            image_path = body["image_path"]
            top_n = body.get("top_n", 10)
            scope = body.get("scope", "all")
            nprobe = body.get("nprobe", 16)

            self.server.reset_idle_timer()

            img = Image.open(image_path)
            embedding = self.server.model.embed([img], normalize=False)[0]

            results = self.server.search_service.search(
                query_vector=embedding,
                top_n=top_n,
                nprobe=nprobe,
                scope=scope,
            )

            self._send_json({"results": results, "count": len(results)})

        except FileNotFoundError:
            self._send_json(
                {"error": f"Image not found: {body.get('image_path', '?')}"}, status=400
            )
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            self._send_json({"error": str(e)}, status=500)

    def _handle_status(self):
        server = self.server
        uptime = time.time() - server.start_time
        idle = time.time() - server.last_request_time
        remaining = (
            max(0, server.idle_timeout * 60 - idle)
            if server.idle_timeout > 0
            else None
        )

        status = {
            "status": "running",
            "pid": os.getpid(),
            "port": server.server_port,
            "device": server.device,
            "uptime_seconds": round(uptime),
            "idle_seconds": round(idle),
            "idle_timeout_minutes": server.idle_timeout,
            "auto_shutdown_in_seconds": round(remaining) if remaining is not None else None,
            "faiss_vectors": server.search_service.total_vectors,
            "duckdb_rows": server.duckdb_rows,
        }
        self._send_json(status)

    def _handle_stop(self):
        self._send_json({"status": "stopping"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _send_json(self, data, status=200):
        response = json.dumps(data, cls=_SafeEncoder).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class SearchServer(HTTPServer):
    """Single-threaded HTTP server with loaded search services."""

    def __init__(self, port, model, search_service, device, idle_timeout, duckdb_rows):
        super().__init__(("127.0.0.1", port), SearchHandler)
        self.model = model
        self.search_service = search_service
        self.device = device
        self.idle_timeout = idle_timeout
        self.duckdb_rows = duckdb_rows
        self.start_time = time.time()
        self.last_request_time = time.time()
        self._idle_timer = None

        if idle_timeout > 0:
            self.reset_idle_timer()

    def reset_idle_timer(self):
        if self._idle_timer:
            self._idle_timer.cancel()
        self.last_request_time = time.time()
        if self.idle_timeout > 0:
            self._idle_timer = threading.Timer(
                self.idle_timeout * 60, self._idle_shutdown
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _idle_shutdown(self):
        logger.info(f"Idle timeout ({self.idle_timeout}m) reached, shutting down")
        threading.Thread(target=self.shutdown, daemon=True).start()

    def server_close(self):
        if self._idle_timer:
            self._idle_timer.cancel()
        self.search_service.close()
        remove_server_info()
        logger.info("Server stopped")
        super().server_close()


def run_server(
    port: int,
    faiss_path: str,
    duckdb_path: str,
    device: str,
    idle_timeout: int,
    model_str: str = "hf-hub:imageomics/bioclip-2",
) -> None:
    """Load services and start the HTTP server. Blocks until shutdown."""
    from bioclip_lite.config import LiteConfig
    from bioclip_lite.services.model_service import ModelService
    from bioclip_lite.services.search_service import SearchService

    print("Loading BioCLIP 2 model...", file=sys.stderr)
    model = ModelService(device=device, model_str=model_str)

    print("Loading FAISS index and DuckDB metadata...", file=sys.stderr)
    search_service = SearchService(
        faiss_index_path=faiss_path,
        duckdb_path=duckdb_path,
        nprobe=16,
        over_fetch_factor=3,
        metadata_columns=LiteConfig.METADATA_COLUMNS,
    )

    duckdb_rows = search_service.conn.execute(
        "SELECT COUNT(*) FROM metadata"
    ).fetchone()[0]

    server = SearchServer(
        port=port,
        model=model,
        search_service=search_service,
        device=device,
        idle_timeout=idle_timeout,
        duckdb_rows=duckdb_rows,
    )

    write_server_info(port, os.getpid())

    def _signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(
        f"Server ready on 127.0.0.1:{port} (PID {os.getpid()})\n"
        f"  FAISS:   {search_service.total_vectors:,} vectors\n"
        f"  DuckDB:  {duckdb_rows:,} rows\n"
        f"  Device:  {device}\n"
        f"  Idle timeout: {idle_timeout}m",
        file=sys.stderr,
    )

    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    """Entry point for `python -m bioclip_lite.server`."""
    parser = argparse.ArgumentParser(description="bioclip-search server")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--faiss-index", default=None)
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--idle-timeout", type=int, default=None)
    parser.add_argument("--model-str", default=None)
    args = parser.parse_args()

    config = load_config()

    port = args.port or int(config.get("port", DEFAULT_PORT))
    faiss_path = args.faiss_index or config.get("faiss_index", "")
    duckdb_path = args.duckdb_path or config.get("duckdb_path", "")
    device = resolve_device(args.device)
    idle_timeout = (
        args.idle_timeout
        if args.idle_timeout is not None
        else int(config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
    )
    model_str = args.model_str or "hf-hub:imageomics/bioclip-2"

    if not faiss_path or not duckdb_path:
        print(
            "Error: FAISS index and DuckDB paths required.\n"
            "Set them with: bioclip-search config --set faiss_index /path/to/index",
            file=sys.stderr,
        )
        sys.exit(1)

    # Setup logging
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CONFIG_DIR / "server.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file),
        ],
    )
    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    run_server(port, faiss_path, duckdb_path, device, idle_timeout, model_str)


if __name__ == "__main__":
    main()
