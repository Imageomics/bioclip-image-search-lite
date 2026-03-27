# CLI Design Specification: `bioclip-search`

## Overview

A command-line interface for searching 234M biological images using BioCLIP 2
embeddings. Complements the existing Gradio web UI with a scriptable,
pipe-friendly search tool.

## Command Name

`bioclip-search` — registered as a console script entry point via
`pyproject.toml`. Distinct from pybioclip's `bioclip` command.

## Installation

```bash
uv pip install -e .
# or
uv pip install bioclip-image-search-lite
```

This registers the `bioclip-search` command.

## Data Setup

The CLI requires two data files hosted on HuggingFace:
https://huggingface.co/imageomics/bioclip-image-search-lite

| File | Size | Description |
|------|------|-------------|
| FAISS index | ~5.5 GB | 234M BioCLIP 2 vector embeddings for similarity search |
| DuckDB metadata | ~14 GB | Taxonomy and source metadata for all 234M images |

### Download command

```bash
# Download to default location (~/.bioclip-search/data/)
$ bioclip-search download

Source:      https://huggingface.co/imageomics/bioclip-image-search-lite
Destination: /home/user/.bioclip-search/data

Downloading FAISS index (faiss/index.index)...
  Saved to /home/user/.bioclip-search/data/faiss/index.index

Downloading DuckDB metadata (duckdb/metadata.duckdb)...
  Saved to /home/user/.bioclip-search/data/duckdb/metadata.duckdb

Download complete. Paths saved to config.
  Config file: /home/user/.bioclip-search/config.json

Verify with: bioclip-search config --show
```

```bash
# Download to a custom location
$ bioclip-search download --data-dir /scratch/shared/bioclip-data
```

### Using existing data files

If you already have the data files, point to them directly:

```bash
bioclip-search config --set faiss_index /path/to/index.index
bioclip-search config --set duckdb_path /path/to/metadata.duckdb
```

### What happens without data

If data files are not found when searching, the CLI exits with a clear message:

```
$ bioclip-search photo.jpg

Error: data files not found (FAISS index, DuckDB metadata).

Download the required data files first:

  bioclip-search download

Or download to a custom location:

  bioclip-search download --data-dir /path/to/dir

Or point to existing files:

  bioclip-search config --set faiss_index /path/to/index.index
  bioclip-search config --set duckdb_path /path/to/metadata.duckdb
```

### Data Path Resolution Order

Highest priority first:

1. CLI flags (`--faiss-index`, `--duckdb-path`)
2. Config file (`~/.bioclip-search/config.json`)
3. Default data directory (`~/.bioclip-search/data/`)

## Server Architecture

The CLI uses a local daemon to avoid reloading the model (~2.5 GB), FAISS index
(~5.5 GB), and DuckDB metadata (~14 GB) on every invocation. The server keeps
these loaded in memory and serves search requests over HTTP on `127.0.0.1`.

### Auto-start (default)

By default, the first search automatically starts the server in the background:

```
$ bioclip-search photo.jpg
Starting server in background (idle timeout: 30m)...
Server ready.
[results]

$ bioclip-search photo2.jpg     # instant — server already warm
[results]
```

Auto-start can be disabled:

```bash
bioclip-search config --set auto_start false
```

### Manual server management

```bash
# Start server in foreground (shows logs, Ctrl+C to stop)
bioclip-search serve

# Check server status
bioclip-search status
# Server running (PID 48291, port 7863)
#   Uptime:  12m 5s
#   Idle:    3m 22s
#   Device:  cuda
#   FAISS:   234,391,308 vectors
#   DuckDB:  234,391,308 rows
#   Auto-shutdown in 26m 38s

# Stop server
bioclip-search stop
```

### Idle timeout

The server automatically shuts down after a configurable idle period (default:
30 minutes) to free memory:

```bash
bioclip-search config --set idle_timeout 60   # minutes, 0 = never
```

### Local mode

For one-off searches or environments where a daemon isn't practical (e.g.,
SLURM jobs), bypass the server entirely:

```bash
bioclip-search photo.jpg --local
```

This loads everything in-process — slower startup but no background process.

## Usage

### Data setup

```bash
bioclip-search download                       # download to default location
bioclip-search download --data-dir /path      # download to custom location
```

### Search (default action)

The primary command. Takes an image path as a positional argument:

```bash
bioclip-search <image> [options]
```

### Server commands

```bash
bioclip-search serve              # start server in foreground
bioclip-search stop               # stop running server
bioclip-search status             # show server status
```

### Setup verification

```bash
$ bioclip-search check

Environment:
  Python:    3.10.13
  Platform:  Linux 5.14.0
  torch:     2.1.0+cu121
  CUDA:      available (NVIDIA A100)

Data:
  FAISS index:  ~/.bioclip-search/data/faiss/index.index (5.4 GB)
  DuckDB:       ~/.bioclip-search/data/duckdb/metadata.duckdb (14.3 GB)

Server:
  Status:  not running

Config:  ~/.bioclip-search/config.json

No issues found.
```

Reports Python version, torch build, GPU detection, data file status, and server
state. Flags issues with suggested fixes (e.g., CUDA GPU detected but CPU-only
torch installed).

### Config management

```bash
bioclip-search config [--show | --set KEY VALUE]
```

The positional argument distinguishes behavior: if it's a command (`download`,
`check`, `config`, `serve`, `stop`, `status`), that action runs; otherwise it's
treated as an image path for search.

## Search Examples

### Basic search (JSON to stdout)

```bash
$ bioclip-search photo.jpg

[
  {
    "rank": 1,
    "distance": 0.0423,
    "species": "Danaus plexippus",
    "common_name": "Monarch Butterfly",
    "kingdom": "Animalia",
    "phylum": "Arthropoda",
    "class": "Insecta",
    "order": "Lepidoptera",
    "family": "Nymphalidae",
    "genus": "Danaus",
    "source_dataset": "gbif",
    "publisher": "iNaturalist",
    "occurrence_url": "https://www.gbif.org/occurrence/4012345678",
    "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/12345/original.jpg"
  },
  ...
]
```

### Table format

```bash
$ bioclip-search photo.jpg --format table --top-n 5

Rank  Distance  Species                    Common Name           Family           Source  Occurrence                                     URL
----  --------  -------------------------  --------------------  ---------------  ------  ---------------------------------------------  -----------------------------------
1     0.0423    Danaus plexippus           Monarch Butterfly     Nymphalidae      gbif    https://www.gbif.org/occurrence/4012345678     https://inaturalist-open-da...
2     0.0587    Danaus chrysippus          Plain Tiger           Nymphalidae      gbif    https://www.gbif.org/occurrence/4018765432     https://inaturalist-open-da...
3     0.0612    Danaus gilippus            Queen Butterfly       Nymphalidae      gbif    https://www.gbif.org/occurrence/4019988776     https://inaturalist-open-da...
4     0.0734    Limenitis archippus        Viceroy               Nymphalidae      gbif    https://www.gbif.org/occurrence/4015543210     https://static.inaturalist....
5     0.0891    Heliconius melpomene       Postman Butterfly     Nymphalidae      eol     -                                              https://content.eol.org/da...

5 results from scope "All Sources (234M images)" | nprobe=16
```

### CSV export

```bash
$ bioclip-search photo.jpg --format csv --output results.csv --top-n 50 --scope inaturalist

Searching 135M iNaturalist images...
50 results saved to results.csv
```

### Pipe-friendly

JSON output goes to stdout; status/progress messages go to stderr. This enables
clean piping:

```bash
# Extract just species names
bioclip-search photo.jpg | jq -r '.[].species'

# Feed into downstream analysis
bioclip-search photo.jpg --top-n 100 --format csv --output - | python analyze.py
```

### Batch scripting

```bash
# Process a directory of images
for img in /data/images/*.jpg; do
    bioclip-search "$img" --format csv --output - >> all_results.csv
done
bioclip-search stop
```

## Config Examples

```bash
# Show current configuration
$ bioclip-search config --show
  auto_start: true
  duckdb_path: ~/.bioclip-search/data/metadata.duckdb
  device: cpu
  faiss_index: ~/.bioclip-search/data/index.index
  idle_timeout: 30
  port: 7863

# Point to custom data locations
$ bioclip-search config --set faiss_index /scratch/shared/index.index
$ bioclip-search config --set duckdb_path /scratch/shared/metadata.duckdb

# Use GPU
$ bioclip-search config --set device cuda

# Override per-run (does not modify config)
$ bioclip-search photo.jpg --faiss-index /tmp/test-index.index
```

### Device auto-detection

The CLI automatically selects the best available device:

1. `--device` flag (highest priority)
2. `device` in config file
3. Auto-detect: CUDA > MPS > CPU

This means on a GPU node, `bioclip-search photo.jpg` will automatically use
CUDA without any configuration.

### Configurable keys

| Key            | Default      | Description                                       |
|----------------|--------------|---------------------------------------------------|
| `faiss_index`  | —            | Path to FAISS index file                          |
| `duckdb_path`  | —            | Path to DuckDB metadata file                      |
| `device`       | auto-detect  | Compute device (`cpu`, `cuda`, `mps`)             |
| `auto_start`   | `true`       | Auto-start server on first search                 |
| `idle_timeout` | `30`         | Server idle timeout in minutes (0 = never)        |
| `port`         | `7863`       | Server port                                       |

## Full CLI Reference

```
usage: bioclip-search [-h] [--top-n N]
                      [--scope {all,url_only,inaturalist,bioclip2_training}]
                      [--nprobe N] [--format {json,table,csv}] [--output PATH]
                      [--local] [--data-dir PATH] [--device {cpu,cuda,mps}]
                      [--faiss-index PATH] [--duckdb-path PATH]
                      [--show] [--set KEY VALUE]
                      image

Search 234M biological images using BioCLIP 2 embeddings.

positional arguments:
  image                 Path to query image, or a command:
                          "download" - fetch data files from HuggingFace
                          "check"    - verify installation and setup
                          "config"   - manage settings (use with --show/--set)
                          "serve"    - start search server in foreground
                          "stop"     - stop running server
                          "status"   - show server status

search options:
  --top-n N             Number of results to return (default: 10)
  --scope {all,url_only,inaturalist,bioclip2_training}
                        Filter result scope (default: all)
  --nprobe N            FAISS search depth, higher=slower+better (default: 16)
  --format {json,table,csv}
                        Output format (default: json)
  --output PATH         Output file path; use "-" for stdout (default: stdout)
  --local               Run search locally (skip server, load everything in-
                        process)

server/data options:
  --data-dir PATH       Download destination directory (default: ~/.bioclip-search/data)
  --device {cpu,cuda,mps}
                        Compute device (default: auto-detect CUDA > MPS > CPU)
  --faiss-index PATH    FAISS index file (overrides config)
  --duckdb-path PATH    DuckDB metadata file (overrides config)

config options (use with 'config' command):
  Valid keys: auto_start, device, duckdb_path, faiss_index, idle_timeout, port

  --show                Show current configuration
  --set KEY VALUE       Set a config value (e.g. --set port 8000)

examples:
  bioclip-search download                           # download data files
  bioclip-search download --data-dir /shared/data   # download to custom path
  bioclip-search check                              # verify installation
  bioclip-search photo.jpg                          # search (auto-starts server)
  bioclip-search photo.jpg --top-n 50 --scope inaturalist --format table
  bioclip-search photo.jpg --format csv --output results.csv
  bioclip-search photo.jpg --local                  # one-off search, no server
  bioclip-search serve                              # start server in foreground
  bioclip-search status                             # show server info
  bioclip-search stop                               # stop the server
  bioclip-search config --show                      # show current settings
  bioclip-search config --set device cuda           # set a config value
```

## Output Fields

Each result contains:

| Field            | Type     | Description                                            |
|------------------|----------|--------------------------------------------------------|
| `rank`           | int      | Result rank (1-indexed)                                |
| `distance`       | float    | L2 distance from query embedding (lower = more similar)|
| `kingdom`        | string?  | Taxonomic kingdom                                      |
| `phylum`         | string?  | Taxonomic phylum                                       |
| `class`          | string?  | Taxonomic class                                        |
| `order`          | string?  | Taxonomic order                                        |
| `family`         | string?  | Taxonomic family                                       |
| `genus`          | string?  | Taxonomic genus                                        |
| `species`        | string?  | Taxonomic species                                      |
| `common_name`    | string?  | Common name (if available)                             |
| `source_dataset` | string?  | Data source (e.g., "gbif", "eol")                      |
| `publisher`      | string?  | Data publisher (e.g., "iNaturalist")                   |
| `occurrence_url` | string?  | GBIF occurrence page, null for non-GBIF sources        |
| `url`            | string?  | Direct image URL (null if not available)                |

## Architecture

```
bioclip-search photo.jpg
       │
       ▼
  cli.py               ← argument parsing, output formatting, config mgmt
       │
       ├── server.py    ← local HTTP daemon (keeps services loaded in memory)
       │    │
       │    ├── ModelService   ← embed query image (768-dim BioCLIP 2 vector)
       │    └── SearchService  ← FAISS search + DuckDB metadata (two-tier optimized)
       │                          └── scope filtering in Python (not SQL WHERE)
       │                          └── URL reconstruction from prefix table
       │
       ├── config.json  ← persistent data paths + settings (~/.bioclip-search/)
       └── server.json  ← PID/port for server discovery (~/.bioclip-search/)
```

The CLI client communicates with the server over `127.0.0.1:{port}` using JSON
over HTTP. When `--local` is used, the services are loaded in-process instead.

No dependency on Gradio, ImageService, or the web UI layer.

## Implementation Notes

- **Entry point:** `[project.scripts]` in `pyproject.toml`:
  `bioclip-search = "bioclip_lite.cli:main"`
- **Modules:** `src/bioclip_lite/cli.py` (client + CLI), `src/bioclip_lite/server.py` (daemon)
- **Stderr for status:** All progress/status messages to stderr so stdout
  stays clean for piping
- **GBIF occurrence URL:** Constructed as
  `https://www.gbif.org/occurrence/{source_id}` when `source_dataset == "gbif"`,
  null otherwise
- **Two-tier search:** Uses the same over-fetch + Python-side scope filtering
  as the web app to avoid the DuckDB boolean WHERE clause performance cliff
- **Config location:** `~/.bioclip-search/config.json`
- **Data location:** `~/.bioclip-search/data/` (default download target)
- **Server info:** `~/.bioclip-search/server.json` (PID, port, start time)
- **Server log:** `~/.bioclip-search/server.log`
