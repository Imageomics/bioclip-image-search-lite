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

## First-Run Setup

On first invocation, if data files are not found, the CLI prompts the user to
download them:

```
$ bioclip-search photo.jpg

bioclip-search: data files not found.

The following files are required:
  FAISS index     (~5.8 GB)  - 234M BioCLIP 2 image embeddings
  DuckDB metadata  (~14 GB)  - taxonomy and source metadata

Download to ~/.bioclip-search/data? [Y/n] y

Downloading from huggingface.co/imageomics/bioclip-image-search-lite...
  FAISS index:     [████████████████████████████] 5.8 GB / 5.8 GB  done
  DuckDB metadata: [██████████████░░░░░░░░░░░░░░] 7.2 GB / 14 GB   51%  ETA 3m
```

After download, paths are saved to `~/.bioclip-search/config.json`:

```json
{
  "faiss_index": "~/.bioclip-search/data/index.index",
  "duckdb_path": "~/.bioclip-search/data/metadata.duckdb"
}
```

Subsequent runs use the saved paths automatically.

### Data Path Resolution Order

Highest priority first:

1. CLI flags (`--faiss-index`, `--duckdb-path`)
2. Config file (`~/.bioclip-search/config.json`)
3. Default data directory (`~/.bioclip-search/data/`)
4. Auto-download prompt

## Server Architecture

The CLI uses a local daemon to avoid reloading the model (~2.5 GB), FAISS index
(~5.8 GB), and DuckDB metadata (~14 GB) on every invocation. The server keeps
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

### Config management

```bash
bioclip-search config [--show | --set KEY VALUE]
```

The positional argument distinguishes behavior: if it's a command (`config`,
`serve`, `stop`, `status`), that action runs; otherwise it's treated as an
image path for search.

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
────  ────────  ─────────────────────────  ────────────────────  ───────────────  ──────  ─────────────────────────────────────────────  ───────────────────────────────────
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
                      [--nprobe N] [--format {json,table,csv}]
                      [--output PATH] [--device {cpu,cuda,mps}]
                      [--faiss-index PATH] [--duckdb-path PATH]
                      [--local] [--show] [--set KEY VALUE]
                      image

Search 234M biological images using BioCLIP 2 embeddings.

positional arguments:
  image                 Path to query image, or a command:
                          "serve"  - start search server in foreground
                          "stop"   - stop running server
                          "status" - show server status
                          "config" - manage settings (use with --show/--set)

search options:
  --top-n N             Number of results to return (default: 10)
  --scope SCOPE         Filter result scope (default: all)
                          all               - All 234M images
                          url_only          - Images with accessible URLs
                          inaturalist       - iNaturalist images only (135M)
                          bioclip2_training - BioCLIP 2 training set (206M)
  --nprobe N            FAISS search depth, higher=slower+better (default: 16)
  --format {json,table,csv}  Output format (default: json)
  --output PATH         Output file path; use "-" for stdout
  --local               Skip server, load everything in-process

data options:
  --device {cpu,cuda,mps}  Compute device (default: auto-detect CUDA > MPS > CPU)
  --faiss-index PATH       FAISS index file (overrides config)
  --duckdb-path PATH       DuckDB metadata file (overrides config)

config options:
  --show                Show current configuration
  --set KEY VALUE       Set a persistent config value
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
