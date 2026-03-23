---
title: BioCLIP Image Search Lite
emoji: 🦋
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
python_version: "3.10"
app_file: app.py
pinned: false
preload_from_hub:
  - imageomics/bioclip-image-search-lite faiss/index.index
  - imageomics/bioclip-image-search-lite duckdb/metadata.duckdb
license: mit
tags:
  - biology
  - biodiversity
  - embeddings
  - faiss
  - bioclip
  - similarity-search
  - tree-of-life
  - imageomics
  - duckdb
  - image-based-search
  - evolutionary-biology
  - taxonomy
  - plants
  - animals
  - fungi
description: >-
  Upload a photo of an organism and find visually similar images from 200M+ TreeOfLife training samples.

---

# BioCLIP Image Search Lite

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite)**

A lightweight version of the [BioCLIP Vector DB](https://github.com/Imageomics/bioclip-vector-db) image search system. Upload a photo of an organism and find visually similar images from 200M+ training samples — without needing 92 TB of local image storage.

The trick: instead of storing images locally, we serve them directly from their source URLs (iNaturalist S3, GBIF, Wikimedia, etc.). This brings the total deployment footprint from ~92 TB down to ~32 GB.

**Source code:** [Imageomics/bioclip-image-search-lite](https://github.com/Imageomics/bioclip-image-search-lite)

## How it works

```
Upload image → BioCLIP 2 embedding → FAISS search (200M vectors) → DuckDB metadata → Fetch from source URLs
```

Everything runs in a single Gradio process. No microservices, no HDF5 files.

| Component | Size |
|-----------|------|
| FAISS index | 5.8 GB |
| DuckDB metadata | ~14 GB (optimized) |
| Model weights | ~2.5 GB (downloaded on first run) |
| Image storage | 0 (fetched from source URLs) |

## Setup

### Environment setup

```bash
# Create venv with uv
uv venv /path/to/venv --python 3.10
source /path/to/venv/bin/activate

# Install PyTorch (CPU or CUDA)
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
# or for GPU: uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Install the package (registers the bioclip-search CLI command)
uv pip install -e .
```

### Data files

Both the web UI and CLI need two data files:

| File | Size | Description |
|------|------|-------------|
| FAISS index | ~5.8 GB | Pre-built 200M BioCLIP 2 vector index |
| DuckDB metadata | ~14 GB | Taxonomy + source URLs for all 234M images |

**Option A: Automatic download.** The CLI will prompt to download these on
first run to `~/.bioclip-search/data/`. The web app also auto-downloads from
HuggingFace if paths are not provided.

**Option B: Manual paths.** Point to existing files:

```bash
# For CLI (saved for future runs)
bioclip-search config --set faiss_index /path/to/index.index
bioclip-search config --set duckdb_path /path/to/metadata.duckdb

# For web app (passed as flags each time)
python app.py --faiss-index /path/to/index.index --duckdb-path /path/to/metadata.duckdb
```

**Option C: Build from upstream.** If you need to rebuild the DuckDB from the
upstream SQLite or existing DuckDB:

```bash
python scripts/data/convert_duckdb_lite.py \
    --from-duckdb /path/to/existing/metadata.duckdb \
    --output /path/to/output/metadata.duckdb
```

Or submit as a SLURM job: `sbatch scripts/data/convert_duckdb_lite.slurm`

## CLI: `bioclip-search`

A pipe-friendly command-line interface for searching the 234M image dataset.
See [docs/cli-design.md](docs/cli-design.md) for the full specification.

### Quick start

```bash
# First search — auto-downloads data (if needed) and starts background server
bioclip-search photo.jpg

# Subsequent searches are fast (server stays warm)
bioclip-search photo.jpg --top-n 50 --scope inaturalist --format table
bioclip-search photo.jpg --format csv --output results.csv
```

### Output formats

- **JSON** (default) — structured output for scripting and piping
- **Table** — human-readable terminal output
- **CSV** — for spreadsheets and downstream analysis

### Server architecture

The CLI uses a background server to keep the model, FAISS index, and DuckDB
loaded in memory. This avoids the ~30s startup cost on every search.

```bash
bioclip-search status             # check if server is running
bioclip-search stop               # shut down server
bioclip-search serve              # start server in foreground (for debugging)
bioclip-search photo.jpg --local  # bypass server for one-off searches
```

The server auto-starts on first search (configurable) and auto-shuts down after
30 minutes of inactivity (configurable).

### Configuration

Persistent settings are stored in `~/.bioclip-search/config.json`:

```bash
bioclip-search config --show
bioclip-search config --set device cuda
bioclip-search config --set idle_timeout 60    # minutes
bioclip-search config --set auto_start false
```

## Web UI

### Run

```bash
python app.py \
    --faiss-index /path/to/index.index \
    --duckdb-path /path/to/metadata.duckdb \
    --device cpu \
    --scope all
```

Or on OSC: `sbatch scripts/launch_lite.slurm`

Then open `http://<hostname>:7860` in your browser.

## Scope filtering

Use the scope dropdown to control which results appear:

| Scope | Images | Description |
|-------|--------|-------------|
| All Sources | 234M | Everything, including results without images |
| URL-Available Only | 234M (99.99%) | Only results with fetchable source URLs |
| iNaturalist Only | 135M (58%) | iNaturalist observations via AWS Open Data |
| BioCLIP 2 Training | 206M (88%) | Records used in BioCLIP 2 model training |

The app over-fetches from FAISS (3x by default) and filters post-search, so you still get the requested number of results after filtering.

### Why scope filtering is done in Python

Scope filters (`has_url`, `in_bioclip2_training`, etc.) are applied in Python after the DuckDB query, not as SQL WHERE clauses. Benchmarking showed that adding boolean WHERE clauses to ID-based lookups causes a ~370x slowdown (4ms to 1500ms for 50 IDs) because DuckDB scans the full boolean column rather than using the index for small IN-list queries. Since the majority of rows pass these filters (e.g., 100% have URLs, 88% are in training), fetching all results and filtering in Python adds negligible overhead (~3ms) while keeping query latency low.

## Architecture

```
src/bioclip_lite/
  config.py              # Configuration and CLI args
  services/
    model_service.py     # BioCLIP 2 embed + predict
    search_service.py    # FAISS vector search + DuckDB metadata
    image_service.py     # URL fetching with rate limiting
app.py                   # Gradio frontend
```

### Optimizations

- **Embed on upload**: The embedding is computed when you upload an image, not when you click Search. Adjusting top_n or nprobe reuses the cached embedding.
- **iNaturalist rate-limit compliance**: `static.inaturalist.org` URLs are throttled to 1 req/sec. AWS Open Data S3 URLs (`inaturalist-open-data.s3.amazonaws.com`) are fetched in parallel without throttling.
- **Full-res images**: Images are fetched at full resolution during search and reused on click.

## Image retrieval and rate-limit compliance

This app doesn't store images — it fetches them from their original sources at query time. The source URL analysis that informed this design is in the upstream repo: [`scripts/research/analyze_source_urls.py`](https://github.com/Imageomics/bioclip-vector-db/blob/main/scripts/research/analyze_source_urls.py).

### Where the images come from

Of the 234M images in the training set, 207M (88%) have stable source URLs. The majority are iNaturalist observations hosted on the [AWS Open Data](https://registry.opendata.aws/inaturalist-open-data/) program (`inaturalist-open-data.s3.amazonaws.com`), which is designed for public bulk access. The remaining URLs point to GBIF, Wikimedia, Flickr, and other providers.

### Respecting image servers

We take rate limiting seriously — especially for iNaturalist, whose [API Recommended Practices](https://www.inaturalist.org/pages/api+recommended+practices) specify strict thresholds (1 req/sec, 5 GB/hr media) with permanent bans for violations.

The key distinction: **AWS Open Data S3 URLs are not subject to iNaturalist rate limits.** These are served from Amazon's infrastructure as part of the Open Data program. Only `static.inaturalist.org` CDN URLs count against iNat's limits — and those are a small fraction of our dataset.

Compliance measures in [`image_service.py`](src/bioclip_lite/services/image_service.py):

- **User-Agent**: Identifies us as `BioCLIP-Lite/1.0 (academic research; imageomics.org)`
- **Per-domain rate limiting**: Token bucket (1 req/sec) for `static.inaturalist.org`. S3 Open Data URLs are fetched in parallel without throttling.
- **Bandwidth tracking**: Logs cumulative bytes per domain per session, warns at 4 GB/hr for rate-limited domains
- **Sequential CDN fetching**: Rate-limited URLs are fetched one at a time, never in parallel
- **No API calls**: We only fetch images via direct URLs from the metadata DB — no iNaturalist API usage

## Deployment

### Hugging Face Spaces

The app is hosted on HF Spaces with auto-deploy from GitHub. See [docs/deployment-hf-spaces.md](docs/deployment-hf-spaces.md) for the full setup guide — tokens, data hosting, CI/CD, resource limits, and upgrade options.

### OSC

```bash
# One-time: prepare DuckDB
sbatch scripts/data/convert_duckdb_lite.slurm

# Launch the app
sbatch scripts/launch_lite.slurm
```

Resources: 16 CPUs, 48 GB RAM, single process. The FAISS index and DuckDB are memory-mapped, so actual RSS is lower.

## Related

- [bioclip-vector-db](https://github.com/Imageomics/bioclip-vector-db) — Full system with HDF5 image storage
- [pybioclip](https://github.com/Imageomics/pybioclip) — BioCLIP Python client
- [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) — The underlying vision model
