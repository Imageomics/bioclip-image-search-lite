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
---

# BioCLIP Image Search Lite

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite)**

A lightweight version of the [BioCLIP Vector DB](https://github.com/Imageomics/bioclip-vector-db) image search system. Upload a photo of an organism and find visually similar images from 200M+ training samples — without needing 92 TB of local image storage.

The trick: instead of storing images locally, we serve them directly from their source URLs (iNaturalist S3, GBIF, Wikimedia, etc.). This brings the total deployment footprint from ~92 TB down to ~32 GB.

## How it works

```
Upload image → BioCLIP-2 embedding → FAISS search (200M vectors) → DuckDB metadata → Fetch from source URLs
```

Everything runs in a single Gradio process. No microservices, no HDF5 files.

| Component | Size |
|-----------|------|
| FAISS index | 5.8 GB |
| DuckDB metadata | 25.8 GB |
| Model weights | ~2.5 GB (downloaded on first run) |
| Image storage | 0 (fetched from source URLs) |

## Quick start

### Environment setup

```bash
# Create venv with uv
uv venv /path/to/venv --python 3.10
source /path/to/venv/bin/activate

# Install PyTorch CPU and dependencies
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install faiss-cpu duckdb pybioclip gradio Pillow requests
```

### Data preparation

The app needs two data files:

1. **FAISS index** — the pre-built 200M vector index
2. **DuckDB metadata** — taxonomy + source URLs for all 234M images

If you need to build the DuckDB from the upstream SQLite:

```bash
python scripts/data/convert_duckdb_lite.py \
    --from-duckdb /path/to/existing/metadata.duckdb \
    --output /path/to/output/metadata.duckdb
```

Or submit as a SLURM job: `sbatch scripts/data/convert_duckdb_lite.slurm`

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

Not all 234M images have source URLs. Use the scope dropdown to control which results appear:

| Scope | Images | Description |
|-------|--------|-------------|
| All Sources | 234M | Everything, including results without images |
| URL-Available Only | 207M (88%) | Only results with fetchable source URLs |
| iNaturalist Only | 135M (58%) | iNaturalist observations via AWS Open Data |

The app over-fetches from FAISS (3x by default) and filters post-search, so you still get the requested number of results after filtering.

## Architecture

```
src/bioclip_lite/
  config.py              # Configuration and CLI args
  services/
    model_service.py     # BioCLIP-2 embed + predict
    search_service.py    # FAISS vector search + DuckDB metadata
    image_service.py     # URL fetching with rate limiting
app.py                   # Gradio frontend
```

### Optimizations

- **Embed on upload**: The embedding is computed when you upload an image, not when you click Search. Adjusting top_n or nprobe reuses the cached embedding.
- **iNaturalist rate-limit compliance**: `static.inaturalist.org` URLs are throttled to 1 req/sec. AWS Open Data S3 URLs (`inaturalist-open-data.s3.amazonaws.com`) are fetched in parallel without throttling.
- **Thumbnails first**: Gallery shows 256px thumbnails. Full resolution is fetched on click.

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
- [BioCLIP-2](https://huggingface.co/imageomics/bioclip-2) — The underlying vision model
