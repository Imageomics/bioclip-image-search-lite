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
  Upload a photo of an organism and find visually similar images from 239M+ TreeOfLife training samples.

---

# BioCLIP Image Search Lite

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite)**

A lightweight version of the [BioCLIP Vector DB](https://github.com/Imageomics/bioclip-vector-db) image search system. Upload a photo of an organism and find visually similar images from 239M+ training samples — without needing 92 TB of local image storage.

The trick: instead of storing images locally, we serve them directly from their source URLs (iNaturalist S3, GBIF, Wikimedia, etc.). This brings the total deployment footprint from ~92 TB down to ~25 GB. 

**Source code:** [Imageomics/bioclip-image-search-lite](https://github.com/Imageomics/bioclip-image-search-lite)

## How it works

```
Upload image → BioCLIP 2 embedding → FAISS search (239M vectors) → DuckDB metadata → Fetch from source URLs
```

Everything runs in a single Gradio process. No microservices, no HDF5 files.

| Component | Size |
|-----------|------|
| FAISS index | ~6 GB |
| DuckDB metadata | ~15 GB (optimized) |
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

1. **FAISS index** — the pre-built 239M vector index
2. **DuckDB metadata** — taxonomy + source URLs (where available) for all 239M images

TODO: in this commit we should probably delete what's in scripts/data which a lot might be irrelevant sqlite conversion stuff. in this section we should provide hf command to download the files from hf repo. The setup section will be improved additionally as the cli-feature gets merged. 

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
| All Sources | 239M | Everything; bioscan rows are included in the index + lookup but cannot be retrieved as images (no public URL) |
| URL-Available Only | 234M (97.8%) | Only results with fetchable source URLs (excludes bioscan) |
| iNaturalist Only | — | iNaturalist observations via AWS Open Data |
| BioCLIP 2 Training | 211M (88.1%) | Records used in BioCLIP 2 model training |
| BioCLIP 2.5 Huge Training | 233M (97.3%) | Records used in BioCLIP 2.5 Huge model training |

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

This app doesn't store images, it fetches them from their original sources at query time. The source URL analysis that informed this design is in the upstream repo: [`scripts/research/analyze_source_urls.py`](https://github.com/Imageomics/bioclip-vector-db/blob/main/scripts/research/analyze_source_urls.py).

### Where the images come from

Of the 239M images in the index, 234M (97.8%) have source URLs. The remaining ~5M rows are all BIOSCAN-5M specimen images, which are included in the index and lookup table for search and taxonomy resolution but cannot be displayed as thumbnails because they have no public URL. All other sources (GBIF, EOL, FathomNet) have URL coverage. The majority of URL-bearing rows are iNaturalist observations hosted on the [AWS Open Data](https://registry.opendata.aws/inaturalist-open-data/) program (`inaturalist-open-data.s3.amazonaws.com`); the remainder point to GBIF publishers, Wikimedia, Flickr, and other providers.

### Respecting image servers

We take rate limiting seriously, especially for iNaturalist, whose [API Recommended Practices](https://www.inaturalist.org/pages/api+recommended+practices) specify strict thresholds (1 req/sec, 5 GB/hr media) with permanent bans for violations.

The key distinction: **AWS Open Data S3 URLs are not subject to iNaturalist rate limits.** These are served from Amazon's infrastructure as part of the Open Data program. Only `static.inaturalist.org` CDN URLs count against iNat's limits — and those are a small fraction of our dataset.

Compliance measures in [`image_service.py`](src/bioclip_lite/services/image_service.py):

- **User-Agent**: Identifies the software as `bioclip-image-search-lite/<version> (+https://github.com/Imageomics/bioclip-image-search-lite)`, following the `name/version (+URL)` convention used by major crawlers ([Googlebot](https://developers.google.com/search/docs/crawling-indexing/google-common-crawlers) is the canonical example).
- **Per-domain rate limiting**: Token bucket (1 req/sec) for `static.inaturalist.org`. S3 Open Data URLs are fetched in parallel without throttling.
- **Bandwidth tracking**: Logs cumulative bytes per domain per session, warns at 4 GB/hr for rate-limited domains
- **Sequential CDN fetching**: Rate-limited URLs are fetched one at a time, never in parallel
- **No API calls**: We only fetch images via direct URLs from the metadata DB. No iNaturalist API usage

## Deployment

### Hugging Face Spaces

The app is hosted on HF Spaces with auto-deploy from GitHub. See [docs/deployment-hf-spaces.md](docs/deployment-hf-spaces.md) for the full setup guide: tokens, data hosting, CI/CD, resource limits, and upgrade options.

## Related

- [bioclip-vector-db](https://github.com/Imageomics/bioclip-vector-db) — Full system with HDF5 image storage
- [pybioclip](https://github.com/Imageomics/pybioclip) — BioCLIP Python client
- [BioCLIP 2](https://huggingface.co/imageomics/BioCLIP 2) — The underlying vision model
