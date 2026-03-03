---
license: cc0-1.0
language:
- en
pretty_name: BioCLIP Image Search Lite
task_categories:
- image-feature-extraction
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
- evolutionary biology
- taxonomy
- plants
- animals
- fungi
size_categories:
- 100M<n<1B
description: >-
  Pre-computed FAISS index (~200M BioCLIP-2 embeddings) and DuckDB metadata
  database (234M rows) for image similarity search across the TreeOfLife-200M dataset. No images from the source dataset are redistributed.
datasets:
  - imageomics/TreeOfLife-200M
  - GBIF
  - bioscan-ml/BIOSCAN-5M
  - EOL
  - FathomNet
---

<!--
NOTE: This repository is typed as a "model" repo on Hugging Face, but it
functions as a **data** repository.  It hosts pre-computed compute artifacts
(a FAISS index and a DuckDB metadata database) — not model weights.
No images are stored or redistributed here.
-->

# BioCLIP Image Search Lite

Pre-computed [FAISS](https://github.com/facebookresearch/faiss/wiki) index and [DuckDB](https://duckdb.org/) metadata database powering the
[BioCLIP Image Search Lite](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite) application —
a lightweight image similarity search engine over the 200M+ organism images from the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M).

The **FAISS index** enables sub-second approximate nearest-neighbor search over ~200M image embeddings, while the **DuckDB database** maps each search record back to its source image URL and other associated metadata. 

> **Why a model repo?** Though this is a data repository, it is hosted as a Hugging Face "model" repo because model repos provide 50 GB of free storage and can be [pre-loaded into a Hugging Face Space](https://huggingface.co/docs/hub/en/spaces-sdks-docker#preloading-models-and-other-data), keeping the index in memory while the Space is active. This lite application relies on URLs that can be queried in real-time to avoid the 92 TB local image storage overhead of the full image set.

## Dataset Details

### Dataset Description

- **Curated by:** Net Zhang, Sreejith Menon, Elizabeth Campolongo, Matthew Thompson, Arnab Nandi, Hilmar Lapp, Jianyang Gu <!-- TODO: confirm full author list -->
- **Demo:** [BioCLIP Image Search Lite Space](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite)
- **Repository:** [Imageomics/bioclip-image-search-lite](https://github.com/Imageomics/bioclip-image-search-lite)
- **Paper:** [BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](https://arxiv.org/abs/2505.23883)
- **License:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (FAISS index; metadata inherits upstream licenses — see [Licensing Information](#licensing-information))
- **Source dataset:** [imageomics/TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
- **Embedding model:** [imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2)

This repository contains two compute artifacts derived from the [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) dataset:

1. **FAISS index** — A trained approximate nearest-neighbor index over ~200M [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) image embeddings, enabling sub-second similarity search. See [Data Collection and Processing](#data-collection-and-processing) for more details.
2. **DuckDB metadata database** — A 234M-row database mapping each image to its taxonomic classification, source provenance, and original URL.

Together, these enable a full image similarity search pipeline: embed a query image with BioCLIP 2, search the FAISS index for nearest neighbors, and look up rich metadata and source image URLs via DuckDB.

**This repository does not contain or redistribute any images.** It contains only compute artifacts (FAISS index) and metadata (DuckDB database). Images are fetched on-demand from their original source URLs (primarily iNaturalist AWS Open Data and other biodiversity platforms utilizing AWS or similar).

### Supported Tasks

- **Image similarity search:** Given a query image of an organism, find the most visually similar images across 200M+ samples from the TreeofLife dataset.
- Find taxonomic information (e.g., species, genera, families) and explore other available metadata associated to these visually similar images.
- **Embedding-based retrieval:** Use the pre-built FAISS index for any downstream task requiring approximate nearest-neighbor search over BioCLIP 2 embeddings.


## Dataset Structure

### Repository Layout

```
imageomics/bioclip-image-search-lite/
    faiss/
        index.index          # FAISS IVF+PQ index (~5.8 GB, ~200M vectors)
    duckdb/
        metadata.duckdb      # DuckDB metadata database (~27 GB, 234M rows)
```

### FAISS Index

| Property | Value |
|----------|-------|
| **Index type** | `IVF65536,PQ16` ([Inverted File Index](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#cell-probe-methods-indexivf-indexes) with [Product Quantization](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#indexivfpq)) |
| **Vectors** | ~200M |
| **Dimensions** | 768 (BioCLIP 2 ViT-L/14 output) |
| **Normalization** | L2-normalized (inner product ≈ cosine similarity) |
| **IVF cells** | 65,536 Voronoi partitions |
| **PQ encoding** | 16 sub-quantizers, 48 dims each, 256 centroids per sub-quantizer (16 bytes/vector) |
| **File size** | ~5.8 GB |

**Search parameters:**
- `nprobe` controls accuracy vs. speed (default: 16, range: 1–128). Higher values probe more IVF cells, increasing recall at the cost of latency.
- Query vectors **must** be L2-normalized before searching.

### DuckDB Metadata Schema

**Table:** `metadata` — 234,391,308 rows

| Column | Type | Description |
|--------|------|-------------|
| `id` | `INTEGER` | FAISS vector index. Maps directly to the vector ID in the FAISS index. |
| `uuid` | `UUID` | Unique identifier for the image in [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M). |
| `kingdom` | `VARCHAR` | Kingdom classification (e.g., `Animalia`, `Plantae`, `Fungi`). |
| `phylum` | `VARCHAR` | Phylum classification. |
| `class` | `VARCHAR` | Class classification. |
| `order` | `VARCHAR` | Order classification. |
| `family` | `VARCHAR` | Family classification. |
| `genus` | `VARCHAR` | Genus classification. |
| `species` | `VARCHAR` | Species epithet (specific epithet only). |
| `common_name` | `VARCHAR` | Vernacular/common name where available (sourced from GBIF Backbone Taxonomy). Corresponds to `common` in TreeOfLife-200M catalog. |
| `source_dataset` | `VARCHAR` | Data source: `gbif`, `eol`, `bioscan`, or `fathomnet`. Corresponds to `data_source` in TreeOfLife-200M catalog. |
| `source_id` | `VARCHAR` | Unique identifier from source (e.g., GBIF `gbifID`, EOL content/page ID). |
| `publisher` | `VARCHAR` | Organization that published the data (GBIF records only, e.g., `iNaturalist`). |
| `img_type` | `VARCHAR` | Image type (e.g., `Citizen Science`, `Museum Specimen: Fungi`, `Camera-trap`). GBIF only; others are `Unidentified`. |
| `identifier` | `VARCHAR` | URL to the original image, or `NULL` if unavailable. Corresponds to `source_url` in TreeOfLife-200M catalog. |
| `has_url` | `BOOLEAN` | Materialized flag: `TRUE` if `identifier` is not null/empty. Used for scope filtering. |

**Column name mapping from [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) catalog:**

| This repo | TreeOfLife-200M | Notes |
|-----------|-----------------|-------|
| `id` | — | New; FAISS vector position index |
| `common_name` | `common` | Renamed |
| `source_dataset` | `data_source` | Renamed |
| `identifier` | `source_url` | Renamed |
| `has_url` | — | Derived; materialized boolean |
| All others | Same name | Direct mapping |

**Columns from TreeOfLife-200M catalog not included:** `scientific_name`, `basis_of_record`, `shard_filename`, `shard_file_path`, `base_dataset_file_path`, `resolution_status`.

For more background on these columns, please see the [data field descriptions from TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M#data-fields).

**Indexes:**
- `idx_id` on `id` (primary lookup for FAISS result mapping)
- `idx_scope` on `(source_dataset, has_url)` (scope filtering)

**Data coverage:**

| Scope | Count | Percentage |
|-------|-------|------------|
| Total rows | 234,391,308 | 100% |
| With URL (`has_url = TRUE`) | ~207M | 88.4% |
| iNaturalist (`source_dataset = 'gbif' AND publisher = 'iNaturalist'`) | ~136M | 58% |
| Without URL | ~27M | 11.6% |

### Data Splits

No predefined splits. The data is used as a single search corpus.


## Usage

Please also see the notes in [Recommendations](#recommendations) for usage suggestions.

### Searching the FAISS Index

```python
import faiss
import numpy as np

# Load the index
index = faiss.read_index("faiss/index.index")
print(f"{index.ntotal:,} vectors, {index.d} dims")

# Tune search accuracy (higher nprobe = better recall, slower)
index.nprobe = 16

# Your query vector (768-dim, from BioCLIP-2)
# Must be L2-normalized before searching
query = np.random.randn(1, 768).astype("float32")  # replace with real embedding
faiss.normalize_L2(query)

# Search
distances, ids = index.search(query, k=10)
print(f"Top-10 IDs: {ids[0]}")
print(f"L2 distances: {distances[0]}")
```

### Looking Up Metadata in DuckDB

```python
import duckdb

con = duckdb.connect("duckdb/metadata.duckdb", read_only=True)

# Look up metadata for FAISS result IDs
faiss_ids = [42, 1337, 99999]  # replace with actual IDs from index.search()
result = con.execute(
    "SELECT * FROM metadata WHERE id IN (SELECT unnest($1::INTEGER[]))",
    [faiss_ids],
).fetchdf()
print(result[["id", "genus", "species", "common_name", "identifier"]])
```

### End-to-End: Image Query → Similar Organisms

```python
import faiss
import duckdb
import torch
import open_clip
from PIL import Image

# 1. Load BioCLIP 2
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:imageomics/bioclip-2"
)
model.eval()

# 2. Embed the query image
image = preprocess(Image.open("query.jpg")).unsqueeze(0)
with torch.no_grad():
    query = model.encode_image(image).numpy().astype("float32")
faiss.normalize_L2(query)

# 3. Search FAISS
index = faiss.read_index("faiss/index.index")
index.nprobe = 16
distances, ids = index.search(query, k=10)

# 4. Fetch metadata
con = duckdb.connect("duckdb/metadata.duckdb", read_only=True)
results = con.execute(
    "SELECT * FROM metadata WHERE id IN (SELECT unnest($1::INTEGER[]))",
    [ids[0].tolist()],
).fetchdf()

for _, row in results.iterrows():
    print(f"{row['genus']} {row['species']} ({row['common_name']}) — {row['identifier']}")
```


## Dataset Creation

### Curation Rationale

The full [BioCLIP Vector DB](https://github.com/Imageomics/bioclip-vector-db) stores 234M images totaling ~92 TB — far too large for lightweight deployment. [BioCLIP Image Search Lite](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite) was created to make the similarity search capability accessible on constrained infrastructure (e.g., Hugging Face Spaces free tier: 2 vCPU, 16 GB RAM, 50 GB disk) by:

1. Replacing local image storage with on-demand URL fetching from publicly accessible external sources (primarily [iNaturalist AWS Open Data](https://github.com/inaturalist/inaturalist-open-data) S3).
2. Compressing the metadata from an 80 GB SQLite database to a ~27 GB DuckDB database (optimized via columnar storage and compression).
3. Packaging the FAISS index (~5.8 GB) and DuckDB metadata as the only deployment artifacts.

This approach trades occasional missing thumbnails (when source URLs are unavailable) for a >1000x reduction in storage requirements. See [Imageomics/bioclip-vector-db#47](https://github.com/Imageomics/bioclip-vector-db/issues/47#issuecomment-3927846723) for the full design rationale.

#### URL Stability

This dataset relies on external image URLs rather than storing images locally. The majority (~65%) of URLs point to the [iNaturalist Open Data S3 bucket](https://registry.opendata.aws/inaturalist-open-data/) (`inaturalist-open-data.s3.amazonaws.com`), which is publicly accessible without authentication via the [AWS Open Data Sponsorship Program](https://aws.amazon.com/opendata/open-data-sponsorship-program/).

These URLs are **reasonably persistent but not guaranteed stable**:

- **No official stability guarantee.** The iNaturalist Open Data [documentation](https://github.com/inaturalist/inaturalist-open-data/blob/main/README.md) warns: *"There may be rows in these tables pointing to images that are no longer in the bucket having been deleted or moved."*
- **User-driven changes.** Photos may be removed from the S3 bucket if a user deletes their observation or changes the photo license to "all rights reserved" (only [CC-licensed photos](https://www.inaturalist.org/posts/84932-updated-choosing-licensing-that-allows-scientists-to-use-your-observations) qualify for the AWS-hosted open data bucket).
- **Historical URL migration.** In 2021, iNaturalist [migrated photo URLs](https://forum.inaturalist.org/t/photo-links-changed/25705) from `static.inaturalist.org` to the S3 bucket, breaking previously stable links.
- **AWS sponsorship is renewable.** The AWS Open Data Sponsorship runs on a [2-year renewable term](https://aws.amazon.com/opendata/open-data-sponsorship-program/terms/) with no uptime SLA.
- **No explicit S3 rate limit.** The iNaturalist [API Recommended Practices](https://www.inaturalist.org/pages/api+recommended+practices) recommend <5 GB/hour and <24 GB/day for media downloads, though it is unclear whether this applies to direct S3 access. The [BioCLIP Image Search Lite application](https://github.com/Imageomics/bioclip-image-search-lite) respects these limits regardless.

The remaining URLs point to other biodiversity platforms ([EOL](https://eol.org/), [BIOSCAN-5M](https://biodiversitygenomics.net/projects/5m-insects/), [FathomNet](https://www.fathomnet.org/)), each with their own availability characteristics. The ~11.6% of records without any URL are still searchable via the FAISS index but cannot display a source image.

### Source Data

#### Data Collection and Processing

**Embeddings → FAISS index:**

All 200M+ images in [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) were embedded using the [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) model (ViT-L/14, 768-dim output). The FAISS index was built in a multi-phase pipeline on the [Ohio Supercomputer Center (OSC)](https://www.osc.edu/) HPC cluster:

1. **Stratified sampling** (Spark, 80 executors): ~15–20M representative vectors sampled from the full corpus, stratified by taxonomic class using capped proportional sampling (seed=42).
2. **Index training** (1 GPU): An `IVF65536,PQ16` index was trained on the stratified sample to learn 65,536 IVF centroids and the PQ codebook.
3. **Vector insertion** (8 parallel GPU jobs): All ~200M L2-normalized vectors were added to the trained index in parallel shards (batch size 3M).
4. **Merge** (CPU, 64 GB RAM): All shards were merged into the final index.

Full training scripts: [Imageomics/bioclip-vector-db/scripts/](https://github.com/Imageomics/bioclip-vector-db/blob/feature/model_server/scripts/).

For additional information on FAISS index types and search parameters, see the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki).

**Metadata → DuckDB:**

The DuckDB metadata database was assembled from two sources produced by the [BioCLIP Vector DB](https://github.com/Imageomics/bioclip-vector-db) project:

1. **FAISS ID ↔ UUID mapping** — A "flight plan" created *before* FAISS training ([`create_lookup.py`](https://github.com/Imageomics/bioclip-vector-db/blob/feature/model_server/src/bioclip_vector_db/batch/create_lookup.py)). This scans all source embedding files and assigns deterministic integer IDs to each record, producing a manifest that maps `id` → `uuid` and ensures contiguous ID space matching the FAISS vector positions.
2. **UUID ↔ catalog metadata** — Taxonomic and provenance metadata derived from the [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) catalog (see [column mapping](#duckdb-metadata-schema) above).

The Lite repo merged these into a single DuckDB database ([`convert_duckdb_lite.py`](https://github.com/Imageomics/bioclip-image-search-lite/blob/main/scripts/data/convert_duckdb_lite.py)) with the following optimizations:

- Added a materialized `has_url` boolean column for efficient scope filtering.
- Created indexes: `idx_id` on `id` (primary FAISS lookup) and `idx_scope` on `(source_dataset, has_url)` (scope filtering).
- Leveraged DuckDB's columnar storage and compression, reducing the database from ~80 GB (SQLite) to ~27 GB.

#### Source Data Producers

- **Images and taxonomic metadata:** [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M). Taxonomic labels were standardized using [TaxonoPy](https://github.com/Imageomics/TaxonoPy).
- **Embeddings:** Generated by the [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) model.

### Personal and Sensitive Information

This repository does not contain or redistribute any images. However, the metadata includes URLs pointing to source images that may occasionally contain humans in the background (e.g., citizen science observations, museum collection documentation). The upstream TreeOfLife-200M dataset applies human face detection filtering to minimize such occurrences. See the [TreeOfLife-200M dataset card (processing section)](https://huggingface.co/datasets/imageomics/TreeOfLife-200M#data-curation-and-processing)  for details.

### Annotations

This dataset does not include annotations created specifically for this repository. All taxonomic labels, common names, and provenance metadata are inherited directly from the TreeOfLife-200M catalog, which aligned the taxonomic names provided by [GBIF](https://www.gbif.org/), [EOL](https://eol.org/), [BIOSCAN-5M](https://github.com/bioscan-ml/BIOSCAN-5M), and [FathomNet](https://www.fathomnet.org/) using [TaxonoPy](https://imageomics.github.io/TaxonoPy/). See the [TreeOfLife-200M dataset card](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) for details on annotation processes and provenance.


## Considerations for Using the Data

### Bias, Risks, and Limitations

This dataset inherits biases and considerations from [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M#considerations-for-using-the-data). The following are exaggerated in this instance (BioCLIP Image Search Lite) due to available image representation (those readily fetched by URL):

- **Taxonomic coverage is uneven.** Despite including 952K+ unique taxa, coverage is heavily biased toward well-photographed organisms. Citizen science observations (primarily iNaturalist) comprise ~58% of the data, skewing representation toward charismatic species and regions where citizen science is most active (Western/developed countries).
- **Incomplete taxonomic labels.** As inherited from TreeOfLife-200M, only ~89% of records have full species-level taxonomy. ~11% lack complete labels due to biodiversity data complexities (`NULL` values at lower ranks).
- **URL availability is not guaranteed.** ~11.6% of records have no source URL. For records with URLs, images may become unavailable over time due to URL rot, server changes, or content removal.
- **FAISS approximation.** The IVF+PQ index trades exactness for speed. Results are approximate nearest neighbors — some true nearest neighbors may be missed depending on the `nprobe` setting. Higher `nprobe` values improve recall at the cost of latency.
- **Embedding bias.** Similarity is determined by BioCLIP 2 embeddings, which may encode biases from the training data.

### Recommendations

- Set `nprobe` appropriately for your accuracy needs (default 16 is a reasonable balance; increase to 64–128 for higher recall).
- When using results for research, verify taxonomic labels against authoritative sources — labels are inherited from community-contributed data.
- Be aware of geographic and taxonomic sampling biases when interpreting similarity search results.
- For issues with specific records (mislabeling, broken URLs, etc.), report via the [Community tab](https://huggingface.co/imageomics/bioclip-image-search-lite/discussions) or [GitHub Issues](https://github.com/Imageomics/bioclip-image-search-lite/issues).


## Licensing Information

The FAISS index in this repository is dedicated to the public domain under the [CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/). The DuckDB metadata inherits licensing terms from its upstream sources (GBIF, EOL, BIOSCAN-5M, FathomNet); see the [TreeOfLife-200M licensing information](https://huggingface.co/datasets/imageomics/TreeOfLife-200M#licensing-information) for per-record details.

**Important:** This repository does not contain or redistribute any images. The metadata includes URLs pointing to images hosted by their original sources. Individual images retain their original source licenses, which vary by provider (ranging from [CC0](https://creativecommons.org/publicdomain/zero/1.0/) to [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/)). Users must respect each image's original license terms when accessing images via the provided URLs. More details on licensing by source and per-image license information is provided in [TreeOfLife-200M provenance descriptions](https://huggingface.co/datasets/imageomics/TreeOfLife-200M#licensing-information).

We ask that you cite this dataset and associated papers if you make use of it in your research.


## Citation

<!-- TODO: confirm full author list and add DOI once generated -->
**Data:**
```bibtex
@misc{zhang2026biocliplite,
  author = {Zhang, Net and Menon, Sreejith and Campolongo, Elizabeth and Thompson, Matthew and Nandi, Arnab and Lapp, Hilmar and Gu, Jianyang},
  title = {{BioCLIP Image Search Lite}},
  year = {2026},
  url = {https://huggingface.co/imageomics/bioclip-image-search-lite},
  publisher = {Hugging Face}
}
```

Please also cite the source dataset, embedding model, and FAISS library:

**TreeOfLife-200M:**
```bibtex
@misc{treeoflife200m,
  title = {{TreeOfLife-200M}},
  year = {2025},
  url = {https://huggingface.co/datasets/imageomics/TreeOfLife-200M},
  doi = {10.57967/hf/6786},
  publisher = {Hugging Face}
}
```

**BioCLIP 2:**
```bibtex
@article{gu2025bioclip,
  title = {{BioCLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
  author = {Gu, Jianyang and Stevens, Samuel and Campolongo, Elizabeth G and Thompson, Matthew J and Zhang, Net and Wu, Jiaman and Kopanev, Andrei and Mai, Zheda and White, Alexander E. and Balhoff, James and Dahdul, Wasila M and Rubenstein, Daniel and Lapp, Hilmar and Berger-Wolf, Tanya and Chao, Wei-Lun and Su, Yu},
  year = {2025},
  eprint = {2505.23883},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2505.23883}
}
```

**FAISS:**
```bibtex
@article{douze2024faiss,
  title = {The Faiss library},
  author = {Douze, Matthijs and Guzhva, Alexandr and Deng, Chengqi and Johnson, Jeff and Szilvasy, Gergely and Mazar\'{e}, Pierre-Emmanuel and Lomeli, Maria and Hosseini, Lucas and J\'{e}gou, Herv\'{e}},
  year = {2024},
  eprint = {2401.08281},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2401.08281}
}
```


## Acknowledgements

This work was supported by the [Imageomics Institute](https://imageomics.org), which is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) (Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

This work used resources of the [Ohio Supercomputer Center (OSC)](https://www.osc.edu/): Ohio Supercomputer Center. 1987. Ohio Supercomputer Center. Columbus OH: Ohio Supercomputer Center. https://ror.org/01apna436.




## Dataset Card Authors

Net Zhang, Elizabeth Campolongo 


## Dataset Card Contact

For questions or issues, please use the [Community tab](https://huggingface.co/imageomics/bioclip-image-search-lite/discussions) on this repository or [GitHub Issues](https://github.com/Imageomics/bioclip-image-search-lite/issues).
