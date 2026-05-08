# Index-build pipeline

This directory contains the canonical pipeline that produces the FAISS index and DuckDB metadata lookup served by the BioCLIP Image Search Lite app. The pipeline is dataset-agnostic: it reads embedding parquet files plus a metadata catalog, and emits an aligned `index.index` + `metadata.duckdb` pair. It is the source of truth for how the artifacts published at [imageomics/bioclip-image-search-lite](https://huggingface.co/imageomics/bioclip-image-search-lite) were built.

For background on FAISS itself (index types, training, search), refer to the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) first. This README documents only what is specific to running this pipeline and to the choices we made for the BioCLIP 2 / TreeOfLife-200M use case.

## Who this is for

You'd like to:

- **Reproduce the published artifacts.** You want the same `index.index` and `metadata.duckdb` that ship to HF. Read the [Quick start](#quick-start), use `config_example.yaml` as-is, and run on a multi-GPU SLURM cluster. The [Resource expectations](#resource-expectations) section sets your budget.
- **Train on your own image embeddings on HPC.** Same scripts, different config. Read [Configuration](#configuration) for the knobs that matter, then [Adapting to your scale and data](#adapting-to-your-scale-and-data) for how to retune `nlist`, sample size, and DuckDB columns.
- **Train on a single workstation or laptop.** GPU phases scale down to one consumer card, or to CPU-only. [Adapting to your scale and data](#adapting-to-your-scale-and-data) walks through the decision tree (VRAM, sample size, when to skip the GPU entirely).

All commands below assume CWD is the repository root.

## Pipeline overview

| Phase | Script | Inputs | Outputs |
|---|---|---|---|
| 01 | `pipeline/01_build_manifest.py` | embedding parquets | `manifest.parquet`, `uuid_to_id.parquet/` |
| 02 | `pipeline/02_stratified_sample_{spark,duckdb}.py` | embeddings + manifest | `leader_sample.parquet/` |
| 03 | `pipeline/03_train_leader.py` | leader sample | `leader.index` (centroids + PQ codebooks) |
| 04 | `pipeline/04_build_shards.py` | embeddings + leader + manifest | `shards/shard_NNNNN.index` |
| 05 | `pipeline/05_merge_shards.py` | shards | `index.index` |
| 06 | `pipeline/06_build_duckdb.py` | catalog parquet + uuid_to_id | `metadata.duckdb` |
| 07 | `pipeline/07_verify_alignment.py` | index + duckdb + catalog | exit 0 if aligned, hard-fail otherwise |

Each phase reads its inputs from disk and writes outputs to disk. There is no implicit cross-phase state. Re-running any phase requires only that the previous phase's outputs exist; pass `--force` to rebuild even if outputs already exist.

## Environment

The data-prep environment is **not** the same as the runtime app environment. The reason is `faiss-gpu-cuvs`, which provides GPU-accelerated index training and is [conda-only](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda). The runtime app uses `faiss-cpu` from PyPI, which is enough for serving queries against an already-built index.

| File | Use | Channels |
|---|---|---|
| `environments/faiss-gpu-cuvs.yml` | Build pipeline (default) | `pytorch -c nvidia -c rapidsai -c conda-forge` |
| `environments/faiss-cpu.yml` | Build pipeline on a CPU-only machine | `conda-forge` |

CPU-only is fine for datasets up to a few million vectors. At hundreds of millions, GPU training is effectively required, since k-means on the leader sample would otherwise dominate wall time.

### Activating the env on SLURM compute nodes

The slurm templates do **not** hardcode any `conda activate` or `module load` lines, so they remain portable across clusters. Activation is delegated to a shell script you point at via the `ENV_SETUP` environment variable. Each `*.slurm` template starts with:

```bash
[[ -n "${ENV_SETUP:-}" ]] && source "$ENV_SETUP"
```

`scripts/slurm/submit_pipeline.sh` propagates `ENV_SETUP` to every job via `sbatch --export=ALL,ENV_SETUP`, so you set it once in your shell and every phase inherits it:

```bash
export ENV_SETUP=/path/to/your/env_setup.sh
bash scripts/slurm/submit_pipeline.sh path/to/config.yaml
```

`scripts/environments/env_setup_example.sh` is a starting point with four commented blocks (conda env from yml, conda at absolute path, plain venv, container). Pick one, fill in paths, save it somewhere (e.g. `~/bioclip-lite-env.sh`), then point `ENV_SETUP` at it.

## Quick start

### Run locally (single GPU or CPU)

```bash
conda env create -f scripts/environments/faiss-gpu-cuvs.yml
conda activate bioclip-lite-build
bash scripts/run_local.sh path/to/config.yaml
```

### Run on SLURM

Submits all 7 phases with `--dependency=afterok` chaining. Reads sbatch params (cpu, mem, walltime, gpu, array_size) from `config.yaml`'s `resources` section.

```bash
export ENV_SETUP=/path/to/your/env_setup.sh
bash scripts/slurm/submit_pipeline.sh path/to/config.yaml
```

### Run a single phase

```bash
python scripts/pipeline/04_build_shards.py --config path/to/config.yaml
```

Useful for debugging or partial reruns.

## Configuration

The minimum config you have to write is seven fields. Everything else has sensible defaults:

```yaml
input:
  embeddings_parquet: "/data/embeddings/*.parquet"
  catalog_parquet:    "/data/catalog.parquet"
  embedding_dim:      768
output:
  workdir: "/work/lite_index"
sample:
  n_total: 15_000_000
  stratify_col: class
duckdb:
  sort_by: [source_dataset, kingdom, phylum, class, order, family, common_name]
```

For the full annotated reference with every tunable, see [`config_example.yaml`](./config_example.yaml).

### The choices that matter

For FAISS theory, refer to the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) first. The notes below explain the specific values our defaults use, so you can adjust them intelligently for your N, D, and hardware.

#### `index.factory = "IVF65536,PQ16"`

- **`nlist=65536`** is `~4 * sqrt(233M)`, near the low end of the [FAISS-recommended][faiss-choose] band of `4 * sqrt(N)` to `16 * sqrt(N)`. We chose it for its small centroid table (192 MiB) and cheap coarse-quantizer assignment, paired with `nprobe=16` at search time. (`nprobe` is the search-time knob that sets how many of the `nlist` cells each query visits; more cells gives higher recall at the cost of higher latency. The same knob is referenced throughout this README.) Re-tune for your N using the same formula.
- **`PQ16`** compresses each 768-dim vector into 16 sub-quantizer codes (1 byte each), so 16 bytes per vector. The constraint is `D % M == 0`; for `D=768` valid M values include 8, 12, 16, 24, 32, 48, 64. 16 bytes is a common production sweet spot. See [FAISS wiki: IVFPQ][faiss-pq] for the theory.
- **Inner product on L2-normalized vectors** is mathematically equivalent to cosine similarity, and matches what BioCLIP 2 (and most CLIP-family models) was trained to produce. Leave both at default unless your model uses a different similarity geometry.

For smaller or recall-critical corpora, alternatives like `Flat` and `IVF<n>,Flat` trade memory for higher recall; see [Picking a factory string](#picking-a-factory-string) under Adapting.

#### `sample.n_total = 15_000_000`

The leader-training k-means step needs `~256 * nlist` vectors for well-converged centroids ([FAISS FAQ][faiss-train-size]). For `nlist=65536` that targets ~16.8M; we used 15M, slightly below, which worked well for our corpus. Going above 20M is fine; going below 5M starts to degrade recall noticeably. PQ codebook training has a much smaller requirement that the IVF target always covers.

#### `duckdb.sort_by`

This is the single biggest knob for DuckDB on-disk size. ZSTD, dictionary, and run-length encoders win when adjacent rows share long byte sequences, so pick a hierarchy that puts similar rows adjacent. For our build, switching from random order to `(source_dataset, kingdom, phylum, ..., common_name)` cut the DuckDB file from 27 GB to 14 GB.

#### `duckdb.enum_cardinality_caps`

DuckDB ENUMs are dictionary-encoded: each value becomes a small integer code, plus one string lookup table. Big space win at low cardinality. The trap is that ENUM creation scans every distinct value at build time and stores them in the column's type signature; if a column you thought was low-cardinality actually has thousands of distinct values, you end up with a multi-MB type definition, slow CREATE, and the storage savings disappear. The cap is the threshold above which a column falls back to VARCHAR. Set per-column based on natural cardinality (`source_dataset` ~5, `publisher` ~2k, `family` ~15k).

## Resource expectations

Wall time and peak RSS measured from the production build of the 233,055,986-vector / 768-dim TreeOfLife-200M index on the Ohio Supercomputer Center [cardinal cluster](https://www.osc.edu/resources/technical_support/supercomputers/cardinal) (April 2026):

| Phase | Wall time | Peak RSS | Compute |
|---|---|---|---|
| 01 manifest | 2 min | 28 GB | 32 cpu |
| 02 stratified_sample (spark) | 3 min | 357 GB cluster-wide | 4 nodes x 96 cpu |
| 03 train_leader | 3 min | 106 GB | 1 H100, 40 cpu |
| 04 build_shards (array of 8) | 6 min | 25-31 GB per task | 8 H100s in parallel |
| 05 merge | 3 min | 12 GB | 8 cpu |
| 06 build_duckdb | 5 min | 83 GB | 16 cpu |
| 07 verify | <1 min | <1 GB | 8 cpu |

End-to-end compute is around 25 minutes when the cluster has resources free; queue wait dominates real wall clock and varies by site. The `resources` section of `config_example.yaml` already includes safety margin on walltimes. Where to optimize for your own scale:

- **Phase 03 is GPU-VRAM-bound.** The leader-training k-means works against the full sample tensor (`sample.n_total * D * 4` bytes; 46 GB for our 15M × 768 float32 sample). The 80 GB H100 has comfortable headroom; below ~24 GB VRAM you must shrink either `nlist` or `sample.n_total`. See [Adapting to your scale and data](#adapting-to-your-scale-and-data).
- **Phase 04 is GPU-bound but streaming.** Wall clock scales linearly with GPU count up to one shard per GPU. With 8 H100s the 233M-vector encode finished in 6 minutes; on a single consumer GPU expect 1 to 2 hours. Per-batch VRAM is `batch_size * D * 4` bytes plus the loaded leader index (a few hundred MB).
- **Phase 06 dominates per-node memory** (83 GB peak for 233M rows × 19 columns). For datasets above ~500M rows or hosts under 200 GB RAM, partition the catalog and process it in chunks (a future optimization).
- **Phases 01, 05, 07 are CPU+IO bound** and small; defaults work.
- **Phase 02 (spark)** scales out cleanly; raise `spark.executor.instances` if your stratify column has heavy skew.

### Validated on smaller corpora

The pipeline scales down cleanly. End-to-end timings on the full TOL-200M and two of its source subsets (EOL and FathomNet):

| Dataset | Factory | Hardware | End-to-end |
|---|---|---|---:|
| TOL-200M (233M)   | `IVF65536,PQ16` | 8x H100 | ~25 min |
| EOL (5.2M)        | `IVF8192,PQ16`  | 1x H100 | 2:43 |
| EOL (5.2M)        | `IVF8192,PQ16`  | 8 cpu, no GPU | 10:09 |
| FathomNet (38k)   | `IVF256,PQ16`   | 1x H100 | 0:08 |
| FathomNet (38k)   | `IVF256,PQ16`   | 8 cpu, no GPU | 0:30 |

GPU advantage concentrates in phases 03 (k-means) and 04 (encode + add); other phases are CPU/IO bound and indifferent. The corpus-size threshold for whether GPU is worth provisioning is in [When the GPU is worth provisioning](#when-the-gpu-is-worth-provisioning) below.

## Adapting to your scale and data

The pipeline is built for biological / taxonomic data but nothing is hardcoded to that domain. The dominant axes are factory choice, N (corpus size), and your hardware; secondary swaps are columns and metric.

### Picking a factory string

The default `IVF65536,PQ16` is sized for the TOL-200M production corpus. For smaller or recall-critical corpora, two alternatives are worth knowing about:

| Factory | When | Storage / vec (768-dim) | Recall | Search time per query |
|---|---|---:|---|---|
| `Flat` | N < ~5M, or exact recall needed | 3,072 B | 100% | O(N), full scan of all vectors |
| `IVF<nlist>,Flat` | Mid-scale (1M-50M), full recall preferred | 3,072 B | ~99% (IVF approximation only) | ~ O((nprobe/nlist) * N), uncompressed cells |
| `IVF<nlist>,PQ16` (default) | >= 50M, memory-constrained | 16 B | ~90-95% | ~ O((nprobe/nlist) * N), PQ-compressed (lowest per-vector cost) |

The complexities follow [FAISS wiki: indexes][faiss-pq]: `Flat` decodes every vector sequentially; IVF-based factories visit only `nprobe / nlist` of the corpus on average. PQ further reduces per-vector cost by computing distances directly on the compressed codes. Absolute latency depends on hardware and `nprobe`; FAISS does not publish a direct A/B benchmark across the three factories. See [FAISS wiki: choosing an index][faiss-choose] for the full menu.

Quick rule by N:

1. **N < 5M**, no disk pressure: use `Flat`. No training, no `nprobe`. Skip Step 1 and Step 2 below.
2. **N < 50M**, RAM headroom, recall priority: use `IVF<4*sqrt(N)>,Flat`.
3. **N >= 50M**, memory-constrained: use `IVF<4*sqrt(N)>,PQ16` (the production default).
4. **Ground-truth evaluation**: use `Flat`, regardless of N.

The pipeline supports any of these via the single `index.factory` config string; the rest of the workflow is identical.

### Scaling down: smaller corpora and consumer GPUs

The defaults assume an H100-class node (80 GB VRAM, hugemem-class CPU host). To adapt for a single workstation or smaller cluster, work through the questions below in order. You'll need:

1. `N` — total number of vectors in your corpus.
2. `D` — embedding dimensionality (e.g. 768 for BioCLIP 2 ViT-L/14).
3. `VRAM_GB` — VRAM on your single GPU.
4. `RAM_GB` — host CPU RAM (relevant for the CPU fallback).

#### Step 1: pick `nlist`

Same formula as the canonical default, sized for your N:

| N (corpus) | suggested `nlist` band | typical pick |
|---:|---|---:|
| 1M          | 4k – 16k       | `IVF4096`   |
| 10M         | 12k – 50k      | `IVF8192`   |
| 100M        | 40k – 160k     | `IVF65536`  |
| 233M (prod) | 60k – 240k     | `IVF65536`  |

The picks (4096, 8192, 65536) are powers of two by convention; FAISS's `IndexIVF` is happiest there and the `IVF<n>` factory string parses cleanly. Round to the nearest power of two within the suggested band; see [FAISS wiki: choosing an index][faiss-choose].

Two trade-offs the table compresses, both worth understanding before tuning:

- **Build cost vs. search cost.** Larger `nlist` means more centroids to learn (slower phase 03, bigger centroid table) but each cell holds fewer vectors, so individual searches are faster. Smaller `nlist` flips the trade.
- **Search vs. brute force.** At search time you visit `nprobe` of the `nlist` cells; the ratio `nprobe / nlist` is roughly the fraction of the corpus you actually scan. As `nprobe` approaches `nlist`, you scan everything, which is brute force: perfect recall, no speedup. The point of IVF is keeping `nprobe / nlist` small while keeping recall high enough.

So lowering `nlist` saves on the centroid table and assignment cost, but typically forces you to raise `nprobe` at search time to maintain recall, which moves the search closer to brute force. As a rule of thumb, doubling `nprobe` recovers most of the recall lost when you halve `nlist`. Above the upper bound of the band, training cost outpaces benefit. For very large N see [FAISS: indexing 1G vectors][faiss-1g].

#### Step 2: decide whether to sample

- If `N <= 256 * nlist`: don't sample. Set `sample.n_total = N` and the sampler returns the input as-is.
- If `N > 256 * nlist`: sample. Set `sample.n_total ≈ 256 * nlist`. Below `30 * nlist` triggers FAISS warnings and degrades centroid quality.

For `nlist=8192`, that targets a sample of about 2M vectors.

#### Step 3: check the training tensor against VRAM

cuVS-FAISS k-means runs fastest when the sample is resident in VRAM. Required tensor size is `sample.n_total * D * 4` bytes; compare against `VRAM_GB - 2` (a couple of GB headroom for index structures and cuVS workspace). If it doesn't fit:

1. **Lower `nlist`.** Halving `nlist` roughly halves the training tensor. Compensate at search time by raising `nprobe`.
2. **Train phase 03 on CPU instead of GPU.** k-means runs fine on CPU RAM (typically 64-256 GB on a workstation, far more than any consumer GPU). It's 5-20x slower wall clock but bypasses VRAM entirely. Force CPU via `CUDA_VISIBLE_DEVICES= python scripts/pipeline/03_train_leader.py ...` or use the `faiss-cpu.yml` env. Phases 04 onward can still use the GPU.

#### Step 4: pick `batch_size` for phase 04

Per-batch VRAM is `batch_size * D * 4` bytes plus the loaded leader index (a few hundred MB at our `nlist`):

```
batch_size <= (VRAM_GB - 2 - leader_size_GB) / (D * 4 bytes)
```

For `D=768`, plan on a `batch_size` of 500k-1M on a 16 GB card and 3M on an 80 GB H100.

#### Step 5: set `shards.n_shards`

`n_shards = 1` for a single GPU. Phase 04 then walks the manifest sequentially. With multiple GPUs, set `n_shards` equal to the GPU count and submit phase 04 as a SLURM array; each task handles one slice of the manifest in parallel.

#### When the GPU is worth provisioning

GPU acceleration concentrates in phases 03 (k-means) and 04 (encode + add). Validation runs on EOL (5.2M) and FathomNet (38k) confirm the advantage scales with corpus size:

| Corpus size | GPU benefit | What to do |
|---|---|---|
| < ~100k | Negligible | CPU is fine; GPU setup cost (kernel JIT, CUDA context) outweighs savings. Use `faiss-cpu.yml`. |
| ~100k to ~5M | Modest (1.5-4x total) | Either works; use what's available. |
| ~5M to ~50M | Meaningful (5-20x)    | GPU recommended; CPU-only is realistic but slow on phases 03/04. |
| > ~50M | Effectively required | CPU phase 04 alone takes hours; production runs need a GPU array. |

A single 16 GB consumer GPU handles everything up to ~50M vectors at 768-dim with `IVF8192,PQ16`. Above that, an 80 GB H100 or a SLURM array of smaller GPUs.

### Adapting columns to your domain

The pipeline defaults to taxonomic biological data; swap by setting:

- **`sample.stratify_col`**: any column in your embeddings parquet that defines a fair sampling axis (taxonomic class, source dataset, year, etc.).
- **`duckdb.sort_by`**: a hierarchy that puts similar rows adjacent.
- **`duckdb.enum_cardinality_caps`**: list whichever low-cardinality columns you have. Empty dict is fine if you have none.
- **`duckdb.url_column`**: your URL column name, or `null` if your catalog has no URLs.

## After the pipeline

Phase 07 confirms that the FAISS index and DuckDB metadata are 1:1 aligned. Once it exits 0, you have two artifacts in `output.workdir`:

- `index.index` — the FAISS index, loaded by the runtime app's `SearchService`.
- `metadata.duckdb` — the metadata table, looked up by FAISS-returned ids.

To run the [BioCLIP Image Search Lite app](https://github.com/Imageomics/bioclip-image-search-lite) locally against these artifacts, **switch out of the conda build env into the runtime app's `uv` venv at the repo root**. The runtime needs `faiss-cpu` from PyPI (lightweight, no CUDA), not the conda `faiss-gpu-cuvs` we used to build the index. Setup is in the [project README](../README.md#environment-setup):

```bash
# deactivate the conda build env if it's still active
conda deactivate

# from the repo root, activate the runtime venv
source .venv/bin/activate     # or wherever you created it with `uv venv`

python app.py \
    --faiss-index path/to/workdir/index.index \
    --duckdb-path path/to/workdir/metadata.duckdb
```

To publish them to a Hugging Face model repo (which is how the lite app preloads them on Spaces), see [`docs/deployment-hf-spaces.md`](../docs/deployment-hf-spaces.md). The short form, using the modern `hf` CLI:

```bash
hf upload <your-org>/<your-repo> path/to/workdir/index.index    faiss/index.index
hf upload <your-org>/<your-repo> path/to/workdir/metadata.duckdb duckdb/metadata.duckdb
```

For full schema details, scope-filter behavior, and citation info, see [`docs/hf-data-card-README.md`](../docs/hf-data-card-README.md).

## References

- FAISS wiki:
  - [Guidelines to choose an index][faiss-choose]
  - [Faiss indexes (IVFPQ)][faiss-pq]
  - [How big is the dataset (training size)][faiss-train-size]
  - [Indexing 1G vectors][faiss-1g]
  - [FAISS on the GPU][faiss-gpu]
- Project artifacts: [imageomics/bioclip-image-search-lite](https://huggingface.co/imageomics/bioclip-image-search-lite)
- Data card: [`docs/hf-data-card-README.md`](../docs/hf-data-card-README.md)
- HF deployment guide: [`docs/deployment-hf-spaces.md`](../docs/deployment-hf-spaces.md)
- Runtime app source: [Imageomics/bioclip-image-search-lite](https://github.com/Imageomics/bioclip-image-search-lite)

[faiss-choose]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
[faiss-pq]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#indexivfpq
[faiss-train-size]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
[faiss-gpu]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
[faiss-1g]: https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors
