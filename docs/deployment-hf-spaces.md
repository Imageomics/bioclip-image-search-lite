# Deploying to Hugging Face Spaces

This guide walks through deploying BioCLIP Image Search Lite to [Hugging Face Spaces](https://huggingface.co/spaces). The app is currently live at **[imageomics/bioclip-image-search-lite](https://huggingface.co/spaces/imageomics/bioclip-image-search-lite)**.

## What you need

Three things on Hugging Face, plus one GitHub secret:

| Resource | What it is | Where it lives |
|----------|------------|----------------|
| **HF Space** | Hosts the running Gradio app | `huggingface.co/spaces/<you>/bioclip-image-search-lite` |
| **HF Model Repo** | Stores the large data files (FAISS + DuckDB) | `huggingface.co/<you>/bioclip-image-search-lite` |
| **HF Token** | Authenticates pushes from GitHub Actions | GitHub repo secret `HF_TOKEN` |
| **GitHub Action** | Auto-syncs code on push to `main` | `.github/workflows/sync-to-hf-space.yml` |

## Step-by-step setup

### 1. Create the HF Model Repository (data hosting)

The FAISS index (5.8 GB) and DuckDB (25.8 GB) are too large for a Space's git repo. We store them in a separate **model repo** and use HF's `preload_from_hub` to pull them at build time.

```bash
# From a machine with huggingface_hub installed
hf repo create bioclip-image-search-lite --type model

# Upload data files (uses Git LFS under the hood)
hf upload bioclip-image-search-lite \
    /path/to/index.index faiss/index.index

hf upload bioclip-image-search-lite \
    /path/to/metadata.duckdb duckdb/metadata.duckdb
```

These uploads take a while — the DuckDB alone is ~26 GB. Once uploaded, the data repo should contain:

```
faiss/index.index      (5.8 GB)
duckdb/metadata.duckdb (25.8 GB)
```

### 2. Create the HF Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and create a Space with:

- **SDK**: Gradio
- **Hardware**: CPU basic (free)
- **Visibility**: Public or Private (your choice)

Or via CLI:

```bash
hf repo create bioclip-image-search-lite --type space --space-sdk gradio
```

### 3. Configure the README metadata

The `README.md` YAML header tells HF Spaces how to build and what data to preload:

```yaml
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
```

The `preload_from_hub` lines are key — they tell HF to download the FAISS and DuckDB files from the model repo into the Space's filesystem before the app starts. The app's `config.py` then uses `huggingface_hub.hf_hub_download()` to locate these cached files at runtime.

### 4. Create an HF Token

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **fine-grained** token (not a classic read/write token).

The GitHub Action only needs to `git push` to the Space. The data files are downloaded by HF's own build system via `preload_from_hub` — that works without a token as long as the model repo is public. So the token can be very tightly scoped.

**What to enable:**

Leave all user-level and org-level permissions **unchecked** (Repositories, Inference, Webhooks, Collections, Discussions, Billing, Jobs — all off). The only thing you configure is the **"Repositories permissions"** override section at the bottom:

1. Search for `spaces/imageomics/bioclip-image-search-lite` (or whatever your Space path is)
2. Check these two boxes for that repo:
   - **Read access to contents of selected repos** — needed for the fetch step of `git push`
   - **Write access to contents/settings of selected repos** — the actual push

That's it. Two permissions, one repo. If this token ever leaks, the blast radius is limited to that single Space.

**When migrating to an org** (e.g., `Imageomics`): create a new token scoped to `spaces/Imageomics/bioclip-image-search-lite`, update the GitHub secret, and update the push URL in the workflow.

> **Security tips:**
> - Never commit a token to git. It lives in GitHub Secrets only.
> - If `HF_TOKEN` is set as an environment variable on your local machine, it overrides any cached login token. This has caused confusion before — if pushes fail with a 403, check `echo $HF_TOKEN` first.
> - Rotate the token if it's ever exposed in logs or commits.

### 5. Add the token to GitHub Secrets

In your GitHub repo: **Settings > Secrets and variables > Actions > New repository secret**

- Name: `HF_TOKEN`
- Value: the fine-grained token from step 4

### 6. GitHub Actions auto-sync (prod branch)

The workflow at `.github/workflows/sync-to-hf-space.yml` builds a **`prod` branch** from `main` on every push, stripping out non-deployment files (`scripts/`, `docs/`, `.github/`), then pushes it to both GitHub and the HF Space:

```yaml
name: Sync to Hugging Face Space
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - name: Build and deploy prod branch
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          MAIN_SHA=$(git rev-parse --short HEAD)
          git checkout -b prod-build
          git rm -rf scripts/ docs/ .github/
          git commit -m "prod: built from main@${MAIN_SHA}"
          git push origin prod-build:prod --force
          git push https://netzhang:$HF_TOKEN@huggingface.co/spaces/imageomics/bioclip-image-search-lite prod-build:main --force
```

The `prod` branch on GitHub contains only the files needed for deployment. HF Spaces receives the filtered code on its `main` branch, rebuilds, and restarts automatically.

You can also trigger it manually from the **Actions** tab in GitHub (the `workflow_dispatch` trigger).

## Branch strategy: `main` vs `prod`

We use two branches to keep development and deployment cleanly separated:

- **`main`** — the development branch. Contains everything: app code, scripts, docs, CI workflows. This is where all work happens.
- **`prod`** — the deployment branch. Auto-generated by the GitHub Action from `main`, with `scripts/`, `docs/`, and `.github/` stripped out. You never commit to this branch directly.

The HF Space only sees what's on `prod`. This means SLURM scripts, data conversion tools, and this very document never get shipped to the Space — keeping the deployment footprint small and the build clean.

**What goes where:**

| Branch | Contents | Who writes to it |
|--------|----------|-----------------|
| `main` | `app.py`, `src/`, `scripts/`, `docs/`, `.github/`, `README.md`, etc. | You (and PRs) |
| `prod` | `app.py`, `src/`, `pyproject.toml`, `requirements.txt`, `README.md` | GitHub Actions only |

**How it works in practice:**

1. You push code to `main` (or merge a PR)
2. The GitHub Action checks out `main`, removes non-deployment files, and commits the result
3. That filtered commit is force-pushed to `prod` on GitHub and to `main` on the HF Space
4. HF detects the push, rebuilds, and restarts the Space

The `prod` branch on GitHub is always a single commit — it gets force-pushed each time, so there's no history to maintain. Think of it as a build artifact rather than a traditional branch.

## Where things live once deployed

| What | Location |
|------|----------|
| Live app | `https://imageomics-bioclip-image-search-lite.hf.space` |
| Space settings | `https://huggingface.co/spaces/imageomics/bioclip-image-search-lite/settings` |
| Build & runtime logs | `https://huggingface.co/spaces/imageomics/bioclip-image-search-lite/logs` (visible from the three-dot menu or the "Logs" tab at the bottom of the Space page) |
| Data repo (FAISS + DuckDB) | `https://huggingface.co/imageomics/bioclip-image-search-lite` |
| GitHub Action runs | `https://github.com/Imageomics/bioclip-image-search-lite/actions` |

### Checking logs

The Space logs are the most useful debugging tool. Access them from:

1. Open the Space page
2. Click the three-dot menu (top right) or scroll to the bottom
3. Select **"Logs"** — you'll see two tabs:
   - **Build**: Dependency installation, pip output
   - **Container**: Runtime output — startup messages, search queries, errors

The app logs startup timing, search operations, and image fetch results to stderr, which all appear in the Container log.

## Current deployment: resources and limits

The app runs on the **free tier** (`cpu-basic`):

| Resource | Value |
|----------|-------|
| CPU | 2 vCPU |
| RAM | 16 GB |
| Disk | 50 GB ephemeral (not persistent — wiped on restart) |
| Cost | Free |
| Sleep timeout | 48 hours of inactivity, then the Space goes to sleep |
| Wake-up | Automatic when someone visits, but triggers a full cold start |

### Cold start breakdown

When the Space wakes from sleep (or restarts), it goes through this sequence:

| Phase | What happens | Estimated time |
|-------|-------------|----------------|
| Build | Install Python packages from `requirements.txt` | ~2–3 min |
| Preload | Download FAISS (5.8 GB) + DuckDB (25.8 GB) from the model repo | ~3–5 min |
| Model download | BioCLIP-2 weights (~2.5 GB) via `pybioclip` on first embed | ~1–2 min |
| FAISS + DuckDB init | Load index into memory, connect to DuckDB | ~30 sec |
| **Total cold start** | | **~7–10 min** |

After cold start, the app is responsive. Searches typically complete in 1–3 seconds (FAISS + DuckDB + image fetching).

### Storage budget

The 50 GB ephemeral disk has to hold everything:

| Item | Size |
|------|------|
| FAISS index | 5.8 GB |
| DuckDB metadata | 25.8 GB |
| BioCLIP-2 weights | ~2.5 GB |
| Python packages | ~3 GB |
| OS + Gradio runtime | ~2 GB |
| **Total** | **~40 GB** |

That leaves ~10 GB of headroom. Tight, but workable.

### Traffic limits

HF doesn't publish explicit per-request rate limits for Spaces. In practice, the free tier handles moderate demo traffic fine. The real bottleneck is single-replica concurrency — heavy simultaneous use will queue requests.

## Upgrade options

### Staying on HF Spaces

| Upgrade | Specs | Monthly cost | What it fixes |
|---------|-------|-------------|---------------|
| **CPU Upgrade** | 8 vCPU, 32 GB RAM | ~$22/month | Faster searches, more concurrent users |
| **+ Medium Persistent Storage** | 150 GB at `/data` | +$25/month | Survives restarts — no re-downloading 32 GB of data on cold start |
| **CPU Upgrade + Storage** | Combined | ~$47/month | Best bang-for-buck upgrade: faster + no cold-start data download |
| **T4 Small** (GPU) | 4 vCPU, 15 GB RAM, 16 GB VRAM | ~$290/month | GPU-accelerated embeddings (overkill for this app) |

**The most impactful upgrade** is adding persistent storage. It eliminates the ~5 min data download on every cold start, cutting wake-up time roughly in half.

With persistent storage, set `HF_HOME=/data/.huggingface` in your Space's environment variables to also cache the BioCLIP-2 model weights there.

Notes on paid tiers:
- Billing is per-minute, only while the Space is Running or Starting
- Paid Spaces can configure a custom sleep timeout (or never sleep) — you control the cost
- Persistent storage is billed continuously, even when the Space is sleeping
- You cannot downgrade persistent storage size; you have to delete and recreate it

### Beyond HF Spaces

If you need always-on hosting, lower cold starts, or more control, here are alternatives:

| Platform | Estimated monthly cost | Pros | Cons |
|----------|----------------------|------|------|
| **HF Spaces cpu-upgrade + storage** | ~$47 | Minimal migration, familiar | Still on shared infra, 2-min build on restart |
| **Fly.io** | ~$50–80 | Cheap volumes ($0.15/GB), global edge, always-on | Requires Dockerfile, more ops work |
| **Railway** | ~$70–190 | Dead-simple git deploy | RAM pricing adds up fast at 16 GB |
| **Google Cloud Run** | ~$30–230 | Scales to zero, pay-per-use | 32 GB data doesn't fit in container image; need GCS mount, slow cold starts |
| **Modal** | ~$15–40 | Generous free credits ($30/mo), serverless | Cold start with 32 GB data is painful; better for bursty GPU workloads |
| **AWS Fargate** | ~$120 | Enterprise-grade, EFS for data | Complex setup, most expensive |
| **Render** | ~$180–460 | Simple deploy | Most expensive option for 16+ GB RAM |

**Bottom line**: For a demo/showcase that tolerates occasional cold starts, the free tier is hard to beat. The natural next step is HF Spaces cpu-upgrade + medium persistent storage at ~$47/month. If you need always-on with zero cold starts, Fly.io offers the best price-to-performance for apps with large data files.

## Troubleshooting

**Space stuck on "Building"**: Check the Build logs. Usually a pip dependency conflict or a missing package in `requirements.txt`.

**403 on GitHub Action push**: The `HF_TOKEN` secret is likely expired, read-only, or not scoped to the Space repo. Regenerate a fine-grained write token and update the GitHub secret.

**`HF_TOKEN` env var confusion**: If you have `HF_TOKEN` set in your local shell, it overrides the token from `hf auth login`. This can cause permission errors if the env var holds a read-only token. Run `unset HF_TOKEN` or ensure it matches the write token.

**Out of disk space**: The 50 GB ephemeral disk is nearly full with our ~40 GB of data + deps. If you add large dependencies, you'll hit this. Consider persistent storage or reducing the DuckDB size (see issue #11).

**Space sleeps after 48 hours**: This is expected on the free tier. Visitors who arrive after sleep trigger a cold start. To keep it awake, upgrade to a paid tier and set a longer (or no) sleep timeout.
