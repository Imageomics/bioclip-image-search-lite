# scripts/environments/env_setup_example.sh
#
# Pick one of the blocks below, copy this file to a location of your choice
# (e.g. ~/bioclip-lite-env.sh), uncomment the block that matches your setup,
# adjust paths, and then point ENV_SETUP at it before submitting:
#
#   ENV_SETUP=~/bioclip-lite-env.sh \
#     bash scripts/slurm/submit_pipeline.sh path/to/config.yaml
#
# Each phase script (scripts/pipeline/NN_*.py) expects `python` on PATH to
# resolve to a Python with: faiss (faiss-gpu-cuvs preferred for phases 03/04,
# faiss-cpu acceptable everywhere else), pyarrow, duckdb, polars, omegaconf,
# pydantic, and pyspark when sample.engine is "spark".


# --- Option A: Conda env from scripts/environments/faiss-gpu-cuvs.yml -----
# Build once with:
#   conda env create -f scripts/environments/faiss-gpu-cuvs.yml
# Then this block activates it on every SLURM compute node.
#
# module load miniconda3            # or whatever your cluster uses
# eval "$(conda shell.bash hook)"
# conda activate bioclip-lite-build


# --- Option B: Conda env at an absolute path (HPC shared install) ---------
# Useful when conda envs live outside the user's home directory.
#
# module load cuda/12.6.2
# module load miniconda3/24.1.2-py310
# eval "$(conda shell.bash hook)"
# conda activate /path/to/shared/conda-envs/bioclip-lite-build


# --- Option C: Plain venv (no conda) --------------------------------------
# CPU-only or single-node setups; works for small datasets where
# faiss-gpu-cuvs is not required.
#
# source /path/to/venv/bin/activate


# --- Option D: Apptainer / Singularity container --------------------------
# Wrap the whole job in a container exec. Requires the slurm template to be
# adapted to delegate to apptainer (not done by default).
#
# module load apptainer
# # Then template needs: apptainer exec --nv /path/to/img.sif python ...
