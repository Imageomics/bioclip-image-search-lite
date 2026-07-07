"""Config loader and validator for the index-build pipeline.

Loads scripts/config_example.yaml-shaped YAML files via OmegaConf (for
${interpolation} support) and validates them against a Pydantic v2 schema.

Usage as a CLI:
    python -m scripts.lib.config path/to/config.yaml
        validates the config and prints the resolved (interpolated) form

Usage from a phase script:
    from scripts.lib.config import load_config
    cfg = load_config("path/to/config.yaml")
    print(cfg.input.embeddings_parquet)
    print(cfg.output.index_path)  # interpolation already resolved
    
Design notes:

    OmegaConf is used for config YAML parsing and interpolation resolution. It supports features like:
    - Loading YAML files into nested Python dictionaries
    - Resolving interpolations (e.g. "${input_dir}/file.parquet") automatically
    - Providing helpful error messages for missing interpolations or syntax errors
    
    A Pydantic model is a Python class that:
    - Declares fields with type annotations
    - Automatically validates input data against those types during runtime
    - Provides clear error messages if validation fails (e.g. missing required field, wrong type, constraint violation)
    
    Why use it here? YAML configs are untyped strings. 
    Pydantic converts them to proper Python types and rejects invalid values 
    with a descriptive error before any phase runs. 
    
    Pydantic terms & concepts used in this code:

    `ConfigDict(extra="forbid")`: 
        It controls model level behavior. 
        Here, `extra="forbid"` means that if the input data contains any fields that are not explicitly declared in the model, 
        Pydantic will raise a validation error instead of silently ignoring them.
        For example, if the YAML contains an unexpected field `foo: 123`, 
        the validation will fail with an error about the extra field `foo`,
        A typo in a field name (e.g. `embedding_dim` misspelled as `embedding_dims`) will also trigger this error.
    
    `Field(...)`:
        It constraints and documents individual fields.
        The `...` means the field is required and with no default value.
        You can also specify constraints (e.g. `gt=0` for positive integers) and descriptions for documentation purposes.
        ```py
        embedding_dim: int = Field(..., gt=0)   # required, must be positive
        n_shards:      int = Field(default=1, ge=1)  # optional, min value 1
        ```
        
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ----------------------------------------------------------------------
# Sub-models (one per config section)
# ----------------------------------------------------------------------

class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embeddings_parquet: str = Field(..., description="Glob or path to embedding parquet files.")
    catalog_parquet:    str = Field(..., description="Per-row metadata parquet for DuckDB.")
    embedding_dim:      int = Field(..., gt=0, description="Vector dimensionality.")
    uuid_col:           str = Field(default="uuid")
    embedding_col:      str = Field(default="emb")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir:     str = Field(..., description="Directory where all phase outputs land.")
    index_path:  Optional[str] = Field(default=None)
    duckdb_path: Optional[str] = Field(default=None)


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_total:       int  = Field(..., gt=0)
    stratify_col:  str  = Field(..., description="Column in embeddings parquet to stratify on.")
    engine:        Literal["spark", "duckdb"] = Field(default="spark")
    seed:          int  = Field(default=42)
    min_per_class: int  = Field(default=1, ge=1)


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factory:      str  = Field(default="IVF65536,PQ16")
    metric:       Literal["inner_product", "l2"] = Field(default="inner_product")
    l2_normalize: bool = Field(default=True)


class ShardsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_shards:   int = Field(default=1, ge=1)
    batch_size: int = Field(default=3_000_000, gt=0)


class DuckDBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enum_cardinality_caps: Dict[str, int] = Field(default_factory=dict)
    url_column:            Optional[str]  = Field(default=None)
    sort_by:               List[str]      = Field(..., min_length=1)
    exclude_cols:          List[str]      = Field(
        default_factory=list,
        description=(
            "Catalog columns to drop from the metadata table "
            "(e.g. internal storage paths that should not be surfaced)."
        ),
    )


class VerifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_roundtrip_samples: int = Field(default=200, ge=1)
    roundtrip_topk:      int = Field(
        default=5,
        ge=1,
        description=(
            "Tolerate PQ-quantized rank shuffles by checking self-membership "
            "in top-K rather than strict top-1. K=5 is enough to absorb near-"
            "duplicate noise while still catching real alignment bugs (a "
            "broken uuid->id mapping never lands the queried uuid in any K)."
        ),
    )


class ResourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu:        Optional[int] = Field(default=None, ge=1)
    mem:        Optional[str] = Field(default=None)
    walltime:   str           = Field(default="01:00:00")
    gpu:        int           = Field(default=0, ge=0)
    array_size: Optional[int] = Field(default=None, ge=1)
    partition:  Optional[str] = Field(
        default=None,
        description="SLURM partition. Omit to use the cluster default.",
    )
    account:    Optional[str] = Field(
        default=None,
        description=(
            "SLURM allocation/account to bill against (sbatch --account). "
            "Omit to fall back to SBATCH_ACCOUNT in the calling shell, or "
            "the cluster default."
        ),
    )
    # The two below are only used by multi-node Spark phases. Single-node
    # phases (everything else) leave them at defaults.
    nodes:           Optional[int] = Field(default=None, ge=1)
    tasks_per_node:  Optional[int] = Field(default=None, ge=1)

class ResourcesConfig(BaseModel):
    """Per-phase SLURM resource hints.

    One ResourceSpec per pipeline phase. Consumed only by
    scripts/slurm/submit_pipeline.sh, which reads each phase's spec and
    injects the corresponding sbatch flags (--cpus-per-task, --mem,
    --time, --partition, --nodes, --ntasks-per-node, --gpus, --array)
    when submitting the phase's job.

    The local runner (scripts/run_local.sh) ignores this section
    entirely; everything runs in-process with whatever Python env is
    currently active.

    Each phase's spec is optional (defaults via ResourceSpec). Omitting a
    field means "let SLURM use its cluster default" rather than "set to
    zero" — see ResourceSpec for the behavior of each field.
    """
    model_config = ConfigDict(extra="forbid")

    manifest:     ResourceSpec = Field(default_factory=ResourceSpec)
    sample:       ResourceSpec = Field(default_factory=ResourceSpec)
    train_leader: ResourceSpec = Field(default_factory=ResourceSpec)
    shards:       ResourceSpec = Field(default_factory=ResourceSpec)
    merge:        ResourceSpec = Field(default_factory=ResourceSpec)
    duckdb:       ResourceSpec = Field(default_factory=ResourceSpec)
    verify:       ResourceSpec = Field(default_factory=ResourceSpec)


# ----------------------------------------------------------------------
# Top-level config
# ----------------------------------------------------------------------

class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input:     InputConfig
    output:    OutputConfig
    sample:    SampleConfig
    index:     IndexConfig     = Field(default_factory=IndexConfig)
    shards:    ShardsConfig    = Field(default_factory=ShardsConfig)
    duckdb:    DuckDBConfig
    verify:    VerifyConfig    = Field(default_factory=VerifyConfig)
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    
    # Validator for OutputConfig that fills in defaults for index_path and duckdb_path 
    # based on workdir if they are not provided.
    #
    # A decorator that registers a method as custom vlidation hook for a specific field.
    # Here, it runs after the standard validation of the `output` field, 
    # and fills in defaults for `index_path` and `duckdb_path` based on `workdir` 
    # if they were not provided in the YAML.
    # 
    # `mode="after"` means this runs after the standard validation of the `output` field.
    # guarantees that `output.workdir` is a valid string when this runs.
    #
    # When index_path and duckdb_path are not provided, 
    # they default to `workdir/index.index` and `workdir/metadata.duckdb` respectively.
    # Because OmegaConf has nothing to interpolate as the keys don't exist. 
    # Pydantic sets them to None and after validation, 
    # this validation method fills in the defaults based on workdir.
    @field_validator("output", mode="after")
    @classmethod
    def _fill_output_defaults(cls, v: OutputConfig) -> OutputConfig:
        # workdir-derived defaults are filled here rather than as Pydantic
        # default_factory because they depend on another field's value.
        if v.index_path is None:
            v.index_path = f"{v.workdir.rstrip('/')}/index.index"
        if v.duckdb_path is None:
            v.duckdb_path = f"{v.workdir.rstrip('/')}/metadata.duckdb"
        return v


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config, resolve OmegaConf interpolations, validate via Pydantic.

    Raises:
        FileNotFoundError: if the YAML path does not exist.
        pydantic.ValidationError: with a clear, multi-line message if any
            required field is missing or any value fails its constraint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = OmegaConf.load(path)
    resolved: Dict[str, Any] = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    return PipelineConfig.model_validate(resolved)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m scripts.lib.config <config.yaml>", file=sys.stderr)
        return 2
    cfg = load_config(argv[1])
    print(cfg.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
