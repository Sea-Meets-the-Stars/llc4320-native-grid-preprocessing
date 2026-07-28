#config
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import argparse

from dbof.config import BaseRunConfig

# Channels every snapshot needs: gradb2 drives sampling, SIarea builds the ice mask.
REQUIRED_FEATURE_CHANNELS = ["gradb2", "SIarea"]

@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    """Cutout run config; inherits run_id + log_dir. Extend here as needed."""

@dataclass(frozen=True)
class GridAccessConfig:
    """Rectangular llc4320 grid store written by generate-global-grid."""
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "dbof"
    folder: str = "native_grid_dbof_training_data"
    dataset_name: str = "llc4320_grid.zarr"

@dataclass(frozen=True)
class InputConfig:
    """Generate-global output consumed as cutout input.

    `folder` is the path (under `bucket`) to the directory holding the
    `date_prefix` snapshots — the generate-global run is just part of this
    path, so no run_id/pipeline is needed.  Each requested feature field is
    mapped to the subset store that contains it at read time.
    """
    folder: str                                  # required: path under bucket to the date_prefix snapshots
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "dbof"
    # Timestamps 'YYYYMMDD_HHMMSS' to process.  Omit / leave empty to process
    # every date_prefix found in the source.
    date_prefixes: Optional[List[str]] = None
    grid_access: GridAccessConfig = field(default_factory=GridAccessConfig)

@dataclass(frozen=True)
class SamplingConfig:
    bias_to_high_gradients: float = 2.0
    sample_points_per_snapshot: int = 100

@dataclass(frozen=True)
class OutputConfig:
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "llc/"
    folder: str = "native_grid_dbof_training_data/"
    dataset_name: str = "cutout_dataset_creation.zarr"
    target_km_res: int = 150
    down_sample_res: int = 64

@dataclass(frozen=True)
class FeaturesConfig:
    feature_channels: List[str] = None

    def __post_init__(self):
        if self.feature_channels is None:
            object.__setattr__(self, "feature_channels",
                               ["Eta", "Salt", "Theta", "U", "V", "W", "gradb2", "SIarea"])

@dataclass(frozen=True)
class RuntimeConfig:
    zarr_async_concurrency: int = 128
    # Dask scheduler: "distributed" (default, spawns worker processes),
    # "synchronous" (single-threaded, all RAM in one process — best for
    # memory-constrained jobs), or "threads" (threaded, shared memory).
    dask_scheduler: str = "distributed"
    # Only used when dask_scheduler == "distributed":
    dask_n_workers: Optional[int] = None          # default: one per core
    dask_threads_per_worker: Optional[int] = None  # default: 1
    dask_memory_limit: Optional[str] = None        # e.g. "40GB"

@dataclass(frozen=True)
class JobConfig:
    run: RunConfig
    input: InputConfig
    sampling: SamplingConfig
    output: OutputConfig
    features: FeaturesConfig
    runtime: RuntimeConfig

# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_config(path: str) -> JobConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    def get(section, default=None):
        return raw.get(section, default if default is not None else {})

    input_raw = dict(get("input"))
    if not input_raw.get("folder"):
        raise ValueError("input.folder must point to the generate-global output directory")
    grid_raw = input_raw.pop("grid_access", {})
    input_cfg = InputConfig(**input_raw, grid_access=GridAccessConfig(**grid_raw))

    cfg = JobConfig(
        run=RunConfig(**get("run")),
        input=input_cfg,
        sampling=SamplingConfig(**get("sampling")),
        output=OutputConfig(**get("output")),
        features=FeaturesConfig(feature_channels=raw.get("feature_channels")),
        runtime=RuntimeConfig(**get("runtime")),
    )

    if cfg.sampling.sample_points_per_snapshot <= 0:
        raise ValueError("sampling.sample_points_per_snapshot must be > 0")
    if cfg.output.down_sample_res <= 0:
        raise ValueError("output.down_sample_res must be > 0")

    missing_required = [c for c in REQUIRED_FEATURE_CHANNELS if c not in cfg.features.feature_channels]
    if missing_required:
        raise ValueError(
            f"feature_channels must include required channels {missing_required} "
            f"(gradb2 for sampling, SIarea for the ice mask)"
        )

    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--run_id", default=None, help="Optional override for run.run_id")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data-access config (reading a generated cutout dataset)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataAccessConfig:
    """Locates a generated cutout dataset (zarr + metadata) for reading.

    All fields are required -- the full path to the data must be specified.
    """
    run_id: str
    folder: str
    bucket: str
    s3_endpoint: str
    dataset_name: str


def load_data_access_config(path: str) -> DataAccessConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    da = dict(raw.get("data_access") or {})
    da.pop("feature_channels", None)  # channel order comes from the store
    required = ("run_id", "folder", "bucket", "s3_endpoint", "dataset_name")
    missing = [k for k in required if not da.get(k)]
    if missing:
        raise ValueError(f"data_access is missing required keys: {missing}")
    return DataAccessConfig(**da)
