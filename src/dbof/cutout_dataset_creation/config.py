#config
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import argparse

from dbof.config import BaseRunConfig

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

#TODO jake fix downstream issues in cutouts logic
@dataclass(frozen=True)
class FeaturesConfig:
    model_data_feature_channels: List[str] = None
    compute_features_channels: List[str] = None

    def __post_init__(self):
        # dataclasses "frozen" workaround for defaults
        if self.model_data_feature_channels is None:
            object.__setattr__(self, "model_data_feature_channels",
                               ["Eta", "Salt", "Theta", "U", "V", "W"])
        if self.compute_features_channels is None:
            object.__setattr__(self, "compute_features_channels", [])
            #object.__setattr__(self, "compute_features_channels", ["log_gradb"])

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
        features=FeaturesConfig(**get("features")),
        runtime=RuntimeConfig(**get("runtime")),
    )

    if cfg.sampling.sample_points_per_snapshot <= 0:
        raise ValueError("sampling.sample_points_per_snapshot must be > 0")
    if cfg.output.down_sample_res <= 0:
        raise ValueError("output.down_sample_res must be > 0")

    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--run_id", default=None, help="Optional override for run.run_id")
    return p.parse_args()
