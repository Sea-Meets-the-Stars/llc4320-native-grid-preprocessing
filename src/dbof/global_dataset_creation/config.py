#config
from dataclasses import dataclass
from typing import Optional, List
import yaml
import argparse

@dataclass(frozen=True)
class RunConfig:
    run_id: str
    log_dir: str = "./logs"

@dataclass(frozen=True)
class DataConfig:
    endpoint_url: str = "https://mghp.osn.xsede.org"
    sampling_step: int = 168
    start_record: int = 1180
    timestep_hours: Optional[int] = None
    date_iterations: Optional[List[str]] = None  # explicit timestamps as 'DDMMYYYY-HH:MM:SS'; overrides range logic
    MIT_data_path: Optional[str] = None           # path to MIT LLC4320 zarr store (MIT surface pipeline)
    k_levels: Optional[List[int]] = None         # depth indices to process; None = all 51 levels

@dataclass(frozen=True)
class SamplingConfig:
    bias_to_high_gradients: float = 2.0
    sample_points_per_snapshot: int = 100

@dataclass(frozen=True)
class OutputConfig:
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "llc/"
    folder: str = "native_grid_dbof_training_data/"
    dataset_name: str = "dataset_creation.zarr"
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
    data: DataConfig
    sampling: SamplingConfig
    output: OutputConfig
    features: FeaturesConfig
    runtime: RuntimeConfig


# ---------------------------------------------------------------------------
# Global pipeline config (slimmed down — no sampling, no range-mode fields)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlobalDataConfig:
    endpoint_url: str = "https://mghp.osn.xsede.org"
    date_iterations: Optional[List[str]] = None
    k_levels: Optional[List[int]] = None

@dataclass(frozen=True)
class GlobalOutputConfig:
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "dbof/"
    folder: str = "surface_fields/"
    dataset_name: str = "global.zarr"

@dataclass(frozen=True)
class GlobalJobConfig:
    run: RunConfig
    data: GlobalDataConfig
    output: GlobalOutputConfig
    features: FeaturesConfig
    runtime: RuntimeConfig

