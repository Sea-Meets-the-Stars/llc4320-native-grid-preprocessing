"""
Configuration dataclasses for the global LLC4320 dataset generation pipeline.

These are used exclusively by ``generate_global.py`` and related global
pipeline tooling.  The cutout / training-data pipeline has its own config
in ``dbof.cutout_dataset_creation.config``.
"""

from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# Pipeline → output folder mapping
# ---------------------------------------------------------------------------

PIPELINE_OUTPUT_FOLDERS = {
    "SURF":  "surface_fields/",
    "OSN":   "surface_fields/",
    "DEPTH": "depth_fields/",
}
"""Default S3 output folder for each pipeline variant.

SURF and OSN produce surface-only diagnostics and share a folder.
DEPTH produces depth-resolved diagnostics and writes to a separate folder.
"""


def default_output_folder(pipeline: str) -> str:
    """Return the default output folder for *pipeline*.

    Parameters
    ----------
    pipeline : str
        One of ``"SURF"``, ``"OSN"``, ``"DEPTH"``.

    Returns
    -------
    str
        Folder path (e.g. ``"surface_fields/"``).

    Raises
    ------
    ValueError
        If *pipeline* is not recognised.
    """
    try:
        return PIPELINE_OUTPUT_FOLDERS[pipeline]
    except KeyError:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'.  "
            f"Expected one of: {list(PIPELINE_OUTPUT_FOLDERS)}"
        )


# ---------------------------------------------------------------------------
# Shared building-block configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    run_id: str
    log_dir: str = "./logs"


@dataclass(frozen=True)
class GlobalDataConfig:
    """Data-source settings for the global pipeline."""
    endpoint_url: str = "https://mghp.osn.xsede.org"
    date_iterations: Optional[List[str]] = None
    k_levels: Optional[List[int]] = None


@dataclass(frozen=True)
class GlobalOutputConfig:
    """S3 output location for generated Zarr stores.

    If ``folder`` is ``None``, it is resolved from the pipeline via
    :func:`default_output_folder` when the :class:`GlobalJobConfig` is
    constructed.
    """
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "dbof/"
    folder: Optional[str] = None
    dataset_name: str = "global.zarr"


@dataclass(frozen=True)
class RuntimeConfig:
    """Dask / zarr concurrency settings."""
    zarr_async_concurrency: int = 128
    # Dask scheduler: "distributed" (default, spawns worker processes),
    # "synchronous" (single-threaded, all RAM in one process — best for
    # memory-constrained jobs), or "threads" (threaded, shared memory).
    dask_scheduler: str = "distributed"
    # Only used when dask_scheduler == "distributed":
    dask_n_workers: Optional[int] = None          # default: one per core
    dask_threads_per_worker: Optional[int] = None  # default: 1
    dask_memory_limit: Optional[str] = None        # e.g. "40GB"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlobalJobConfig:
    """
    Full specification for a global pipeline run.

    Holds everything needed to reproduce the run: which pipeline variant,
    which subsets, which dates, runtime tuning, and output location.
    Channel lists are NOT stored here — they come from
    ``subset_definitions.py`` keyed by ``active_subsets``.
    """
    run: RunConfig
    data: GlobalDataConfig
    output: GlobalOutputConfig
    runtime: RuntimeConfig
    pipeline: str                                   # "SURF", "OSN", or "DEPTH"
    active_subsets: List[str]                        # e.g. ["stratification", "kinematic"]
    depth_suffixes: Optional[List[str]] = None       # override; None = use per-subset defaults

