"""Standalone configuration for the LLC4320 transfer pipeline.

This module defines its own dataclasses and reuses only
the shared :class:`~dbof.config.BaseRunConfig` (``run_id`` + ``log_dir``).

A single YAML drives both transfer modes; ``transfer.mode`` selects between
them::

    run:        run_id, log_dir
    data:       MIT_data_path, date_iterations, endpoint_url
    output:     s3_endpoint, bucket, folder, chunks_prefix
    runtime:    zarr_async_concurrency, dask_scheduler
    transfer:   mode, variables, static_variables, tile_j, tile_i,
                static_dataset_name, location{lat,lon,chunk_name}, timestamps
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional

import yaml

from dbof.config import BaseRunConfig


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig(BaseRunConfig):
    """Run identity for a transfer run (inherits ``run_id`` + ``log_dir``)."""


@dataclass(frozen=True)
class DataConfig:
    """Source-data settings.

    ``date_iterations`` is only used by the ``all`` mode; the ``chunks`` mode
    uses ``transfer.timestamps`` instead.
    """
    MIT_data_path: str = ""
    date_iterations: Optional[List[str]] = None
    endpoint_url: str = "https://mghp.osn.xsede.org"


@dataclass(frozen=True)
class OutputConfig:
    """S3 output location."""
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "dbof/"
    # all-data mode: {bucket}/{folder}/{grid.zarr | YYYYMMDDTHH.zarr}
    folder: str = "LLC4320"
    # chunks mode: {bucket}/{chunks_prefix}/{chunk_name}/{grid.zarr | timestamp}
    chunks_prefix: str = "LLC_CHUNKS_RAW"


@dataclass(frozen=True)
class LocationConfig:
    """A lat/lon point that resolves to one native 720x720 chunk.

    ``chunk_name`` is the human-readable directory label under
    ``chunks_prefix`` (e.g. ``"monterey_bay"``).  The resolved face and
    face-local slices are recorded as store attributes for provenance.
    """
    lat: float
    lon: float
    chunk_name: str


@dataclass(frozen=True)
class TransferConfig:
    """Transfer-specific settings (top-level ``transfer:`` YAML section)."""
    mode: str = "all"                       # "all" | "chunks"
    static_variables: List[str] = field(default_factory=list)
    static_dataset_name: str = "grid.zarr"
    variables: List[str] = field(default_factory=list)
    tile_j: int = 720
    tile_i: int = 720
    # chunks-mode only:
    location: Optional[LocationConfig] = None
    timestamps: Optional[List[str]] = None  # ISO 'YYYY-MM-DD HH:MM:SS'


@dataclass(frozen=True)
class RuntimeConfig:
    """Dask / zarr concurrency settings."""
    zarr_async_concurrency: int = 128
    # "synchronous" is safest for single-timestep / single-chunk transfers.
    dask_scheduler: str = "synchronous"


@dataclass(frozen=True)
class JobConfig:
    run: RunConfig
    data: DataConfig
    output: OutputConfig
    runtime: RuntimeConfig
    transfer: TransferConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _only_known(cls, raw: dict) -> dict:
    """Keep only keys that are fields of dataclass *cls*.

    Tolerates legacy / unused keys in the YAML (e.g. the old ``data`` section
    carried ``sampling_step`` / ``start_record``) without raising.
    """
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (raw or {}).items() if k in names}


def load_config(path: str) -> JobConfig:
    """Load a transfer YAML into a :class:`JobConfig`."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    transfer_raw = dict(raw.get("transfer", {}))
    if not transfer_raw:
        raise ValueError(
            f"Missing 'transfer' section in {path}. "
            "Expected at least 'variables' (and 'static_variables' / "
            "'location' / 'timestamps' depending on mode)."
        )
    loc_raw = transfer_raw.pop("location", None)
    location = LocationConfig(**loc_raw) if loc_raw else None

    cfg = JobConfig(
        run=RunConfig(**_only_known(RunConfig, raw.get("run", {}))),
        data=DataConfig(**_only_known(DataConfig, raw.get("data", {}))),
        output=OutputConfig(**_only_known(OutputConfig, raw.get("output", {}))),
        runtime=RuntimeConfig(**_only_known(RuntimeConfig, raw.get("runtime", {}))),
        transfer=TransferConfig(
            location=location,
            **_only_known(TransferConfig, transfer_raw),
        ),
    )

    _validate(cfg, path)
    return cfg


def _validate(cfg: JobConfig, path: str) -> None:
    """Mode-aware sanity checks."""
    if cfg.transfer.mode not in ("all", "chunks"):
        raise ValueError(
            f"transfer.mode='{cfg.transfer.mode}' in {path}; "
            "expected 'all' or 'chunks'."
        )
    if not cfg.data.MIT_data_path:
        raise ValueError(f"data.MIT_data_path must be set in {path}.")

    if cfg.transfer.mode == "chunks":
        if cfg.transfer.location is None:
            raise ValueError(
                f"transfer.location (lat, lon, chunk_name) is required for "
                f"mode 'chunks' in {path}."
            )
        if not cfg.transfer.timestamps:
            raise ValueError(
                f"transfer.timestamps (list of ISO 'YYYY-MM-DD HH:MM:SS') is "
                f"required for mode 'chunks' in {path}."
            )
