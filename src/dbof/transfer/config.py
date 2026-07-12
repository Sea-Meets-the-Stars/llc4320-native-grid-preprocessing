"""Standalone configuration for the LLC4320 transfer pipeline.

This module defines its own dataclasses and reuses only
the shared :class:`~dbof.config.BaseRunConfig` (``run_id`` + ``log_dir``).

A single YAML drives both transfer modes; ``transfer.mode`` selects between
them::

    run:        run_id, log_dir
    data:       MIT_data_path, date_iterations, endpoint_url
    output:     s3_endpoint, bucket, raw_prefix, folder, chunks_subdir
    runtime:    zarr_async_concurrency, dask_scheduler
    transfer:   mode, variables, static_variables, tile_j, tile_i,
                static_dataset_name, location{lat,lon,chunk_name}

Both modes transfer the dates in ``data.date_iterations`` (or a single CLI
``--date``); ``transfer.location`` selects the chunk extent for mode 'chunks'.
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

    ``date_iterations`` (ISO ``'YYYY-MM-DD HH:MM:SS'``) is the list of dates to
    transfer; both the full and chunk extents loop over it.
    """
    MIT_data_path: str = ""
    date_iterations: Optional[List[str]] = None
    endpoint_url: str = "https://mghp.osn.xsede.org"


@dataclass(frozen=True)
class OutputConfig:
    """S3 output location.

    All raw transfers live under ``{bucket}/{raw_prefix}/``::

        full  mode: {bucket}/{raw_prefix}/{folder}/{grid.zarr | YYYYMMDDTHH.zarr}
        chunk mode: {bucket}/{raw_prefix}/{chunks_subdir}/{chunk_name}/{...}

    For the full extent ``folder`` is the subset directory (``SURFACE`` /
    ``DEPTH``) that downstream pipelines (e.g. generate_global) read from.
    """
    s3_endpoint: str = "https://s3-west.nrp-nautilus.io"
    bucket: str = "dbof/"
    raw_prefix: str = "LLC4320_RAW"     # top-level prefix shared by all transfers
    folder: str = ""                    # full mode subdir: SURFACE | DEPTH
    chunks_subdir: str = "CHUNKS"       # chunk mode subdir under raw_prefix


@dataclass(frozen=True)
class LocationConfig:
    """A lat/lon point that resolves to one native 720x720 chunk.

    ``chunk_name`` is the human-readable directory label under
    ``{raw_prefix}/{chunks_subdir}`` (e.g. ``"monterey_bay"``).  The resolved
    face and face-local slices are recorded as store attributes for provenance.
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
            "'location' for mode 'chunks')."
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
    elif not cfg.output.folder:
        raise ValueError(
            f"output.folder (e.g. 'SURFACE' or 'DEPTH') is required for "
            f"mode 'all' in {path}."
        )
