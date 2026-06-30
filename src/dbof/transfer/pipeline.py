"""Unified LLC4320 -> S3 transfer pipeline.

A single :func:`run` drives both transfer extents; the *only* thing that varies
is the spatial selection.  Everything lives under ``{bucket}/{raw_prefix}/``:

* **full** -- the whole native dataset (all faces, all 720x720 tiles) under the
  subset ``folder`` (``SURFACE`` / ``DEPTH``).
* **chunk** -- a single native 720x720 chunk surrounding a lat/lon
  (``transfer.location``) under ``{chunks_subdir}/{chunk_name}``; every store
  also carries chunk provenance attrs.

Both name their stores identically: ``grid.zarr`` for the static grid and
``{YYYYMMDDTHH}.zarr`` per date (the latter matching what ``generate_global``
reads).  The flow is identical: open the source, resolve the spatial target,
write the time-invariant **static grid once**, then loop over the configured
dates (``data.date_iterations``, or a single ``date_override``) writing one
time-varying store per date.  The per-store open/attrs/write/log is
:func:`dbof.transfer.zarr_io.write_store`.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import xarray as xr

from dbof.transfer import config as config
from dbof.transfer import chunk_selection, zarr_io
from dbof.llc4320_ingestion.date_iterations import (
    DATE_FMT,
    mit_date_to_iteration,
    mit_date_to_time_idx,
)


# ---------------------------------------------------------------------------
# Naming + variable-selection helpers
# ---------------------------------------------------------------------------

def _dataset_name_from_date(date_str: str) -> str:
    """Time-varying store naming (all extents): '2012-11-09 12:00:00' -> '20121109T12.zarr'.

    Matches the name ``generate_global`` reads (see
    ``dbof.llc4320_ingestion.get_raw_data.get_llc_timestep_data``).
    """
    dt = datetime.strptime(date_str, DATE_FMT)
    return dt.strftime("%Y%m%dT%H") + ".zarr"


def _apply_variable_override(static_variables, time_variables, subset,
                             variables_override):
    """Filter the static/time variable lists by a CLI ``--variables`` override."""
    if variables_override is None:
        return static_variables, time_variables

    override_set = set(variables_override)
    if subset in ("static", "all"):
        filtered_static = [v for v in static_variables if v in override_set]
        extra = [v for v in variables_override
                 if v not in filtered_static and v not in time_variables]
        static_variables = filtered_static + extra if subset == "static" else filtered_static
    if subset in ("time", "all"):
        filtered_time = [v for v in time_variables if v in override_set]
        extra = [v for v in variables_override
                 if v not in filtered_time and v not in static_variables]
        time_variables = filtered_time + extra if subset == "time" else filtered_time
    logging.info(f"--variables override active: static={static_variables}, time={time_variables}")
    return static_variables, time_variables


# ---------------------------------------------------------------------------
# Spatial target (full dataset vs single chunk)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Target:
    """Everything that differs between the full and chunk transfer extents."""
    ds: xr.Dataset                       # source, possibly sliced to a chunk
    folder: str                          # S3 folder under the bucket
    tile_j: int
    tile_i: int
    base_attrs: dict                     # root attrs stamped on every store


def _resolve_target(cfg: config.JobConfig, ds: xr.Dataset) -> _Target:
    """Resolve the spatial target for this run (full dataset or one chunk).

    Both extents live under ``{raw_prefix}/...``: the full extent under the
    subset ``folder`` (SURFACE / DEPTH), the chunk extent under
    ``{chunks_subdir}/{chunk_name}``.
    """
    t = cfg.transfer
    out = cfg.output
    source = cfg.data.MIT_data_path
    raw_prefix = out.raw_prefix.strip("/")

    if t.mode == "chunks":
        loc = t.location
        sel = chunk_selection.resolve_chunk(ds, loc.lat, loc.lon, tile=t.tile_i)
        logging.info(
            f"Resolved ({loc.lat}, {loc.lon}) -> face={sel.face_idx}, "
            f"j={sel.j0}:{sel.j0 + sel.tile}, i={sel.i0}:{sel.i0 + sel.tile} "
            f"(nearest cell j={sel.nearest_j}, i={sel.nearest_i} at "
            f"lat={sel.nearest_lat:.4f}, lon={sel.nearest_lon:.4f})"
        )
        folder = f"{raw_prefix}/{out.chunks_subdir.strip('/')}/{loc.chunk_name.strip('/')}"
        return _Target(
            ds=chunk_selection.slice_to_chunk(ds, sel),
            folder=folder,
            tile_j=sel.tile,
            tile_i=sel.tile,
            base_attrs=chunk_selection.chunk_provenance(sel, loc.lat, loc.lon,
                                                        loc.chunk_name, source),
        )

    # Full native dataset -> {raw_prefix}/{folder} (e.g. LLC4320_RAW/SURFACE).
    return _Target(
        ds=ds,
        folder=f"{raw_prefix}/{out.folder.strip('/')}",
        tile_j=t.tile_j,
        tile_i=t.tile_i,
        base_attrs={"source_path": source},
    )


def _resolve_dates(cfg: config.JobConfig, date_override: Optional[str]) -> List[str]:
    """Resolve the list of dates to transfer (single override, or all configured)."""
    if date_override is not None:
        return [date_override]
    if cfg.data.date_iterations:
        return list(cfg.data.date_iterations)
    raise ValueError(
        "No date(s) provided.  Set data.date_iterations in the config or pass "
        "a date_override (CLI --date)."
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    cfg: config.JobConfig,
    init_store: bool = False,
    subset: str = "all",
    skip_existing: bool = False,
    date_override: Optional[str] = None,
    variables_override: Optional[List[str]] = None,
) -> None:
    """Run a transfer for the configured spatial extent and dates.

    Parameters
    ----------
    cfg : JobConfig
        Fully resolved transfer config.  ``transfer.mode`` selects the spatial
        extent ('all' = full dataset, 'chunks' = one lat/lon chunk).
    init_store : bool
        If ``True``, wipe and re-initialise each output store before writing.
    subset : str
        ``"static"`` (grid only), ``"time"`` (time-varying only), or ``"all"``.
    skip_existing : bool
        If ``True``, skip variables already present in the target store.
    date_override : str, optional
        Single date (ISO ``%Y-%m-%d %H:%M:%S``) to process instead of every
        entry in ``data.date_iterations``.
    variables_override : list of str, optional
        Override the configured variable lists (applied to the selected subset).
    """
    t = cfg.transfer
    static_variables = list(t.static_variables or [])
    time_variables = list(t.variables or [])
    static_variables, time_variables = _apply_variable_override(
        static_variables, time_variables, subset, variables_override
    )
    if not static_variables and not time_variables:
        raise ValueError("No variables to transfer (both static and time-varying lists are empty).")

    ds = zarr_io.open_source(cfg.data.MIT_data_path)
    zarr_io.validate_variables_present(ds, static_variables + time_variables)

    target = _resolve_target(cfg, ds)

    # --- Static grid variables: written once -------------------------------
    if static_variables and subset in ("static", "all"):
        s3_url = zarr_io._build_s3_url(cfg.output.bucket, target.folder, t.static_dataset_name)
        zarr_io.write_store(
            target.ds, static_variables,
            s3_url=s3_url, s3_endpoint=cfg.output.s3_endpoint,
            tile_j=target.tile_j, tile_i=target.tile_i, init_store=init_store,
            time_idx=None, attrs=target.base_attrs,
            skip_existing=skip_existing, label="Static grid transfer",
            # Grid 3D vars (hFacC/S/W, masks) chunked one level / all faces, the
            # layout the global grid reader expects (get_llc_depth_gridfile).
            level_chunked_3d=True,
        )

    # --- Time-varying fields: one store per date ---------------------------
    if time_variables and subset in ("time", "all"):
        if "time" not in ds.dims:
            raise ValueError("Source dataset has no 'time' dimension but time-varying variables were requested.")
        ntime = ds.sizes["time"]
        for date_str in _resolve_dates(cfg, date_override):
            iteration = mit_date_to_iteration(date_str)
            time_idx = mit_date_to_time_idx(date_str, ntime)
            s3_url = zarr_io._build_s3_url(cfg.output.bucket, target.folder,
                                           _dataset_name_from_date(date_str))
            zarr_io.write_store(
                target.ds, time_variables,
                s3_url=s3_url, s3_endpoint=cfg.output.s3_endpoint,
                tile_j=target.tile_j, tile_i=target.tile_i, init_store=init_store,
                time_idx=time_idx, skip_existing=skip_existing,
                attrs={
                    **target.base_attrs,
                    "selected_iteration": int(iteration),
                    "selected_date_utc": date_str,
                },
                label=f"Time-varying transfer {date_str}",
                detail=f"iteration={iteration}  time_idx={time_idx}",
            )

    logging.info("All transfers complete.")
