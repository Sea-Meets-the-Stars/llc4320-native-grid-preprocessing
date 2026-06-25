"""All-data transfer mode (the original ``transfer_llc4320`` behaviour).

Transfers every spatial tile of a timestep:

* **Static grid variables** (geometry, masks, vertical coords) -> written once
  to ``{folder}/{static_dataset_name}`` (default ``grid.zarr``).
* **Time-varying fields** (Theta, Salt, ...) -> written per-date to
  ``{folder}/{YYYYMMDDTHH}.zarr``.

All read/write/verify logic lives in :mod:`dbof.transfer.zarr_io`; this module
only orchestrates which variables go to which store for the requested
date/subset, and owns the all-data store-naming convention
(:func:`_dataset_name_from_date`).
"""

import logging
from datetime import datetime
from typing import List, Optional

from dbof.transfer import config as config
from dbof.transfer import zarr_io
from dbof.llc4320_ingestion.date_iterations import (
    DATE_FMT,
    mit_date_to_iteration,
    mit_date_to_time_idx,
)


def _dataset_name_from_date(date_str: str) -> str:
    """All-data store naming: '2012-11-09 12:00:00' -> '20121109T12.zarr'."""
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


def run(
    cfg: config.JobConfig,
    date_str: str,
    init_store: bool = False,
    subset: str = "all",
    skip_existing: bool = False,
    variables_override: Optional[List[str]] = None,
) -> None:
    """Execute an all-data transfer for a single date.

    Parameters
    ----------
    cfg : JobConfig
        Fully resolved transfer config.
    date_str : str
        Date for time-varying fields (ISO ``%Y-%m-%d %H:%M:%S``).
    init_store : bool
        If ``True``, wipe and re-initialise output stores.
    subset : str
        ``"static"`` (grid only), ``"time"`` (time-varying only), or ``"all"``.
    skip_existing : bool
        If ``True``, skip variables already present in the target store.
    variables_override : list of str, optional
        Override the configured variable lists (applied to the selected subset).
    """
    t = cfg.transfer
    source = cfg.data.MIT_data_path

    static_variables = list(t.static_variables or [])
    time_variables = list(t.variables or [])
    static_variables, time_variables = _apply_variable_override(
        static_variables, time_variables, subset, variables_override
    )
    tile_j, tile_i = t.tile_j, t.tile_i

    if not static_variables and not time_variables:
        raise ValueError("No variables to transfer (both static and time-varying lists are empty).")

    ds = zarr_io.open_source(source)
    zarr_io.validate_variables_present(ds, static_variables + time_variables)

    # --- Static grid variables ---------------------------------------------
    if static_variables and subset in ("static", "all"):
        s3_url = zarr_io._build_s3_url(cfg.output.bucket, cfg.output.folder, t.static_dataset_name)
        logging.info(f"--- Static grid transfer: {len(static_variables)} variables -> {s3_url} ---")
        root = zarr_io.open_zarr_store(s3_url, cfg.output.s3_endpoint, init_store)
        zarr_io.safe_set_attrs(root, {"source_path": source})
        zarr_io.transfer_variables(ds, static_variables, root, tile_j, tile_i,
                                   time_idx=None, skip_existing=skip_existing)
        logging.info("Static grid transfer complete.")

    # --- Time-varying fields -----------------------------------------------
    if time_variables and subset in ("time", "all"):
        if "time" not in ds.dims:
            raise ValueError("Source dataset has no 'time' dimension but time-varying variables were requested.")
        iteration = mit_date_to_iteration(date_str)
        time_idx = mit_date_to_time_idx(date_str, ds.sizes["time"])

        ds_name = _dataset_name_from_date(date_str)
        s3_url = zarr_io._build_s3_url(cfg.output.bucket, cfg.output.folder, ds_name)

        logging.info(
            f"--- Time-varying transfer: {len(time_variables)} variables -> {s3_url} ---\n"
            f"    date={date_str}  iteration={iteration}  time_idx={time_idx}"
        )
        root = zarr_io.open_zarr_store(s3_url, cfg.output.s3_endpoint, init_store)
        zarr_io.safe_set_attrs(root, {
            "source_path": source,
            "selected_iteration": int(iteration),
            "selected_date_utc": date_str,
        })
        zarr_io.transfer_variables(ds, time_variables, root, tile_j, tile_i,
                                   time_idx=time_idx, skip_existing=skip_existing)
        logging.info("Time-varying transfer complete.")

    logging.info("All transfers complete.")
