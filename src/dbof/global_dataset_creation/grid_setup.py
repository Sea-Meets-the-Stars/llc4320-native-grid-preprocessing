"""
Grid setup helpers for the unified global pipeline.

Two loading paths depending on the pipeline:

OSN / SURF
    Grid is fetched from the OSN kerchunk endpoint using
    ``get_remote_gridfile`` and processed with the 2D grid function.

DEPTH
    Grid is fetched from an S3 grid store using ``get_s3_gridfile``
    and processed with the 3D grid function.  The grid is eagerly
    computed into memory (~1.5 GB) and vertical coordinates are
    dropped (xgcm only needs the horizontal grid).

Both paths return the same triple ``(ds_grid, land_mask, grid)``
so the caller can treat them interchangeably.
"""

import logging

import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.llc4320_ingestion.get_raw_data as get_raw_data
from dbof.llc4320_ingestion.grid import set_xgcm_grid


# Vertical coordinates that xgcm doesn't need (DEPTH pipeline only).
_VERTICAL_VARS = {"Z", "Zl", "Zu", "Zp1", "drF"}


# ---------------------------------------------------------------------------
# OSN / SURF grid loading
# ---------------------------------------------------------------------------

def set_up_grid_osn(endpoint_url: str):
    """
    Load the LLC4320 grid and land mask from the OSN kerchunk endpoint.

    Parameters
    ----------
    endpoint_url : str
        OSN endpoint (e.g. ``'https://mghp.osn.xsede.org'``).

    Returns
    -------
    ds_grid : xr.Dataset
        Processed LLC4320 grid dataset.
    land_mask : xr.DataArray
        ``hFacC`` field (0 = land, >0 = ocean).
    grid : xgcm.Grid
        xgcm Grid with LLC face connections.
    """
    logging.info("Fetching grid file from OSN kerchunk endpoint")
    co = get_raw_data.get_remote_gridfile(endpoint_url)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    land_mask = ds_grid.hFacC
    grid = set_xgcm_grid(ds_grid, use_connections=True)

    logging.info("Grid and land mask loaded (OSN)")
    return ds_grid, land_mask, grid


# ---------------------------------------------------------------------------
# DEPTH (S3) grid loading
# ---------------------------------------------------------------------------

def set_up_grid_s3(s3_source: dict):
    """
    Load the LLC4320 3D grid and land mask from an S3 grid store.

    The grid is eagerly computed into memory and vertical coordinates
    are dropped so xgcm only operates on the horizontal stencil.

    Parameters
    ----------
    s3_source : dict
        Must contain ``s3_endpoint``, ``bucket``, ``folder``.
        Optionally ``grid_folder`` (defaults to ``folder``).

    Returns
    -------
    ds_grid : xr.Dataset
        Processed LLC4320 3D grid dataset (eagerly loaded).
    land_mask : xr.DataArray
        ``hFacC`` field (0 = land, >0 = ocean).
    grid : xgcm.Grid
        xgcm Grid with LLC face connections.
    """
    grid_folder = s3_source.get("grid_folder", s3_source["folder"])
    logging.info(f"Fetching grid file from S3 grid store (folder={grid_folder})")
    co = get_raw_data.get_s3_gridfile(
        s3_source["s3_endpoint"],
        s3_source["bucket"],
        grid_folder,
    )
    ds_grid = preproc_llc_core_data.process_llc4320_3d_grid(co)

    # Eagerly load into memory -- the grid is small (~1.5 GB).
    logging.info("Eagerly loading grid into memory...")
    ds_grid = ds_grid.compute()
    logging.info("Grid loaded into memory.")

    land_mask = ds_grid.hFacC

    # Drop vertical coords -- xgcm only needs the horizontal grid.
    drop = [v for v in _VERTICAL_VARS if v in ds_grid]
    grid_for_xgcm = ds_grid.drop_vars(drop) if drop else ds_grid
    grid = set_xgcm_grid(grid_for_xgcm, use_connections=True)

    logging.info("Grid and land mask loaded (S3)")
    return ds_grid, land_mask, grid


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def set_up_grid(pipeline: str, data_source: dict | None, endpoint_url: str | None = None):
    """
    Load grid, land mask, and xgcm Grid for the given pipeline.

    Parameters
    ----------
    pipeline : str
        ``"SURF"``, ``"OSN"``, or ``"DEPTH"``.
    data_source : dict or None
        S3 data-source dict (from ``data_sources.get_data_source()``).
        Required for DEPTH; ignored for SURF/OSN.
    endpoint_url : str or None
        OSN endpoint URL.  Required for SURF/OSN; ignored for DEPTH.

    Returns
    -------
    ds_grid, land_mask, grid
    """
    if pipeline in ("SURF", "OSN"):
        if endpoint_url is None:
            from dbof.global_dataset_creation.data_sources import OSN_ENDPOINT
            endpoint_url = OSN_ENDPOINT
        return set_up_grid_osn(endpoint_url)
    elif pipeline == "DEPTH":
        if data_source is None:
            raise ValueError("DEPTH pipeline requires a data_source dict.")
        return set_up_grid_s3(data_source)
    else:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'.  Expected SURF, OSN, or DEPTH."
        )
