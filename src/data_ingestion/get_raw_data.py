"""
Utilities for loading LLC4320 model output and grid files
from a remote kerchunk-based S3 endpoint.

Functions
---------
get_remote_llc_data(endpoint_url)
    Load raw LLC4320 variables for a selected set of faces/timesteps.

get_remote_gridfile(endpoint_url)
    Load grid metadata (XC, YC, metrics, CS/SN, etc.) for all 13 LLC faces.
"""

import s3fs
import xarray as xr
import ujson
import dask
import numpy as np
from functools import partial


# ---------------------------------------------------------------------
# Helper: close all open dataset references
# ---------------------------------------------------------------------
def _multi_file_closer(closers):
    """Invoke all delayed _close() handlers for lazily-opened datasets."""
    for closer in closers:
        closer()


# ---------------------------------------------------------------------
# Load model variables
# ---------------------------------------------------------------------
## todo add in temporal inputs
def get_remote_llc_data(endpoint_url):
    """
    Load LLC4320 raw variables for a subset of timesteps and faces
    from a kerchunk-referenced S3 endpoint.

    Parameters
    ----------
    endpoint_url : str
        S3-compatible endpoint (e.g., OSN / Open Storage Network).

    Returns
    -------
    xarray.Dataset
        Combined dataset containing Theta, U, V, W, Salt, Eta for all
        selected times and all 13 LLC4320 faces.
    """
    # -----------------------------
    # Configuration parameters
    # -----------------------------
    varnames = ["Theta", "U", "V", "W", "Salt", "Eta"]
    timestep_hours = 12                     # how many hours to load
    sampling_step = 12                      # stride in timesteps
    ts_per_hour = 144                       # model cadence: 25 s → 144 steps/hr
    iter_step = sampling_step * ts_per_hour # iteration Δ between samples
    face_range = range(13)

    # First valid wind/forcing record begins ~1180
    start_record = 1180

    # Compute iter numbers
    start_iter = 10368 + start_record * ts_per_hour
    end_iter = start_iter + timestep_hours * ts_per_hour
    iter_range = np.arange(start_iter, end_iter, iter_step)

    # Include SSH
    get_eta_files = True

    # -----------------------------
    # Build file list
    # -----------------------------
    if get_eta_files:
        pattern = "cnh-bucket-1/llc_surf/kerchunk_files/llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{it}.json"
    else:
        pattern = ("cnh-bucket-1/llc_wind/kerchunk_files/"
                   "llc4320_KPPhbl-PhiBot-oceTAUX-oceTAUY-SIarea_f{face}_k0_iter_{it}.json")

    filelist = [
        pattern.format(face=face, it=it)
        for face in face_range
        for it in iter_range
    ]

    print(f"Opening {len(filelist)} Kerchunk JSON files...")

    # S3 filesystem
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": endpoint_url}
    )

    # Open JSON files
    mapper = [fs.open(f, mode="rb") for f in filelist]

    print("Parsing JSON metadata into Python dicts...")
    reflist = [ujson.load(m) for m in mapper]

    # -----------------------------
    # Build lazy xarray openers
    # -----------------------------
    open_delayed = dask.delayed(xr.open_dataset)
    getattr_delayed = dask.delayed(getattr)

    backend_kwargs_list = [
        {
            "storage_options": {
                "fo": ref,
                "asynchronous": True,
                "remote_protocol": "s3",
                "remote_options": {
                    "client_kwargs": {"endpoint_url": endpoint_url},
                    "asynchronous": True,
                    "anon": True
                }
            },
            "consolidated": False
        }
        for ref in reflist
    ]

    print("Creating lazy xarray datasets...")
    datasets = [
        open_delayed(
            "reference://",
            engine="zarr",
            backend_kwargs=kwargs,
            chunks={"i": 720, "j": 720}
        )
        for kwargs in backend_kwargs_list
    ]
    closers = [getattr_delayed(ds, "_close") for ds in datasets]

    # Actually open metadata
    print("Computing delayed datasets...")
    datasets, closers = dask.compute(datasets, closers)

    # -----------------------------
    # Combine into a single dataset
    # -----------------------------
    print("Combining datasets by coordinates...")
    ds = xr.combine_by_coords(
        datasets,
        compat="override",
        coords="minimal",
        combine_attrs="override"
    )

    # Close each underlying file handle
    for ds_local in datasets:
        ds_local.close()

    # Register custom closer
    ds.set_close(partial(_multi_file_closer, closers))

    print("Dataset combined successfully.")

    # Select the first time and depth level (as in original code)
    ds = ds.isel(time=0, k=0, k_l=0)
    return ds


# ---------------------------------------------------------------------
# Load grid files (XC, YC, metrics, CS, SN, etc.)
# ---------------------------------------------------------------------
def get_remote_gridfile(endpoint_url):
    """
    Load the LLC4320 grid variables for all 13 faces
    using kerchunk pointers stored in S3.

    Parameters
    ----------
    endpoint_url : str
        S3-compatible endpoint.

    Returns
    -------
    xarray.Dataset
        Grid fields combined into a single LLC4320 dataset.
    """
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": endpoint_url}
    )

    filelist = [
        f"cnh-bucket-1/llc_surf/kerchunk_files/llc4320_grid_f{face}.json"
        for face in range(13)
    ]

    mapper = [fs.open(f, mode="rb") for f in filelist]
    reflist = [ujson.load(m) for m in mapper]

    open_delayed = dask.delayed(xr.open_dataset)
    getattr_delayed = dask.delayed(getattr)

    backend_kwargs_list = [
        {
            "storage_options": {
                "fo": ref,
                "asynchronous": True,
                "remote_protocol": "s3",
                "remote_options": {
                    "client_kwargs": {"endpoint_url": endpoint_url},
                    "asynchronous": True,
                    "anon": True
                },
            },
            "consolidated": False
        }
        for ref in reflist
    ]

    datasets = [
        open_delayed("reference://", engine="zarr", backend_kwargs=kwargs, chunks={})
        for kwargs in backend_kwargs_list
    ]
    closers = [getattr_delayed(ds, "_close") for ds in datasets]

    datasets, closers = dask.compute(datasets, closers)

    # Combine all faces
    grid = xr.combine_by_coords(
        datasets,
        compat="override",
        coords="minimal",
        combine_attrs="override"
    )

    # Close individual datasets
    for ds_local in datasets:
        ds_local.close()

    return grid