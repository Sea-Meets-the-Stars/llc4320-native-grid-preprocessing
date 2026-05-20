import s3fs
import xarray as xr
import ujson
import dask
from functools import partial

# Default chunks for the LLC4320 grid
#  Both the data variables and grid need the same chunking
global_chunks={"i": 720, "j": 720, "i_g": 720, "j_g": 720}

# ---------------------------------------------------------------------
# Helper: close all open dataset_creation references
# ---------------------------------------------------------------------
def _multi_file_closer(closers):
    """Invoke all delayed _close() handlers for lazily-opened datasets."""
    for closer in closers:
        closer()

def get_remote_llc_data(endpoint_url, it, face_range):
    """
    Load a single-iteration snapshot of LLC4320 model output from a remote
    kerchunk-backed S3 endpoint. See Accessing_Raw_LLC4320 documentation for details.

    This function opens kerchunk JSON references for a selected set of LLC
    faces and a single model iteration, constructs lazily-evaluated xarray
    datasets via Zarr, and combines them into a single dataset_creation using
    coordinate-based merging. Data access is deferred using Dask until
    explicitly computed.

    Parameters
    ----------
    endpoint_url : str
        URL of the S3-compatible object store hosting the kerchunk JSON files.
    it : int
        LLC model iteration number to load.
    face_range : iterable of int
        Iterable of LLC face indices to load (e.g., ``range(13)`` or a subset).

    Returns
    -------
    xarray.Dataset
        Combined LLC4320 dataset_creation for the requested faces and iteration,
        containing surface-level variables with lazy Dask-backed arrays.

    Notes
    -----
    - Data are accessed anonymously over S3 using kerchunk reference files.
    - Chunking is applied in the horizontal dimensions (``i``, ``j``).
    - A custom close handler is attached to ensure all underlying datasets
      are properly closed when the combined dataset_creation is closed.
    """

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
        # for it in iter_range
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
            chunks=global_chunks,
        )
        for kwargs in backend_kwargs_list
    ]
    closers = [getattr_delayed(ds, "_close") for ds in datasets]

    # Actually open metadata
    print("Computing delayed datasets...")
    datasets, closers = dask.compute(datasets, closers)

    # -----------------------------
    # Combine into a single dataset_creation
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
        Grid fields combined into a single LLC4320 dataset_creation.
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
        open_delayed("reference://", engine="zarr", backend_kwargs=kwargs, 
        chunks=global_chunks)
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


# ---------------------------------------------------------------------------
# S3 shared helpers
# ---------------------------------------------------------------------------

def _build_s3_store_url(bucket: str, folder: str, dataset_name: str) -> str:
    """Build a clean S3 URL from config parts."""
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    return f"s3://{bucket}/{folder}/{dataset_name}"


# ---------------------------------------------------------------------------
# S3 timestep store access - SURFACE ONLY
# ---------------------------------------------------------------------------

# Chunks matching the on-disk layout of stores written by transfer_llc4320.py
s3_timestep_sfc_chunks = {"face": 1, "k": 51, "j": 720, "i": 720}


def _s3_storage_options(s3_endpoint, anon=None):
    """S3 storage options with automatic credential detection.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL.
    anon : bool or None
        Force anonymous access (True) or credentialed access (False).
        If None, uses anonymous for the OSN endpoint and credentialed
        for all others (e.g. NRP Nautilus).
    """
    if anon is None:
        anon = "mghp.osn.xsede.org" in s3_endpoint
    return {
        "anon": anon,
        "client_kwargs": {"endpoint_url": s3_endpoint},
    }


def get_s3_timestep_data(
    s3_endpoint,
    bucket,
    folder,
    date_str,
    face_range=None,
    vars_requested=None,
    chunks=None,
    storage_options=None,
):
    """
    Load a single-timestep snapshot from an S3 timestep store.

    Works for both surface-only and full-depth reads — pass the
    appropriate ``chunks`` and ``storage_options`` for your pipeline.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL.
    bucket : str
        S3 bucket (e.g. ``'dbof/'``).
    folder : str
        S3 folder within the bucket (e.g. ``'LLC4320_v1'``).
    date_str : str
        Date in 'YYYY-MM-DD HH:MM:SS' format.
    face_range : iterable of int or None
        LLC face indices to include.  ``None`` loads all.
    vars_requested : list[str] or None
        Variables to extract.  ``None`` returns all.
    chunks : dict or None
        Dask chunk specification.  Defaults to ``s3_timestep_sfc_chunks``.
        Pass ``s3_timestep_3D_chunks`` for the depth pipeline.
    storage_options : dict or None
        S3 storage options for ``xr.open_zarr``.  Defaults to
        ``_s3_storage_options(s3_endpoint)``.  Pass
        ``_s3_storage_options_3D(s3_endpoint)`` for the depth pipeline.

    Returns
    -------
    xarray.Dataset
    """
    from datetime import datetime as _dt

    if chunks is None:
        chunks = s3_timestep_sfc_chunks
    if storage_options is None:
        storage_options = _s3_storage_options(s3_endpoint)

    date_tag = _dt.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime("%Y%m%dT%H")
    store_name = f"{date_tag}.zarr"
    s3_url = _build_s3_store_url(bucket, folder, store_name)

    ds = xr.open_zarr(
        s3_url,
        consolidated=False,
        chunks=chunks,
        storage_options=storage_options,
    )

    if vars_requested is not None:
        available = [v for v in vars_requested if v in ds]
        ds = ds[available]

    if face_range is not None and 'face' in ds.dims:
        face_list = list(face_range)
        if face_list != list(range(ds.sizes['face'])):
            ds = ds.isel(face=face_list)
            ds = ds.chunk({"face": ds.sizes["face"]})

    print(f"S3 timestep data loaded: {store_name}, vars={list(ds.data_vars)}")
    return ds


# ---------------------------------------------------------------------------
# Timestamp cross-check: OSN vs. S3
# ---------------------------------------------------------------------------

def verify_osn_s3_timestamp(ds_osn, s3_source, date_str, face_range):
    """
    Verify that the OSN and S3 datasets refer to the same physical time.

    OSN time is CF-encoded (``seconds since 2011-09-10``), decoded by xarray
    to ``datetime64``.  MIT/S3 time is stored directly as ``datetime64[ns]``.
    Both should resolve to the same datetime for a given date.

    Parameters
    ----------
    ds_osn : xr.Dataset
        The kerchunk dataset for this snapshot (after ``isel(time=0)``).
    s3_source : dict
        Keys: ``s3_endpoint``, ``bucket``, ``folder``.
    date_str : str
        The date being processed (e.g. ``'2012-11-09 12:00:00'``).
    face_range : range or list
        LLC faces to request.

    Raises
    ------
    RuntimeError
        If the timestamps do not match.
    """
    import logging
    import numpy as np

    try:
        osn_time = np.datetime64(ds_osn['time'].values, 'ns')

        ds_check = get_s3_timestep_data(
            s3_source['s3_endpoint'],
            s3_source['bucket'],
            s3_source['folder'],
            date_str,
            face_range=face_range,
            vars_requested=['time'],
        )

        if 'time' in ds_check:
            mit_time = np.datetime64(ds_check['time'].values.flat[0], 'ns')

            if osn_time == mit_time:
                logging.info(
                    f"TIMESTAMP CHECK PASSED: OSN time={osn_time}, "
                    f"MIT time={mit_time}, date='{date_str}'"
                )
            else:
                logging.error(
                    f"TIMESTAMP MISMATCH: OSN time={osn_time}, "
                    f"MIT time={mit_time} (date='{date_str}')"
                )
                raise RuntimeError(
                    f"Timestep alignment failure for '{date_str}': "
                    f"OSN time={osn_time} != MIT time={mit_time}."
                )
        else:
            logging.warning(
                f"Timestamp cross-check skipped for '{date_str}': "
                f"no 'time' variable in S3 store. Re-run "
                f"transfer_llc4320.py with --variables time to enable."
            )

        ds_check.close()
    except Exception as exc:
        if "alignment" in str(exc).lower():
            raise  # re-raise our own RuntimeError
        logging.warning(
            f"Timestamp cross-check could not be completed for "
            f"'{date_str}': {exc}"
        )



# ---------------------------------------------------------------------------
# S3 timestep store access - DEPTH
# ---------------------------------------------------------------------------
# These functions read from the S3 timestep stores created by
# ``cli.transfer_llc4320.py``.  Layout:
#   {folder}/grid.zarr          — static grid variables
#   {folder}/{YYYYMMDDTHH}.zarr — per-timestep model fields
# ---------------------------------------------------------------------------


# Each S3 GET now retrieves the full water column for one face tile.
s3_timestep_3D_chunks = {"face": 1, "k": 51, "k_l": 51, "k_u": 51, "k_p1": 52, "i": 720, "j": 720, "i_g": 720, "j_g": 720}


# ---------------------------------------------------------------------------
# S3 storage_options for 3D stores
# ---------------------------------------------------------------------------
def _s3_storage_options_3D(s3_endpoint: str) -> dict:
    """Return ``storage_options`` for ``xr.open_zarr`` on an S3-compatible store.

    The Nautilus NRP S3 endpoint intermittently serves corrupt bytes
    (HTTP 200 but garbled body).  We disable s3fs read caching so that
    dask task retries always re-fetch from S3 rather than replaying
    cached bad data.
    """
    return {
        "client_kwargs": {"endpoint_url": s3_endpoint},
        "config_kwargs": {
            "signature_version": "s3v4",
            "retries": {"max_attempts": 5, "mode": "adaptive"},
            "s3": {"addressing_style": "path"},
            # Timeout so hung reads fail fast instead of blocking forever.
            # connect_timeout: max seconds to establish TCP connection.
            # read_timeout: max seconds to wait for data on an open socket.
            "connect_timeout": 30,
            "read_timeout": 60,
        },
        # Cap per-filesystem async connections to limit S3 concurrency.
        "max_concurrency": 10,
        # Disable s3fs read caching so retries always re-fetch fresh bytes.
        "default_cache_type": "none",
    }


def get_s3_gridfile(s3_endpoint: str, bucket: str, folder: str, grid_store_name: str = "grid.zarr"):
    """
    Load LLC4320 grid variables from an S3 grid store written by
    ``transfer_llc4320.py``.

    Returns a Dataset compatible with
    ``preproc_llc_core_data.process_llc4320_grid()``.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL.
    bucket : str
        S3 bucket name.
    folder : str
        S3 folder within the bucket.
    grid_store_name : str
        Name of the grid zarr store (default ``'grid.zarr'``).

    Returns
    -------
    xarray.Dataset
        Grid fields for all 13 faces, with xgcm comodo coordinate attributes.
    """
    s3_url = _build_s3_store_url(bucket, folder, grid_store_name)

    # Grid store chunks: match on-disk layout to avoid rechunk overhead.
    # 2D vars: (face=13, j=720, i=720).
    # 3D vars (hFacC/S/W): on-disk k=1 — keep k=1 so isel(k=0) reads
    # exactly one stored object per spatial tile instead of all 51.
    _grid_chunks = {
        "face": 13, "k": 1, "k_l": 1,
        "j": 720, "i": 720, "j_g": 720, "i_g": 720,
    }
    grid = xr.open_zarr(
        s3_url,
        consolidated=False,
        chunks=_grid_chunks,
        storage_options=_s3_storage_options_3D(s3_endpoint),
    )

    # Select only the grid variables needed for processing.
    # Vertical coordinates (Z, Zl, Zu, Zp1, drF) are required by the
    # depth-diagnostic pipeline for MLD, vertical derivatives, etc.
    grid_vars = ['XC', 'YC', 'dxC', 'dyC', 'dxG', 'dyG', 'rAz', 'rA',
                 'Depth', 'hFacC', 'SN', 'CS',
                 'Z', 'Zl', 'Zu', 'Zp1', 'drF']
    available = [v for v in grid_vars if v in grid]
    grid = grid[available]

    # hFacC may have a k dimension; collapse to surface level.
    if 'hFacC' in grid and 'k' in grid['hFacC'].dims:
        grid = grid.assign(hFacC=grid['hFacC'].isel(k=0, drop=True))

    # Add xgcm comodo coordinate attributes for grid operations.
    coord_meta = {
        'j':   {'axis': 'Y'},
        'j_g': {'axis': 'Y', 'c_grid_axis_shift': 0.5},
        'i':   {'axis': 'X'},
        'i_g': {'axis': 'X', 'c_grid_axis_shift': 0.5},
    }
    coords_update = {}
    for dim, attrs in coord_meta.items():
        if dim in grid.dims:
            existing = (grid.coords[dim] if dim in grid.coords
                        else xr.DataArray(range(grid.sizes[dim]), dims=dim))
            coords_update[dim] = existing.assign_attrs(attrs)
    if coords_update:
        grid = grid.assign_coords(coords_update)

    print(f"S3 grid file loaded from {s3_url}.")
    return grid