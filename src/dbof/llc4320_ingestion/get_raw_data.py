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
# S3 timestep store access - SURFACE ONLY
# ---------------------------------------------------------------------------

# Chunks matching the on-disk layout of stores written by transfer_llc4320.py
s3_timestep_chunks = {"face": 1, "k": 51, "j": 720, "i": 720}


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
):
    """
    Load a single-timestep snapshot from an S3 timestep store.

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

    Returns
    -------
    xarray.Dataset
    """
    from datetime import datetime as _dt

    date_tag = _dt.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime("%Y%m%dT%H")
    store_name = f"{date_tag}.zarr"
    s3_url = f"s3://{bucket}{folder}/{store_name}"

    ds = xr.open_zarr(
        s3_url,
        consolidated=False,
        chunks=s3_timestep_chunks,
        storage_options=_s3_storage_options(s3_endpoint),
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