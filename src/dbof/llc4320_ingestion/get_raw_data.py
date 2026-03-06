import s3fs
import xarray as xr
import ujson
import dask
from functools import partial


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
            chunks={"i": 720, "j": 720}
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
        chunks={'i': 720, 'j': 720})
        #chunks={})
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