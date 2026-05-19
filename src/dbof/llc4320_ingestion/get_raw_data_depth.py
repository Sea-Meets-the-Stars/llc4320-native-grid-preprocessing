import s3fs
import xarray as xr
import ujson
import dask
from functools import partial


# ---------------------------------------------------------------------------
# S3 timestep store access (derived-products pipeline)
# ---------------------------------------------------------------------------
# These functions read from the S3 timestep stores created by
# ``cli.transfer_llc4320.py``.  Layout:
#   {folder}/grid.zarr          — static grid variables
#   {folder}/{YYYYMMDDTHH}.zarr — per-timestep model fields
# ---------------------------------------------------------------------------

# After re-transfer with MIT-matching layout: k=all, face=1, j=720, i=720.
# Each S3 GET now retrieves the full water column for one face tile.
s3_timestep_chunks = {"face": 1, "k": 51, "k_l": 51, "k_u": 51, "k_p1": 52, "i": 720, "j": 720, "i_g": 720, "j_g": 720}


# ---------------------------------------------------------------------------
# S3 storage_options shared by all S3 zarr reads — configures botocore-level
# retries so that transient read corruption (truncated bytes → Zstd errors)
# is retried at the HTTP layer before zarr/dask ever sees the failure.
# ---------------------------------------------------------------------------
def _s3_storage_options(s3_endpoint: str) -> dict:
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


def _build_s3_store_url(bucket: str, folder: str, dataset_name: str) -> str:
    """Build a clean S3 URL from config parts."""
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    return f"s3://{bucket}/{folder}/{dataset_name}"


def get_s3_timestep_data(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    date_str: str,
    face_range=None,
    vars_requested=None,
):
    """
    Load a single-timestep snapshot from an S3 timestep store written by
    ``transfer_llc4320.py``.

    The store contains all variables for one date in native LLC4320 layout
    (k, face, j, i) for 3D fields, (face, j, i) for 2D fields.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL (e.g. ``'https://s3-west.nrp-nautilus.io'``).
    bucket : str
        S3 bucket name (e.g. ``'dbof/'``).
    folder : str
        S3 folder within the bucket (e.g. ``'LLC4320'``).
    date_str : str
        Date string in 'YYYY-MM-DD HH:MM:SS' format.  Converted to the
        store name ``YYYYMMDDTHH.zarr``.
    face_range : iterable of int or None
        LLC face indices to include (e.g. ``range(13)``).  ``None`` loads all.
    vars_requested : list[str] or None
        Variables to extract.  Names absent from the store are silently
        skipped.  If ``None``, all data variables are returned.

    Returns
    -------
    xarray.Dataset
        Single-timestep dataset with lazy Dask-backed arrays retaining all
        k dimensions.  Compatible with downstream xgcm grid operations.
    """
    from datetime import datetime as _dt
    date_tag = _dt.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime("%Y%m%dT%H")
    store_name = f"{date_tag}.zarr"
    s3_url = _build_s3_store_url(bucket, folder, store_name)

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
        # Only subset + rechunk if face_range is a strict subset of the
        # stored face dimension.  When all faces are requested (the common
        # case), the isel + rechunk is a no-op that creates redundant dask
        # graph layers with duplicate key aliases, triggering the
        # "Detected different run_spec for key" warning in the distributed
        # scheduler and causing downstream decompression failures.
        if face_list != list(range(ds.sizes['face'])):
            ds = ds.isel(face=face_list)
            ds = ds.chunk({"face": ds.sizes["face"]})

    print(f"S3 timestep data loaded: {store_name}, vars={list(ds.data_vars)}")
    return ds


def _is_decompression_error(exc):
    """Return True if *exc* looks like a Zstd / blosc / codec failure."""
    exc_str = str(exc).lower()
    return any(
        kw in exc_str
        for kw in ("zstd", "blosc", "decompress", "corrupt", "lz4")
    )


def _load_chunk_with_retry(da, sel, label, max_retries, reopen_fn):
    """
    Load a single spatial chunk of a DataArray with per-chunk retry.

    On decompression failure the zarr store is re-opened (so s3fs drops
    any cached bad bytes) and only the failing chunk is re-fetched.

    Parameters
    ----------
    da : xarray.DataArray
        Lazy (dask-backed) array already isel'd to the target face.
    sel : dict
        Indexing dict for this chunk, e.g. ``{'j': slice(0,720), 'i': slice(0,720)}``.
    label : str
        Human-readable label for log messages.
    max_retries : int
    reopen_fn : callable() -> xarray.DataArray
        Called on retry to get a fresh lazy DataArray (new s3fs connection).

    Returns
    -------
    numpy.ndarray
    """
    import time as _time

    current_da = da
    for attempt in range(1, max_retries + 1):
        try:
            return current_da.isel(sel).values
        except Exception as exc:
            if _is_decompression_error(exc) and attempt < max_retries:
                wait = min(2 ** attempt, 30)
                print(
                    f"  {label}: decompression error (attempt {attempt}/{max_retries}), "
                    f"retrying in {wait}s — {exc}"
                )
                _time.sleep(wait)
                # Re-open the store so s3fs doesn't replay cached bad bytes.
                current_da = reopen_fn()
                continue
            raise


def get_s3_timestep_data_single_face(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    date_str: str,
    face_idx: int,
    vars_requested=None,
    max_retries: int = 10,
):
    """
    Load a single face of a single-timestep snapshot from S3, one stored
    chunk at a time with per-chunk retry on decompression errors.

    On-disk chunk layout (zarr v3, after re-transfer)::

        Theta: shape [51, 13, 4320, 4320]
               chunk [51,  1, 720,  720]

    Each stored object holds **all k-levels × one face × 720×720 spatial
    tile** (~100 MB compressed).  Selecting a single face now reads only
    that face's data — no wasted download of other faces.

    The retry loop iterates over ``(j_block, i_block)`` — each iteration is
    exactly one S3 GET request (the full water column for one spatial tile).
    If a request returns corrupt bytes (intermittent Nautilus NRP issue),
    the store is re-opened (fresh s3fs connection) and only that one object
    is re-fetched.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL.
    bucket, folder : str
        S3 bucket and folder.
    date_str : str
        Date string in ``'YYYY-MM-DD HH:MM:SS'`` format.
    face_idx : int
        Single face index (0–12).
    vars_requested : list[str] or None
        Variables to extract.  ``None`` loads all data variables.
    max_retries : int
        Per-chunk retry limit (default 10).

    Returns
    -------
    xarray.Dataset
        Single-face, single-timestep dataset with all k-levels, eagerly
        loaded into memory (numpy-backed).  The ``face`` dimension is
        retained with size 1.
    """
    import numpy as _np
    from datetime import datetime as _dt

    date_tag = _dt.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime("%Y%m%dT%H")
    store_name = f"{date_tag}.zarr"
    s3_url = _build_s3_store_url(bucket, folder, store_name)
    sopts = _s3_storage_options(s3_endpoint)

    # -- helper: open a fresh lazy single-face dataset ----------------------
    # Chunks match the post-re-transfer on-disk layout (k=all, face=1) so
    # that each dask task corresponds to exactly one S3 GET request.
    _on_disk_chunks = {
        "k": 51, "k_l": 51, "k_u": 51, "k_p1": 52,
        "face": 1,
        "j": 720, "i": 720, "j_g": 720, "i_g": 720,
    }

    def _open_lazy():
        ds = xr.open_zarr(
            s3_url, consolidated=False,
            chunks=_on_disk_chunks, storage_options=sopts,
        )
        if vars_requested is not None:
            available = [v for v in vars_requested if v in ds]
            ds = ds[available]
        if 'face' in ds.dims:
            ds = ds.isel(face=[face_idx])
        return ds

    ds_lazy = _open_lazy()

    # -- identify chunk grid for each variable ------------------------------
    loaded_vars = {}

    for vname in list(ds_lazy.data_vars):
        da = ds_lazy[vname]

        # Determine which dims are chunked spatially.
        has_j = ('j' in da.dims or 'j_g' in da.dims)
        has_i = ('i' in da.dims or 'i_g' in da.dims)

        if not (has_j and has_i):
            # 1D coordinate / small variable — load in one shot.
            try:
                loaded_vars[vname] = da.compute()
            except Exception:
                loaded_vars[vname] = da
            continue

        # Detect the vertical dimension (if any).
        k_dim = None
        for candidate in ("k", "k_l", "k_u", "k_p1"):
            if candidate in da.dims:
                k_dim = candidate
                break

        j_dim = 'j_g' if 'j_g' in da.dims else 'j'
        i_dim = 'i_g' if 'i_g' in da.dims else 'i'

        # Build edge arrays for spatial chunks.
        j_chunks_tuple = da.chunks[da.dims.index(j_dim)]
        i_chunks_tuple = da.chunks[da.dims.index(i_dim)]
        j_edges = _np.concatenate([[0], _np.cumsum(j_chunks_tuple)])
        i_edges = _np.concatenate([[0], _np.cumsum(i_chunks_tuple)])
        n_j = len(j_chunks_tuple)
        n_i = len(i_chunks_tuple)

        # Build edge arrays for vertical chunks.
        if k_dim is not None:
            k_chunks_tuple = da.chunks[da.dims.index(k_dim)]
            k_edges = _np.concatenate([[0], _np.cumsum(k_chunks_tuple)])
            n_k = len(k_chunks_tuple)
        else:
            k_edges = None
            n_k = 1

        total_chunks = n_k * n_j * n_i
        print(f"  {vname}: {total_chunks} on-disk chunks "
              f"(k={n_k}, j={n_j}, i={n_i}), loading with retry...")

        # -- Load every (k, j, i) chunk with retry -------------------------
        # Accumulate into a pre-allocated numpy array to avoid holding
        # thousands of small arrays.
        full_shape = tuple(da.sizes[d] for d in da.dims)
        result = _np.empty(full_shape, dtype=_np.float32)
        result[:] = _np.nan

        chunks_ok = 0
        chunks_retried = 0

        for kk in range(n_k):
            for jj in range(n_j):
                for ii in range(n_i):
                    sel = {
                        j_dim: slice(int(j_edges[jj]), int(j_edges[jj + 1])),
                        i_dim: slice(int(i_edges[ii]), int(i_edges[ii + 1])),
                    }
                    if k_dim is not None:
                        sel[k_dim] = slice(int(k_edges[kk]), int(k_edges[kk + 1]))

                    # Build the numpy indexing tuple (same dim order as da.dims).
                    np_idx = []
                    for d in da.dims:
                        if d in sel:
                            np_idx.append(sel[d])
                        else:
                            np_idx.append(slice(None))
                    np_idx = tuple(np_idx)

                    label = (f"face {face_idx}/{vname} "
                             f"k={kk} j={jj} i={ii}")

                    def _reopen(vn=vname):
                        fresh = _open_lazy()
                        return fresh[vn]

                    arr = _load_chunk_with_retry(
                        da, sel, label, max_retries, _reopen,
                    )
                    result[np_idx] = arr
                    chunks_ok += 1

        if chunks_retried > 0:
            print(f"  {vname}: done ({chunks_retried} chunks needed retries)")

        loaded_vars[vname] = xr.DataArray(
            result, dims=da.dims, coords=da.coords, attrs=da.attrs,
        )

    ds_result = xr.Dataset(loaded_vars, coords=ds_lazy.coords, attrs=ds_lazy.attrs)
    print(f"S3 face {face_idx} loaded: {store_name} (all chunks OK)")
    return ds_result


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
        storage_options=_s3_storage_options(s3_endpoint),
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


