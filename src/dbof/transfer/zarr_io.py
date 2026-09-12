"""Zarr / source IO and the per-variable writers shared by both transfer modes.

This module spans the transfer-specific IO machinery:

* **Source read** -- :func:`open_source` opens the local LLC4320 zarr store.
* **Destination** -- :func:`_build_s3_url` / :func:`open_zarr_store` build and
  open the S3 zarr target.
* **Writers** -- per-variable layout inference, the tiled writers that preserve
  the native ``(k, face, j, i)`` LLC4320 layout, read-back verification, and the
  variable dispatcher.

The unified pipeline (:mod:`dbof.transfer.pipeline`) imports from here so there
is a single implementation of the read/write/verify logic for both spatial
extents.

The writers are extent-agnostic: each sizes its target array from the
*dimension sizes of the dataset it is handed* and iterates ``tile_j x tile_i``
tiles.  The ``all`` mode passes the full dataset; the ``chunks`` mode passes a
dataset already sliced to one ``(face=1, j=720, i=720)`` block with
``tile_j = tile_i = 720``, so the same loop writes exactly one chunk.

S3 filesystem construction is reused from
:func:`dbof.io.filesystems.create_s3_filesystems` rather than re-implemented.
Date/iteration conversions live in
:mod:`dbof.llc4320_ingestion.date_iterations`; the all-data store-naming
convention lives in :mod:`dbof.transfer.pipeline`.
"""

# stdlib
import logging

from dbof.llc4320_ingestion.grid import COMODO_COORD_META
import time as _time

# numerical / compute
import numpy as np
import xarray as xr

# Note: ``zarr`` and ``fsspec`` (via dbof.io.filesystems) are imported lazily
# inside open_zarr_store so this module stays importable -- and the layout /
# resolution logic stays unit-testable -- on machines without those heavy
# deps installed.


# ---------------------------------------------------------------------------
# Transfer-specific layout dimensions
# ---------------------------------------------------------------------------
H_J_DIMS = ("j", "j_g")
H_I_DIMS = ("i", "i_g")
VERT_DIMS = ("k", "k_l", "k_u", "k_p1")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _build_s3_url(bucket: str, folder: str, dataset_name: str) -> str:
    """Build a clean S3 URL from config parts, stripping stray slashes.

    ``folder`` may contain nested path segments (e.g.
    ``"LLC_CHUNKS_RAW/monterey_bay"``); leading/trailing slashes are trimmed.
    """
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    if folder:
        return f"s3://{bucket}/{folder}/{dataset_name}"
    return f"s3://{bucket}/{dataset_name}"


def open_zarr_store(s3_url: str, s3_endpoint: str, init_store: bool):
    """Open (or create) a zarr store at *s3_url*.

    Uses the repo-standard async S3 filesystem (:func:`create_s3_filesystems`)
    wrapped in a :class:`zarr.storage.FsspecStore`, matching the global writer.
    """
    import zarr
    from dbof.io.filesystems import create_s3_filesystems

    if not s3_url.startswith("s3://"):
        raise ValueError("Output must be an s3://... path")
    fs, _ = create_s3_filesystems(s3_endpoint)
    store = zarr.storage.FsspecStore(path=s3_url, fs=fs)
    if init_store:
        logging.info(f"Initializing store: {s3_url}")
        return zarr.group(store=store, overwrite=True)
    logging.info(f"Opening existing store: {s3_url}")
    return zarr.open_group(store=store, mode="a", use_consolidated=False)


# ---------------------------------------------------------------------------
# Tile / verification helpers
# ---------------------------------------------------------------------------

def starts(n: int, step: int):
    """Return tile-start indices for a dimension of length *n*."""
    return list(range(0, n, step))


def _verify_tile(z_var, idx, expected, label, max_retries=3):
    """Read back a tile from S3 and compare to the source data.

    If the read-back doesn't match, re-write and re-verify up to
    *max_retries* times.  Raises on persistent mismatch.
    """
    for attempt in range(1, max_retries + 1):
        try:
            readback = z_var[idx]
        except Exception as exc:
            logging.warning(f"  VERIFY {label}: read-back failed ({exc})")
            if attempt < max_retries:
                _time.sleep(2 ** attempt)
                logging.info(f"  VERIFY {label}: re-writing (attempt {attempt}/{max_retries})")
                z_var[idx] = expected
                continue
            raise

        # Compare: treat NaN == NaN as equal.
        if np.issubdtype(expected.dtype, np.floating):
            match = np.allclose(readback, expected, equal_nan=True, rtol=0, atol=0)
        else:
            match = np.array_equal(readback, expected)

        if match:
            return  # success

        logging.warning(
            f"  VERIFY {label}: mismatch on read-back "
            f"(attempt {attempt}/{max_retries})"
        )
        if attempt < max_retries:
            _time.sleep(2 ** attempt)
            logging.info(f"  VERIFY {label}: re-writing...")
            z_var[idx] = expected
        else:
            raise RuntimeError(
                f"VERIFY FAILED after {max_retries} attempts: {label}"
            )


def safe_set_attrs(zobj, attrs: dict):
    """Set zarr attributes, silently skipping any that fail."""
    for k, v in attrs.items():
        try:
            zobj.attrs[k] = v
        except Exception:
            pass


def ensure_coord_written(root, ds, coord_name, chunk_len=None):
    """Write a coordinate array into the zarr root if it doesn't exist."""
    if coord_name in root:
        return
    vals = ds[coord_name].values
    if chunk_len is None:
        chunk_len = vals.shape[0]
    zc = root.create_array(
        coord_name,
        shape=vals.shape,
        chunks=(min(vals.shape[0], max(1, chunk_len)),),
        dtype=vals.dtype,
        overwrite=True,
        dimension_names=(coord_name,),
    )
    zc[:] = vals

    attrs = dict(ds[coord_name].attrs) if coord_name in ds.coords else {}
    if 'axis' not in attrs and coord_name in COMODO_COORD_META:
        attrs.update(COMODO_COORD_META[coord_name])
    if attrs:
        safe_set_attrs(zc, attrs)


# ---------------------------------------------------------------------------
# Layout inference
# ---------------------------------------------------------------------------

def infer_layout(da: xr.DataArray):
    """
    Classify a DataArray as 1D-time-only, 1D-vertical, 2D-horizontal, or
    3D-horizontal based on its dimension names.  Drives which writer function
    is used to transfer the variable to S3.
    """
    dims = da.dims
    has_time = "time" in dims
    has_face = "face" in dims

    jdims = [d for d in H_J_DIMS if d in dims]
    idims = [d for d in H_I_DIMS if d in dims]
    vdims = [d for d in VERT_DIMS if d in dims]

    if len(vdims) > 1:
        raise ValueError(f"{da.name} dims {dims}: multiple vertical dims found")
    vdim = vdims[0] if vdims else None

    # 1D time-only (e.g. the "time" coordinate variable itself)
    if dims == ("time",):
        return {"kind": "1d_time", "has_time": True}

    # 1D vertical profile
    if vdim is not None and not has_face:
        expected = (("time",) if has_time else ()) + (vdim,)
        if dims != expected:
            raise ValueError(f"{da.name} dims are {dims}, expected {expected}")
        return {"kind": "1d_vertical", "has_time": has_time, "vdim": vdim}

    # 2D horizontal
    if vdim is None and has_face:
        if len(jdims) != 1 or len(idims) != 1:
            raise ValueError(f"{da.name} dims {dims}: expected one j-like and one i-like dim")
        jdim = jdims[0]
        idim = idims[0]
        expected = (("time",) if has_time else ()) + ("face", jdim, idim)
        if dims != expected:
            raise ValueError(f"{da.name} dims are {dims}, expected {expected}")
        return {
            "kind": "2d_horizontal",
            "has_time": has_time,
            "jdim": jdim,
            "idim": idim,
        }

    # 3D horizontal
    if vdim is not None and has_face:
        if len(jdims) != 1 or len(idims) != 1:
            raise ValueError(f"{da.name} dims {dims}: expected one j-like and one i-like dim")
        jdim = jdims[0]
        idim = idims[0]
        expected = (("time",) if has_time else ()) + (vdim, "face", jdim, idim)
        if dims != expected:
            raise ValueError(f"{da.name} dims are {dims}, expected {expected}")
        return {
            "kind": "3d_horizontal",
            "has_time": has_time,
            "vdim": vdim,
            "jdim": jdim,
            "idim": idim,
        }

    raise ValueError(f"Unsupported dims for {da.name}: {dims}")


# ---------------------------------------------------------------------------
# Per-layout writers
# ---------------------------------------------------------------------------

def write_2d_horizontal(root, ds, da, time_idx, tile_j, tile_i):
    """Write a 2D (face, j, i) variable into the zarr store tile-by-tile."""
    info = infer_layout(da)
    jdim = info["jdim"]
    idim = info["idim"]
    has_time = info["has_time"]

    ensure_coord_written(root, ds, "face", ds.sizes["face"])
    ensure_coord_written(root, ds, jdim, tile_j)
    ensure_coord_written(root, ds, idim, tile_i)

    nface = ds.sizes["face"]
    nj = ds.sizes[jdim]
    ni = ds.sizes[idim]

    logging.info(f"Creating target array for {da.name} with dims (face, {jdim}, {idim})")
    z_var = root.create_array(
        da.name,
        shape=(nface, nj, ni),
        chunks=(nface, tile_j, tile_i),
        dtype=da.dtype,
        overwrite=True,
        fill_value=np.nan if np.issubdtype(da.dtype, np.floating) else 0,
        dimension_names=("face", jdim, idim),
    )
    safe_set_attrs(z_var, da.attrs)

    j_starts = starts(nj, tile_j)
    i_starts = starts(ni, tile_i)
    total_tiles = len(j_starts) * len(i_starts)
    tile_count = 0

    for j0 in j_starts:
        j1 = min(j0 + tile_j, nj)
        for i0 in i_starts:
            i1 = min(i0 + tile_i, ni)
            tile_count += 1

            logging.info(
                f"[{da.name} {tile_count}/{total_tiles}] "
                f"reading face=:, {jdim}={j0}:{j1}, {idim}={i0}:{i1}"
            )
            indexer = {"face": slice(None), jdim: slice(j0, j1), idim: slice(i0, i1)}
            if has_time:
                indexer["time"] = time_idx
            tile = da.isel(**indexer).values

            logging.info(
                f"[{da.name} {tile_count}/{total_tiles}] "
                f"writing face=:, {jdim}={j0}:{j1}, {idim}={i0}:{i1}"
            )
            z_var[:, j0:j1, i0:i1] = tile

            _verify_tile(
                z_var, (slice(None), slice(j0, j1), slice(i0, i1)),
                tile, f"{da.name} tile {tile_count}/{total_tiles}",
            )


def write_3d_horizontal(root, ds, da, time_idx, tile_j, tile_i, level_chunked=False):
    """Write a 3D (vdim, face, j, i) variable into the zarr store tile-by-tile.

    Two chunk layouts are supported:

    * ``level_chunked=False`` (default) -- ``(nk, 1, tile_j, tile_i)``: all depth
      levels in one chunk, one face per chunk.  This matches the MIT LLC4320
      on-disk layout and is optimal for time-varying depth fields, where a
      single S3 GET returns the full water column for one face tile.
    * ``level_chunked=True`` -- ``(1, nface, tile_j, tile_i)``: one depth level,
      all faces, per chunk.  This is the layout used by the static grid store
      (``hFacC``/``hFacS``/``hFacW`` etc.) that the global grid reader expects
      (``get_llc_depth_gridfile``), so that ``isel(k=0)`` reads exactly one
      stored object per spatial tile instead of the whole column.

    The on-disk *shape* and ``dimension_names`` are identical in both cases;
    only the chunk grid and the write order differ.
    """
    info = infer_layout(da)
    vdim = info["vdim"]
    jdim = info["jdim"]
    idim = info["idim"]
    has_time = info["has_time"]

    ensure_coord_written(root, ds, vdim)
    ensure_coord_written(root, ds, "face", ds.sizes["face"])
    ensure_coord_written(root, ds, jdim, tile_j)
    ensure_coord_written(root, ds, idim, tile_i)

    nk = ds.sizes[vdim]
    nface = ds.sizes["face"]
    nj = ds.sizes[jdim]
    ni = ds.sizes[idim]

    chunks = (1, nface, tile_j, tile_i) if level_chunked else (nk, 1, tile_j, tile_i)
    logging.info(
        f"Creating target array for {da.name} with dims ({vdim}, face, {jdim}, {idim}), "
        f"chunks={chunks}"
    )
    z_var = root.create_array(
        da.name,
        shape=(nk, nface, nj, ni),
        chunks=chunks,
        dtype=da.dtype,
        overwrite=True,
        fill_value=np.nan if np.issubdtype(da.dtype, np.floating) else 0,
        dimension_names=(vdim, "face", jdim, idim),
    )
    safe_set_attrs(z_var, da.attrs)

    j_starts = starts(nj, tile_j)
    i_starts = starts(ni, tile_i)

    if level_chunked:
        # One stored object per (level, spatial tile), spanning all faces.
        total_tiles = nk * len(j_starts) * len(i_starts)
        tile_count = 0
        for kk in range(nk):
            for j0 in j_starts:
                j1 = min(j0 + tile_j, nj)
                for i0 in i_starts:
                    i1 = min(i0 + tile_i, ni)
                    tile_count += 1

                    logging.info(
                        f"[{da.name} {tile_count}/{total_tiles}] "
                        f"reading {vdim}={kk}, face=:, {jdim}={j0}:{j1}, {idim}={i0}:{i1}"
                    )
                    indexer = {vdim: kk, "face": slice(None),
                               jdim: slice(j0, j1), idim: slice(i0, i1)}
                    if has_time:
                        indexer["time"] = time_idx
                    tile = da.isel(**indexer).values

                    logging.info(
                        f"[{da.name} {tile_count}/{total_tiles}] "
                        f"writing {vdim}={kk}, face=:, {jdim}={j0}:{j1}, {idim}={i0}:{i1}"
                    )
                    z_var[kk, :, j0:j1, i0:i1] = tile

                    _verify_tile(
                        z_var, (kk, slice(None), slice(j0, j1), slice(i0, i1)),
                        tile, f"{da.name} tile {tile_count}/{total_tiles}",
                    )
        return

    # Default: one stored object per (face, spatial tile), full water column.
    total_tiles = nface * len(j_starts) * len(i_starts)
    tile_count = 0
    for ff in range(nface):
        for j0 in j_starts:
            j1 = min(j0 + tile_j, nj)
            for i0 in i_starts:
                i1 = min(i0 + tile_i, ni)
                tile_count += 1

                logging.info(
                    f"[{da.name} {tile_count}/{total_tiles}] "
                    f"reading {vdim}=:, face={ff}, {jdim}={j0}:{j1}, {idim}={i0}:{i1}"
                )
                indexer = {vdim: slice(None), "face": ff, jdim: slice(j0, j1), idim: slice(i0, i1)}
                if has_time:
                    indexer["time"] = time_idx
                tile = da.isel(**indexer).values

                logging.info(
                    f"[{da.name} {tile_count}/{total_tiles}] "
                    f"writing {vdim}=:, face={ff}, {jdim}={j0}:{j1}, {idim}={i0}:{i1}"
                )
                z_var[:, ff, j0:j1, i0:i1] = tile

                _verify_tile(
                    z_var, (slice(None), ff, slice(j0, j1), slice(i0, i1)),
                    tile, f"{da.name} tile {tile_count}/{total_tiles}",
                )


def write_1d_vertical(root, ds, da, time_idx):
    """Write a 1D vertical-profile variable into the zarr store."""
    info = infer_layout(da)
    vdim = info["vdim"]
    has_time = info["has_time"]

    ensure_coord_written(root, ds, vdim)

    nk = ds.sizes[vdim]

    logging.info(f"Writing vertical profile {da.name} with dim {vdim}")
    z_var = root.create_array(
        da.name,
        shape=(nk,),
        chunks=(nk,),
        dtype=da.dtype,
        overwrite=True,
        fill_value=np.nan if np.issubdtype(da.dtype, np.floating) else 0,
        dimension_names=(vdim,),
    )
    safe_set_attrs(z_var, da.attrs)

    indexer = {vdim: slice(None)}
    if has_time:
        indexer["time"] = time_idx
    vals = da.isel(**indexer).values
    z_var[:] = vals

    _verify_tile(z_var, (slice(None),), vals, f"{da.name} (1D)")


def write_1d_time(root, ds, da, time_idx):
    """Write the scalar time value for a single timestep into the zarr store."""
    val = da.isel(time=time_idx).values
    out = np.atleast_1d(val)

    logging.info(f"Writing time value for {da.name}: time_idx={time_idx} -> value={val}")
    z_var = root.create_array(
        da.name,
        shape=out.shape,
        chunks=out.shape,
        dtype=out.dtype,
        overwrite=True,
        fill_value=0,
        dimension_names=("time",),
    )
    safe_set_attrs(z_var, da.attrs)
    z_var[:] = out

    _verify_tile(z_var, (slice(None),), out, f"{da.name} (1D time)")


# ---------------------------------------------------------------------------
# Variable dispatcher
# ---------------------------------------------------------------------------

def transfer_variables(ds, variables, root, tile_j, tile_i,
                       time_idx=None, skip_existing=False, level_chunked_3d=False):
    """Dispatch each variable to the appropriate per-layout writer.

    ``level_chunked_3d`` controls the chunk layout of 3D variables -- see
    :func:`write_3d_horizontal`.  Use ``True`` for the static grid store so its
    3D fields (hFacC/S/W, masks) are stored one level / all faces per chunk.
    """
    for var in variables:
        if skip_existing and var in root:
            logging.info(f"Skipping {var} (already exists in store)")
            continue
        da = ds[var]
        info = infer_layout(da)
        kind = info["kind"]

        logging.info(f"=== {var}: {kind}, dims={da.dims} ===")

        if kind == "1d_time":
            write_1d_time(root, ds, da, time_idx)
        elif kind == "2d_horizontal":
            write_2d_horizontal(root, ds, da, time_idx, tile_j, tile_i)
        elif kind == "3d_horizontal":
            write_3d_horizontal(root, ds, da, time_idx, tile_j, tile_i,
                                level_chunked=level_chunked_3d)
        elif kind == "1d_vertical":
            write_1d_vertical(root, ds, da, time_idx)
        else:
            raise ValueError(f"Unhandled layout kind '{kind}' for variable {var}")


def write_store(ds, variables, *, s3_url, s3_endpoint, tile_j, tile_i,
                init_store=False, time_idx=None, attrs=None,
                skip_existing=False, label="transfer", detail="",
                level_chunked_3d=False):
    """Write one group of variables to a single S3 zarr store.

    This bundles the open-store -> stamp-attrs -> write-variables -> log
    sequence that every transfer performs.  Both modes call it: the all-data
    mode for its static grid store and each per-date store, the chunks mode for
    its grid store and each per-timestamp store.  Mode-specific concerns
    (which dataset, destination URL, tile size, attrs, label) are passed in.

    Parameters
    ----------
    ds : xarray.Dataset
        Source to read variables from (full dataset, or a chunk-sliced one).
    variables : list of str
        Variable names to write.
    s3_url : str
        Fully built ``s3://...`` destination for this store.
    s3_endpoint : str
        S3 endpoint URL.
    tile_j, tile_i : int
        Spatial tile sizes for zarr chunking.
    init_store : bool
        If ``True``, wipe/re-initialise the store before writing.
    time_idx : int or None
        Time index for time-varying variables; ``None`` for static variables.
    attrs : dict, optional
        Root attributes to stamp on the store.
    skip_existing : bool
        Skip variables already present in the target store.
    label : str
        Human-readable label for log messages (e.g. ``"Static grid transfer"``).
    detail : str
        Optional extra detail appended to the header log line (e.g. the date /
        iteration / time index for a time-varying store).
    level_chunked_3d : bool
        Chunk 3D variables one level / all faces per chunk (see
        :func:`write_3d_horizontal`).  Set ``True`` for the static grid store.
    """
    header = f"--- {label}: {len(variables)} variables -> {s3_url} ---"
    if detail:
        header += f"\n    {detail}"
    logging.info(header)

    root = open_zarr_store(s3_url, s3_endpoint, init_store)
    if attrs:
        safe_set_attrs(root, attrs)
    transfer_variables(ds, variables, root, tile_j, tile_i,
                       time_idx=time_idx, skip_existing=skip_existing,
                       level_chunked_3d=level_chunked_3d)
    logging.info(f"{label} complete.")


def validate_variables_present(ds, variables) -> None:
    """Raise if any requested variable is missing from the source dataset.

    Checks both ``data_vars`` and ``coords`` (dimension coordinates such as
    ``i``, ``j``, ``face``, ``k`` live in ``ds.coords``).
    """
    for var in variables:
        if var not in ds and var not in ds.coords:
            raise ValueError(f"Variable '{var}' not found in source store")


def open_source(source: str) -> xr.Dataset:
    """Open the local/cluster LLC4320 zarr source store."""
    if not source:
        raise ValueError(
            "data.MIT_data_path must be set in the config to the path of the "
            "local LLC4320 zarr store."
        )
    logging.info(f"Opening source store: {source}")
    return xr.open_zarr(source, consolidated=False)
