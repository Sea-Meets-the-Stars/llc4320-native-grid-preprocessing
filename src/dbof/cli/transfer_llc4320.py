#!/usr/bin/env python3
"""
Transfer LLC4320 variables from a local zarr store to S3 Zarr with a
unified tiled layout.

#NOTE: this script is designed to be run from the MIT machines.

Two categories of variables are transferred to separate stores:

1. **Static grid variables** (geometry, masks, vertical coordinates) —
   written once to ``{folder}/grid.zarr``.
2. **Time-varying fields** (Theta, Salt, …) — written per-date to
   ``{folder}/{YYYYMMDDTHH}.zarr``.

CLI usage
---------
    transfer-timestep --config configs/transfer/run.yaml --init-store
    transfer-timestep --config configs/transfer/run.yaml --subset static --init-store
    transfer-timestep --config configs/transfer/run.yaml --subset time --variable Theta

    # Override date from CLI:
    transfer-timestep --config configs/transfer/run.yaml \\
        --date "2012-11-09 12:00:00" --init-store

Config design
-------------
The YAML reuses the standard ``data``, ``output``, and ``runtime`` sections
from JobConfig, plus a transfer-specific top-level ``transfer:`` key:

    transfer:
      static_variables: [XC, YC, ...]
      static_dataset_name: grid.zarr
      variables: [Theta, Salt, ...]
      tile_j: 720
      tile_i: 720
"""

# stdlib
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# numerical / compute
import numpy as np
import xarray as xr
import zarr
import fsspec
import yaml

# internal
import dbof.dataset_creation.config as config


# ---------------------------------------------------------------------------
# LLC4320 model constants and dimensions
# ---------------------------------------------------------------------------
TS_PER_HOUR = 144
LLC4320_START_DATE = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS = 25
DATE_FMT = "%Y-%m-%d %H:%M:%S"

H_J_DIMS = ("j", "j_g")
H_I_DIMS = ("i", "i_g")
VERT_DIMS = ("k", "k_l", "k_u", "k_p1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_logging(cfg: config.JobConfig) -> None:
    """Configure file + stdout logging for a pipeline run."""
    log_root = Path(cfg.run.log_dir).expanduser().resolve()
    run_dir = log_root / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "transfer_llc4320.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def date_to_iteration(date_str: str) -> int:
    """Convert an ISO date string to an LLC4320 iteration number."""
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(f"Date {date_str} is before LLC4320 start date.")
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def _dataset_name_from_date(date_str: str) -> str:
    """Convert a date string to a dataset name.

    '2012-11-09 12:00:00' → '20121109T12.zarr'
    """
    dt = datetime.strptime(date_str, DATE_FMT)
    return dt.strftime("%Y%m%dT%H") + ".zarr"


def _build_s3_url(bucket: str, folder: str, dataset_name: str) -> str:
    """Build a clean S3 URL from config parts, stripping stray slashes."""
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    return f"s3://{bucket}/{folder}/{dataset_name}"


def build_s3_mapper(s3_url: str, endpoint: str):
    """Create an fsspec S3 mapper for the given URL."""
    if not s3_url.startswith("s3://"):
        raise ValueError("Output must be an s3://... path")
    return fsspec.get_mapper(
        s3_url,
        client_kwargs={"endpoint_url": endpoint},
        config_kwargs={
            "signature_version": "s3v4",
            "s3": {"addressing_style": "path"},
        },
    )


def starts(n: int, step: int):
    """Return tile-start indices for a dimension of length *n*. Used to iterate over spatial tiles."""
    return list(range(0, n, step))


def _verify_tile(z_var, idx, expected, label, max_retries=3):
    """Read back a tile from S3 and compare to the source data.

    If the read-back doesn't match, re-write and re-verify up to
    *max_retries* times.  Raises on persistent mismatch.

    Parameters
    ----------
    z_var : zarr.Array
        Target zarr array (already written to S3).
    idx : tuple of slices
        Indexing tuple to select the tile.
    expected : numpy.ndarray
        Source data that was written.
    label : str
        Human-readable label for log messages.
    max_retries : int
        Number of re-write + re-verify attempts.
    """
    import time as _time

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


# ---------------------------------------------------------------------------
# Layout inference
# ---------------------------------------------------------------------------

def infer_layout(da: xr.DataArray):
    """
    Classify a DataArray as 1D-time-only, 1D-vertical, 2D-horizontal, or
    3D-horizontal based on its dimension names.
    Drives which writer function is used to transfer the variable to S3.
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
        return {
            "kind": "1d_time",
            "has_time": True,
        }

    # 1D vertical profile
    if vdim is not None and not has_face:
        expected = (("time",) if has_time else ()) + (vdim,)
        if dims != expected:
            raise ValueError(f"{da.name} dims are {dims}, expected {expected}")
        return {
            "kind": "1d_vertical",
            "has_time": has_time,
            "vdim": vdim,
        }

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


def write_3d_horizontal(root, ds, da, time_idx, tile_j, tile_i):
    """Write a 3D (vdim, face, j, i) variable into the zarr store tile-by-tile.

    Chunk layout: ``(nk, 1, tile_j, tile_i)`` — all depth levels in one
    chunk, one face per chunk, spatial tiles of (tile_j × tile_i).

    This matches the MIT LLC4320 on-disk layout and is optimal for
    depth-diagnostic pipelines that need the full water column per face.
    The write loop iterates **face-by-face**, so each stored object
    contains the complete vertical profile for one spatial tile of one
    face.
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

    logging.info(
        f"Creating target array for {da.name} with dims ({vdim}, face, {jdim}, {idim}), "
        f"chunks=({nk}, 1, {tile_j}, {tile_i})"
    )
    z_var = root.create_array(
        da.name,
        shape=(nk, nface, nj, ni),
        chunks=(nk, 1, tile_j, tile_i),
        dtype=da.dtype,
        overwrite=True,
        fill_value=np.nan if np.issubdtype(da.dtype, np.floating) else 0,
        dimension_names=(vdim, "face", jdim, idim),
    )
    safe_set_attrs(z_var, da.attrs)

    j_starts = starts(nj, tile_j)
    i_starts = starts(ni, tile_i)
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
    """Write the scalar time value for a single timestep into the zarr store.

    The source ``da`` has shape ``(ntime,)`` over the full time dimension.
    We select ``time_idx`` and store the resulting scalar as a 1-element
    array so downstream readers can retrieve the MIT-epoch time index for
    this snapshot.
    """
    val = da.isel(time=time_idx).values
    # Store as a scalar (0-d) or 1-element array — 1-element is safer for
    # zarr readers that expect at least one dimension.
    out = np.atleast_1d(val)

    logging.info(f"Writing time value for {da.name}: time_idx={time_idx} → value={val}")
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
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse CLI arguments.

    Only ``--config`` is required.  ``--date`` overrides the config default;
    ``--init-store`` is intentionally CLI-only (destructive; could overwrite existing data).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Transfer LLC4320 variables from a local zarr store to S3 Zarr. "
            "Static grid variables are written once; time-varying fields are "
            "written per-date."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (e.g. configs/transfer/run.yaml).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Override date for time-varying variables (ISO format, e.g. "
            "'2012-11-09 12:00:00').  Overrides the first entry in "
            "data.date_iterations."
        ),
    )
    parser.add_argument(
        "--init-store",
        action="store_true",
        help=(
            "Initialize/reset output stores before writing.  "
            "Use only on the first run — this wipes existing data."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=["static", "time", "all"],
        default="all",
        help=(
            "Which variable group to transfer: 'static' (grid.zarr only), "
            "'time' (time-varying only), or 'all' (default)."
        ),
    )
    parser.add_argument(
        "--variables",
        default=None,
        help=(
            "Comma-separated list of variable names to transfer, overriding "
            "the config.  Applied to whichever group --subset selects.  "
            "E.g. '--subset time --variables Theta' transfers only Theta."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip variables that already exist in the target zarr store.  "
            "Useful for adding newly configured variables without "
            "re-writing ones already transferred."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_transfer_config(config_file: str):
    """
    Load the YAML config and return ``(cfg, transfer_opts)``.

    ``cfg`` is a fully resolved JobConfig with all standard sections 
    (run, data, sampling, output, features, runtime).  
    
    ``transfer_opts`` is a dict of transfer-specific settings 
    (variables, static_variables, tile_j, tile_i, etc.) extracted from the top-level "transfer" section of the YAML.
    """
    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    transfer_opts = raw.get("transfer", {})
    if not transfer_opts:
        raise ValueError(
            f"Missing 'transfer' section in {config_file}.  "
            "Expected keys: variables, static_variables, tile_j, tile_i."
        )

    cfg = config.JobConfig(
        run=config.RunConfig(**raw.get("run", {})),
        data=config.DataConfig(**raw.get("data", {})),
        sampling=config.SamplingConfig(**raw.get("sampling", {})),
        output=config.OutputConfig(**raw.get("output", {})),
        features=config.FeaturesConfig(**raw.get("features", {})),
        runtime=config.RuntimeConfig(**raw.get("runtime", {})),
    )

    return cfg, transfer_opts


# ---------------------------------------------------------------------------
# Transfer pipeline
# ---------------------------------------------------------------------------

def _open_zarr_store(s3_url: str, s3_endpoint: str, init_store: bool):
    """Open (or create) a zarr store at *s3_url*."""
    store = build_s3_mapper(s3_url, s3_endpoint)
    if init_store:
        logging.info(f"Initializing store: {s3_url}")
        return zarr.group(store=store, overwrite=True)
    logging.info(f"Opening existing store: {s3_url}")
    return zarr.open_group(store=store, mode="a")


def _transfer_variables(ds, variables, root, tile_j, tile_i,
                        time_idx=None, skip_existing=False):
    """Dispatch each variable to the appropriate per-layout writer."""
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
            write_3d_horizontal(root, ds, da, time_idx, tile_j, tile_i)
        elif kind == "1d_vertical":
            write_1d_vertical(root, ds, da, time_idx)
        else:
            raise ValueError(f"Unhandled layout kind '{kind}' for variable {var}")


def run_transfer(
    cfg: config.JobConfig,
    transfer_opts: dict,
    date_str: str,
    init_store: bool = False,
    subset: str = "all",
    skip_existing: bool = False,
    variables_override: list = None,
) -> None:
    """
    Execute the transfer: open source store, write static grid variables
    to one store, write time-varying fields to a date-named store.

    Parameters
    ----------
    cfg : config.JobConfig
        Fully resolved config.
    transfer_opts : dict
        Transfer-specific settings (variables, static_variables, tiles, …).
    date_str : str
        Date for time-varying fields (ISO format).
    init_store : bool
        If ``True``, wipe and re-initialise output stores.
    subset : str
        ``"static"`` to transfer only grid variables, ``"time"`` for only
        time-varying fields, or ``"all"`` (default) for both.
    skip_existing : bool
        If ``True``, skip variables that already exist in the target store.
    variables_override : list of str, optional
        If provided, overrides the variable lists from the config.  Applied
        to whichever group(s) ``subset`` selects.
    """
    source = cfg.data.MIT_data_path
    if not source:
        raise ValueError(
            "data.MIT_data_path must be set in the config to the "
            "path of the local LLC4320 zarr store."
        )

    static_variables = transfer_opts.get("static_variables", [])
    time_variables = transfer_opts.get("variables", [])

    # --variables CLI override: filter each list to only the requested names.
    if variables_override is not None:
        override_set = set(variables_override)
        if subset in ("static", "all"):
            filtered_static = [v for v in static_variables if v in override_set]
            # Also include any override names not in either list (user may
            # want to transfer an arbitrary variable from the source store).
            extra = [v for v in variables_override
                     if v not in filtered_static and v not in time_variables]
            static_variables = filtered_static + extra if subset == "static" else filtered_static
        if subset in ("time", "all"):
            filtered_time = [v for v in time_variables if v in override_set]
            extra = [v for v in variables_override
                     if v not in filtered_time and v not in static_variables]
            time_variables = filtered_time + extra if subset == "time" else filtered_time
        logging.info(f"--variables override active: static={static_variables}, time={time_variables}")
    tile_j = transfer_opts.get("tile_j", 720)
    tile_i = transfer_opts.get("tile_i", 720)

    if not static_variables and not time_variables:
        raise ValueError("No variables to transfer (both static and time-varying lists are empty).")

    # --- Open source store ---------------------------------------------------
    logging.info(f"Opening source store: {source}")
    ds = xr.open_zarr(source, consolidated=False)

    # Validate all requested variables exist in the source
    # (check both data_vars and coords — dimension coordinates like
    #  i, j, face, k, etc. live in ds.coords, not ds.data_vars)
    for var in static_variables + time_variables:
        if var not in ds and var not in ds.coords:
            raise ValueError(f"Variable '{var}' not found in source store")

    # --- Static grid variables -----------------------------------------------
    if static_variables and subset in ("static", "all"):
        static_name = transfer_opts.get("static_dataset_name", "grid.zarr")
        s3_url = _build_s3_url(cfg.output.bucket, cfg.output.folder, static_name)

        logging.info(f"--- Static grid transfer: {len(static_variables)} variables → {s3_url} ---")
        root = _open_zarr_store(s3_url, cfg.output.s3_endpoint, init_store)
        safe_set_attrs(root, {"source_path": source})

        _transfer_variables(ds, static_variables, root, tile_j, tile_i,
                            time_idx=None, skip_existing=skip_existing)
        logging.info("Static grid transfer complete.")

    # --- Time-varying fields -------------------------------------------------
    if time_variables and subset in ("time", "all"):
        iteration = date_to_iteration(date_str)
        time_idx = iteration // TS_PER_HOUR

        # Validate time dimension
        if "time" not in ds.dims:
            raise ValueError("Source dataset has no 'time' dimension but time-varying variables were requested.")
        ntime = ds.sizes["time"]
        if not (0 <= time_idx < ntime):
            raise ValueError(f"time_idx={time_idx} out of range [0, {ntime})")

        ds_name = _dataset_name_from_date(date_str)
        s3_url = _build_s3_url(cfg.output.bucket, cfg.output.folder, ds_name)

        logging.info(
            f"--- Time-varying transfer: {len(time_variables)} variables → {s3_url} ---\n"
            f"    date={date_str}  iteration={iteration}  time_idx={time_idx}"
        )
        root = _open_zarr_store(s3_url, cfg.output.s3_endpoint, init_store)
        safe_set_attrs(root, {
            "source_path": source,
            "selected_iteration": int(iteration),
            "selected_date_utc": date_str,
        })

        _transfer_variables(ds, time_variables, root, tile_j, tile_i,
                            time_idx=time_idx, skip_existing=skip_existing)
        logging.info("Time-varying transfer complete.")

    logging.info("All transfers complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_file: str = None, date: str = None, init_store: bool = False,
         subset: str = "all", skip_existing: bool = False,
         variables_override: list = None) -> None:
    """
    Entry point for the LLC4320 transfer script.

    Can be called from the CLI (no arguments — reads from ``sys.argv``) or
    directly from Python by passing arguments explicitly.

    Parameters
    ----------
    config_file : str, optional
        Path to the YAML config.  If ``None``, ``--config`` is read from
        ``sys.argv``.
    date : str, optional
        Override date string (ISO format).  If ``None``, falls back to
        ``data.date_iterations[0]`` in the config.
    init_store : bool
        If ``True``, wipe output stores before writing.
    subset : str
        ``"static"``, ``"time"``, or ``"all"`` (default).
    skip_existing : bool
        If ``True``, skip variables already present in the target store.
    variables_override : list of str, optional
        Override which variables to transfer (applied to the selected subset).
    """
    # --- Resolve CLI arguments -----------------------------------------------
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        date = date or cli.date
        init_store = init_store or cli.init_store
        subset = cli.subset
        skip_existing = cli.skip_existing
        if cli.variables is not None:
            variables_override = [v.strip() for v in cli.variables.split(",")]

    # --- Load config ---------------------------------------------------------
    cfg, transfer_opts = _load_transfer_config(config_file)

    # --- Resolve date --------------------------------------------------------
    if date is None:
        if cfg.data.date_iterations:
            date = cfg.data.date_iterations[0]
    if date is None and transfer_opts.get("variables") and subset in ("time", "all"):
        raise ValueError(
            "No date provided.  Set data.date_iterations in the config "
            "or pass --date on the CLI."
        )

    # --- Logging -------------------------------------------------------------
    generate_logging(cfg)
    logging.info(f"Config loaded from: {config_file}")

    # --- Run -----------------------------------------------------------------
    run_transfer(
        cfg=cfg,
        transfer_opts=transfer_opts,
        date_str=date,
        init_store=init_store,
        subset=subset,
        skip_existing=skip_existing,
        variables_override=variables_override,
    )


if __name__ == "__main__":
    main()
