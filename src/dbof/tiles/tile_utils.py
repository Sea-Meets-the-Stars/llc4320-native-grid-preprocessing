"""
Shared building blocks for generating a 3D LLC4320 tile of a single property.

This module holds everything that ``generate_tile.py`` orchestrates:

* timing constants and small helpers (date <-> iteration, git commit lookup),
* S3 config loader,
* default output-path builder (per-property filename prefix),
* tile-restricted grid and tracer loaders,
* the property registry (``TILE_PROPERTIES``) plus per-property compute
  callbacks (potential density, potential temperature, salinity),
* the shared compute scaffolding (``compute_tile_property``),
* output ``xr.Dataset`` assembly and the QA-plot writer,
* the end-to-end orchestrator ``run(...)``.

Adding a new property is a one-entry edit in ``TILE_PROPERTIES`` -- supply the
input tracer variables, a compute callback that returns a lazy ``xr.DataArray``
on dims ``(face, k, j, i)``, plus output metadata (variable name, units,
long_name, filename prefix).
"""

# stdlib
from __future__ import annotations
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Allow ``import tile_mapping`` whether this file is imported as a package
# module or via a sys.path insertion done by a caller.  Harmless either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# numerical / IO
import numpy as np
import xarray as xr
import yaml

# plotting -- use Agg so QA-plot writing works on headless nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dask
from dask.diagnostics import ProgressBar

# repo modules (installed as the ``dbof`` package).
import dbof.llc4320_ingestion.get_raw_data as get_raw_data
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.utils.physical_calculations as physical_calculations

# local -- tile_mapping lives next to this file.
from tile_mapping import rect_ij_to_tile, TileInfo  # noqa: E402


# ---------------------------------------------------------------------------
# LLC4320 timing constants -- mirrors generate_global_depth_dask.py
# ---------------------------------------------------------------------------
LLC4320_START_DATE    = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS = 25
DATE_FMT              = "%Y-%m-%d %H:%M:%S"

# configs/global_depth.yaml at the repo root -- 3 levels up from this file.
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "global_depth.yaml"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _date_to_iteration(date_str: str) -> int:
    """Convert ``'YYYY-MM-DD HH:MM:SS'`` to an LLC4320 iteration number.

    Parameters
    ----------
    date_str : str
        Timestamp string in ``DATE_FMT`` (UTC).

    Returns
    -------
    int
        Number of LLC4320 timesteps (25 s each) since the model start
        2011-09-13 00:00:00 UTC.

    Raises
    ------
    ValueError
        If the date precedes the LLC4320 start date.
    """
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start "
            f"({LLC4320_START_DATE.date()})."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def _git_commit() -> str:
    """Return the current git commit hash for provenance attrs.

    Returns
    -------
    str
        Full commit hash, or the literal ``'unknown'`` if ``git rev-parse``
        fails for any reason (not in a repo, no git on PATH, etc.).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, check=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _load_s3_config(config_path: Path) -> dict:
    """Load the ``s3_source`` block from a ``global_depth.yaml``-style config.

    Parameters
    ----------
    config_path : Path
        Path to the YAML file with an ``s3_source`` top-level key.

    Returns
    -------
    dict
        Dict with at least ``s3_endpoint``, ``bucket``, ``folder``, and
        ``grid_folder``.  ``grid_folder`` falls back to ``folder`` if absent
        (matches the behaviour in ``generate_global_depth_dask.py``).

    Raises
    ------
    ValueError
        If ``s3_source`` is missing or empty.
    """
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    s3 = raw.get("s3_source")
    if not s3:
        raise ValueError(f"Missing 's3_source' section in {config_path}")
    s3.setdefault("grid_folder", s3["folder"])
    return s3


def _build_output_path(
    user_output: str | None,
    tile_idx: int,
    date_str: str,
    filename_prefix: str,
) -> Path:
    """Resolve the NetCDF output path with per-property defaults.

    Parameters
    ----------
    user_output : str or None
        User-supplied output path.  Three accepted forms:
          * ``None`` -- use the default name in CWD.
          * existing directory -- place the default name inside it.
          * full file path -- used verbatim.
    tile_idx : int
        Flat rect-grid tile index, 0..431.
    date_str : str
        Timestamp string in ``DATE_FMT``; used to build ``YYYYMMDDTHH``.
    filename_prefix : str
        Property-specific prefix (e.g. ``'density'``, ``'theta'``, ``'salt'``).

    Returns
    -------
    Path
        Resolved absolute output path.  Default form is
        ``./{filename_prefix}_tile{tile_idx:03d}_{YYYYMMDDTHH}.nc``.
    """
    dt = datetime.strptime(date_str, DATE_FMT)
    stamp = dt.strftime("%Y%m%dT%H")
    default_name = f"{filename_prefix}_tile{tile_idx:03d}_{stamp}.nc"

    if user_output is None:
        return Path(default_name).resolve()
    p = Path(user_output)
    if p.is_dir():
        return (p / default_name).resolve()
    return p.resolve()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_grid_for_tile(s3_cfg: dict, tile: TileInfo) -> xr.Dataset:
    """Fetch ``grid.zarr`` from S3 and reduce it to the tile extent.

    Parameters
    ----------
    s3_cfg : dict
        S3 source config (see :func:`_load_s3_config`).
    tile : TileInfo
        Resolved tile metadata.

    Returns
    -------
    xr.Dataset
        Eagerly loaded grid for the tile, containing at least ``XC``, ``YC``,
        ``Z``, ``hFacC``, with dims ``(face=1, k=51, j=720, i=720)`` (subset).
    """
    logging.info(
        f"Loading grid (face={tile.face_idx}, "
        f"j={tile.j_face_slice}, i={tile.i_face_slice})"
    )
    co = get_raw_data.get_s3_gridfile(
        s3_cfg["s3_endpoint"],
        s3_cfg["bucket"],
        s3_cfg["grid_folder"],
    )
    ds_grid = preproc_llc_core_data.process_llc4320_3d_grid(co)
    ds_grid_tile = ds_grid.isel(
        face=[tile.face_idx],
        j=tile.j_face_slice,
        i=tile.i_face_slice,
    ).compute()
    return ds_grid_tile


def _load_tracers_for_tile(
    s3_cfg: dict,
    date_str: str,
    tile: TileInfo,
    vars_needed: list[str],
) -> xr.Dataset:
    """Open the timestep zarr lazily and slice down to the tile.

    Parameters
    ----------
    s3_cfg : dict
        S3 source config (see :func:`_load_s3_config`).
    date_str : str
        Timestamp string in ``DATE_FMT``.
    tile : TileInfo
        Resolved tile metadata.
    vars_needed : list of str
        Tracer variable names to request from the timestep store
        (e.g. ``['Theta', 'Salt']``).

    Returns
    -------
    xr.Dataset
        Dask-backed Dataset with the requested variables on dims
        ``(face=1, k=51, j=720, i=720)``.  Nothing is read from S3 until
        ``.compute()``.
    """
    logging.info(
        f"Opening timestep store for {date_str} (face={tile.face_idx}, "
        f"vars={vars_needed})"
    )
    ds = get_raw_data.get_s3_timestep_data(
        s3_cfg["s3_endpoint"],
        s3_cfg["bucket"],
        s3_cfg["folder"],
        date_str,
        face_range=[tile.face_idx],
        vars_requested=vars_needed,
    )
    ds_tile = ds.isel(j=tile.j_face_slice, i=tile.i_face_slice)
    return ds_tile


# ---------------------------------------------------------------------------
# Property registry + per-property compute callbacks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileProperty:
    """Specification for a single property that can be extracted as a tile.

    Attributes
    ----------
    name : str
        Public name used at the CLI (``'density'``, ``'temperature'``, ...).
    vars_needed : tuple of str
        S3 tracer variables required to compute the field.
    out_name : str
        Variable name written to the output NetCDF.
    units : str
        ``units`` attribute for the output variable.
    long_name : str
        ``long_name`` attribute for the output variable.
    filename_prefix : str
        Prefix used in the default output filename.
    compute : Callable
        Callback ``(ds_tracers_tile) -> xr.DataArray``.  Must return a lazy
        (or eager) DataArray on dims ``(face, k, j, i)``.  Materialization
        and dtype handling are done by :func:`compute_tile_property`.
    """
    name:             str
    vars_needed:      tuple[str, ...]
    out_name:         str
    units:            str
    long_name:        str
    filename_prefix:  str
    compute:          Callable[[xr.Dataset], xr.DataArray]


def _compute_sigma0(ds_tracers_tile: xr.Dataset) -> xr.DataArray:
    """Potential density anomaly referenced to the surface.

    Uses ``dbof.utils.physical_calculations.density_of_field``, which calls
    JMD95 with ``p=0`` and an ``apply_ufunc(..., dask='parallelized')`` path,
    then subtracts 1000.

    Parameters
    ----------
    ds_tracers_tile : xr.Dataset
        Tile dataset with ``Theta`` (deg C) and ``Salt`` (PSU).

    Returns
    -------
    xr.DataArray
        ``sigma0`` (kg m^-3) on the same dims as inputs (still dask-backed
        if the inputs were).
    """
    rho = physical_calculations.density_of_field(ds_tracers_tile)
    return rho - 1000.0


def _compute_theta(ds_tracers_tile: xr.Dataset) -> xr.DataArray:
    """Potential temperature passthrough.

    Parameters
    ----------
    ds_tracers_tile : xr.Dataset
        Tile dataset with ``Theta``.

    Returns
    -------
    xr.DataArray
        ``Theta`` (deg C) -- the same array, returned for uniform handling
        by :func:`compute_tile_property`.
    """
    return ds_tracers_tile["Theta"]


def _compute_salt(ds_tracers_tile: xr.Dataset) -> xr.DataArray:
    """Salinity passthrough.

    Parameters
    ----------
    ds_tracers_tile : xr.Dataset
        Tile dataset with ``Salt``.

    Returns
    -------
    xr.DataArray
        ``Salt`` (PSU) -- the same array, returned for uniform handling
        by :func:`compute_tile_property`.
    """
    return ds_tracers_tile["Salt"]


# Registry: add new properties by appending an entry here.  No other code
# changes are required for properties that fit the (face, k, j, i) shape.
TILE_PROPERTIES: dict[str, TileProperty] = {
    "density": TileProperty(
        name="density",
        vars_needed=("Theta", "Salt"),
        out_name="sigma0",
        units="kg m-3",
        long_name="potential density anomaly referenced to surface (JMD95, p=0)",
        filename_prefix="density",
        compute=_compute_sigma0,
    ),
    "temperature": TileProperty(
        name="temperature",
        vars_needed=("Theta",),
        out_name="Theta",
        units="degC",
        long_name="potential temperature",
        filename_prefix="theta",
        compute=_compute_theta,
    ),
    "salinity": TileProperty(
        name="salinity",
        vars_needed=("Salt",),
        out_name="Salt",
        units="psu",
        long_name="salinity",
        filename_prefix="salt",
        compute=_compute_salt,
    ),
}


# ---------------------------------------------------------------------------
# Shared compute scaffolding
# ---------------------------------------------------------------------------

def compute_tile_property(
    ds_tracers_tile: xr.Dataset,
    prop: TileProperty,
) -> xr.DataArray:
    """Run a property's compute callback, materialise the tile, and finalise dtype/attrs.

    No land masking is applied: the returned array carries whatever the
    compute callback produced over land cells (typically a numerical value
    derived from whatever Theta/Salt the model stored there).

    Parameters
    ----------
    ds_tracers_tile : xr.Dataset
        Lazy tracer Dataset sliced to the tile, dims ``(face=1, k=51, j=720, i=720)``.
    prop : TileProperty
        The property spec; its ``compute`` callback is invoked.

    Returns
    -------
    xr.DataArray
        The materialised property field, float32, dims ``(k, j, i)`` -- the
        size-1 ``face`` dim is squeezed.  ``name``, ``units`` and ``long_name``
        attrs are set from ``prop``.
    """
    # Build the lazy graph via the property's compute callback.
    field = prop.compute(ds_tracers_tile)

    # Single .compute() materialises the entire lazy graph (matches the
    # one-compute pattern used by generate_global_depth_dask.py).
    logging.info(
        f"Computing {prop.name} for tile (single .compute() over the lazy graph)"
    )
    with ProgressBar():
        field = field.compute()

    # Drop the size-1 face dim -- single-face tile output.
    if "face" in field.dims:
        field = field.isel(face=0)

    # Float32 keeps the saved file compact; precision loss is well below the
    # natural variability of the fields we care about.
    field = field.astype(np.float32)
    field.name = prop.out_name
    field.attrs["units"] = prop.units
    field.attrs["long_name"] = prop.long_name
    return field


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _build_output_dataset(
    field: xr.DataArray,
    ds_grid_tile: xr.Dataset,
    tile: TileInfo,
    prop: TileProperty,
    date_str: str,
    iteration: int,
    rect_i_user: int,
    rect_j_user: int,
) -> xr.Dataset:
    """Wrap the computed field + coords + provenance attrs into a final xr.Dataset.

    Parameters
    ----------
    field : xr.DataArray
        Output of :func:`compute_tile_property`, dims ``(k, j, i)``.
    ds_grid_tile : xr.Dataset
        Tile-extent grid (XC, YC, Z) used to populate output coordinates.
    tile : TileInfo
        Resolved tile metadata.
    prop : TileProperty
        Property spec; its ``out_name`` becomes the dataset's data variable.
    date_str : str
        Timestamp string in ``DATE_FMT``.
    iteration : int
        LLC4320 iteration number.
    rect_i_user, rect_j_user : int
        Original rect-grid pixel the user supplied (recorded for provenance).

    Returns
    -------
    xr.Dataset
        Dataset with the property as its sole data variable, 2D ``XC``/``YC``
        and 1D ``Z`` coords, and provenance attrs.
    """
    # Drop the size-1 face dim from the grid so its coords align with field.
    grid = ds_grid_tile.isel(face=0)
    XC = grid["XC"].astype(np.float64)
    YC = grid["YC"].astype(np.float64)
    Z  = grid["Z"].astype(np.float64)

    ds_out = xr.Dataset(
        data_vars={prop.out_name: field},
        coords={
            "XC": XC,   # 2D lon
            "YC": YC,   # 2D lat
            "Z":  Z,    # 1D depth
            # Scalar provenance coords.
            "tile_index":   tile.tile_idx,
            "face_index":   tile.face_idx,
            "rect_i_start": tile.rect_i_slice.start,
            "rect_j_start": tile.rect_j_slice.start,
        },
        attrs={
            "timestamp":     date_str,
            "iteration":     iteration,
            "tile_index":    tile.tile_idx,
            "tile_j_rect":   tile.tile_j_rect,
            "tile_i_rect":   tile.tile_i_rect,
            "face_index":    tile.face_idx,
            "rect_i_user":   rect_i_user,
            "rect_j_user":   rect_j_user,
            "property":      prop.name,
            "source_script": "dev/pot_density/generate_tile.py",
            "git_commit":    _git_commit(),
        },
    )
    return ds_out


# ---------------------------------------------------------------------------
# QA plot
# ---------------------------------------------------------------------------

def _qa_plot(
    ds_out: xr.Dataset,
    prop: TileProperty,
    png_path: Path,
) -> None:
    """Save a single surface pcolormesh of the property next to the NetCDF.

    Parameters
    ----------
    ds_out : xr.Dataset
        The output dataset produced by :func:`_build_output_dataset`.
    prop : TileProperty
        Used to look up the variable name and units for the plot.
    png_path : Path
        Where to write the PNG (typically ``out_path.with_suffix('.png')``).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    surf = ds_out[prop.out_name].isel(k=0)
    pcm = ax.pcolormesh(
        ds_out["XC"].values,
        ds_out["YC"].values,
        surf.values,
        shading="auto",
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        f"{prop.out_name} (k=0)  tile={int(ds_out['tile_index'])}  "
        f"{ds_out.attrs['timestamp']}"
    )
    fig.colorbar(pcm, ax=ax, label=f"{prop.out_name} [{prop.units}]")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run(
    i_rect: int,
    j_rect: int,
    timestamp: str,
    property: str = "density",
    output: str | None = None,
    config_path: Path = DEFAULT_CONFIG,
    clobber: bool = False,
    gen_qa_plot: bool = False,
) -> Path:
    """End-to-end pipeline: resolve tile -> load -> compute -> save NetCDF + PNG.

    Parameters
    ----------
    i_rect, j_rect : int
        Rect-grid pixel coordinates; any pixel inside the desired tile is OK.
    timestamp : str
        Snapshot timestamp in ``DATE_FMT``.
    property : str, default ``'density'``
        Key into :data:`TILE_PROPERTIES`; selects which field to extract.
    output : str or None
        Output path -- see :func:`_build_output_path`.
    config_path : Path
        Path to the YAML with an ``s3_source`` block.
    clobber: bool = False,
        If True, overwrite existing output files.
    gen_qa_plot: bool = False,
        If True, generate a QA plot next to the NetCDF.

    Returns
    -------
    Path
        Absolute path of the written NetCDF.

    Raises
    ------
    KeyError
        If ``property`` is not a registered key in :data:`TILE_PROPERTIES`.
    """
    if property not in TILE_PROPERTIES:
        raise KeyError(
            f"Unknown property '{property}'.  "
            f"Available: {sorted(TILE_PROPERTIES)}"
        )
    prop = TILE_PROPERTIES[property]

    # 1-2: resolve tile geometry and iteration number.
    tile = rect_ij_to_tile(i_rect, j_rect)
    iteration = _date_to_iteration(timestamp)
    logging.info(
        f"Resolved tile: idx={tile.tile_idx} "
        f"(tile_j_rect={tile.tile_j_rect}, tile_i_rect={tile.tile_i_rect})  "
        f"face={tile.face_idx}  "
        f"face j={tile.j_face_slice}, i={tile.i_face_slice}  "
        f"property={prop.name}"
    )

    # Create out_path
    out_path = _build_output_path(
        output, tile.tile_idx, timestamp, filename_prefix=prop.filename_prefix,
    )
    if out_path.exists() and not clobber:
        logging.info(f"Output file {out_path} already exists. Skipping.")
        return

    # Proceed
    s3_cfg = _load_s3_config(Path(config_path))

    # 3: load grid for the tile (used purely for output coords now -- no mask).
    ds_grid_tile = _load_grid_for_tile(s3_cfg, tile)

    # 4-5: lazy-open tracers (only the vars this property needs) and slice.
    ds_tracers_tile = _load_tracers_for_tile(
        s3_cfg, timestamp, tile, vars_needed=list(prop.vars_needed),
    )

    # 6: compute property (no masking).
    from IPython import embed; embed(header='compute_tile_property 643')
    field = compute_tile_property(ds_tracers_tile, prop)

    # 7: assemble output dataset with coords + provenance.
    ds_out = _build_output_dataset(
        field=field,
        ds_grid_tile=ds_grid_tile,
        tile=tile,
        prop=prop,
        date_str=timestamp,
        iteration=iteration,
        rect_i_user=i_rect,
        rect_j_user=j_rect,
    )

    # 8: resolve filename and save.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving NetCDF: {out_path}")
    ds_out.to_netcdf(
        out_path,
        engine="h5netcdf",
        encoding={
            prop.out_name: {"zlib": True, "complevel": 4, "dtype": "float32"},
        },
    )

    # 9: surface QA plot next to the NetCDF.
    if gen_qa_plot:
        png_path = out_path.with_suffix(".png")
        logging.info(f"Saving QA plot: {png_path}")
        _qa_plot(ds_out, prop, png_path)

    return out_path
