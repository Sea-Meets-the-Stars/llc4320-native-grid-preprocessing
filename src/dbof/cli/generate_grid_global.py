"""
generate_grid_global.py
-----------------------
Extracts LLC4320 static grid variables, converts them from the native face
format (13, 4320, 4320) to a rectangular lat/lon image (12960, 17280), and
saves the result directly to an S3 Zarr store alongside the snapshot data
produced by generate_fronts_global.py.

This is the S3 Zarr counterpart to generate_grid_netcdf.py (which writes
a local NetCDF).  Having the grid in the same S3 location as the snapshots
makes it easy to load both from the same bucket in notebooks or inference
pipelines without managing a separate local file.

Grid variables saved
--------------------
  T-grid  (face, j, i)    → (j, i):  XC, YC, rA, Depth, hFacC, SN, CS
  U-grid  (face, j, i_g)  → (j, i):  dxC, dyG
  V-grid  (face, j_g, i)  → (j, i):  dyC, dxG
  Z-grid  (face, j_g, i_g)→ (j, i):  rAz

  Convenience aliases: lat = YC, lon = XC

Store location
--------------
  s3://{bucket}/{folder}/{dataset_name}
  (default dataset_name: llc4320_grid.zarr)

  This sits in the same S3 folder as the snapshot Zarr stores, so notebooks
  can reach both via the same bucket/folder config.

Two separate S3 endpoints
--------------------------
  1. OSN (input):   https://mghp.osn.xsede.org — public, read-only
  2. NRP S3 (output): https://s3-west.nrp-nautilus.io — requires credentials

Usage
-----
  generate-global-grid-zarr \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data

  # Use a custom store name:
  generate-global-grid-zarr \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data \\
      --dataset-name my_grid.zarr

  # Override the OSN source endpoint (rarely needed):
  generate-global-grid-zarr \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data \\
      --osn-endpoint https://mghp.osn.xsede.org
"""

import argparse
import logging
import sys

import numpy as np
import xarray as xr
from xmitgcm.llcreader import llcmodel
from xmitgcm.llcreader.llcmodel import (
    _faces_coords_to_latlon,
    _faces_to_latlon_scalar,
    _faces_to_latlon_vector,
    _drop_facedim,
)

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
from dbof.io.filesystems import create_s3_filesystems
from dbof.dataset_creation.zarr_grid_global import GlobalGridZarrWriter

# Source: raw LLC4320 kerchunk files on OSN (public, read-only, no credentials needed)
DEFAULT_OSN_ENDPOINT  = "https://mghp.osn.xsede.org"
DEFAULT_DATASET_NAME  = "llc4320_grid.zarr"

# Variables grouped by their native staggered grid.
# _faces_dataset_to_latlon needs homogeneous stagger per call when mixing
# i/j with i_g/j_g, so we convert each group separately then merge.
_T_GRID_VARS = ['XC', 'YC', 'rA', 'Depth', 'SN', 'CS']   # (face, j, i)
_U_GRID_VARS = ['dxC', 'dyG']                               # (face, j, i_g)
_V_GRID_VARS = ['dyC', 'dxG']                               # (face, j_g, i)
_Z_GRID_VARS = ['rAz']                                      # (face, j_g, i_g)
_HFACC_VAR   = 'hFacC'                                      # (face, j, i [, k])


# ---------------------------------------------------------------------------
# Xarray-version-safe faces_dataset_to_latlon
# ---------------------------------------------------------------------------

def _faces_dataset_to_latlon(ds, metric_vector_pairs):
    """
    Xarray-version-safe replacement for llcmodel.faces_dataset_to_latlon.

    The upstream function uses ``ds_new = ds_new.update(data_vars)`` which
    returns None in xarray < 0.17, causing an AttributeError on the next line.
    This version uses xr.merge() instead, which has stable semantics across
    all xarray versions.
    """
    coord_vars = list(ds.coords)
    ds_new = _faces_coords_to_latlon(ds)
    nfaces = len(ds['face'])

    vector_pairs = []
    vnames = list(ds.reset_coords().variables)
    for vname in list(vnames):
        try:
            mate = ds[vname].attrs['mate']
        except KeyError:
            mate = None
        if mate is not None:
            vector_pairs.append((vname, mate))
            try:
                vnames.remove(mate)
            except ValueError:
                raise ValueError(
                    f"If '{vname}' in varnames, '{mate}' must also be in varnames"
                )

    all_vector_components = [
        inner for outer in (vector_pairs + metric_vector_pairs) for inner in outer
    ]
    scalars = [v for v in vnames if v not in all_vector_components]
    data_vars = {}

    for vname in scalars:
        if vname == 'face' or vname in ds_new:
            continue
        if 'face' in ds[vname].dims:
            data = _faces_to_latlon_scalar(ds[vname].data, nfaces=nfaces)
            dims = _drop_facedim(ds[vname].dims)
        else:
            data = ds[vname].data
            dims = ds[vname].dims
        data_vars[vname] = xr.Variable(dims, data, ds[vname].attrs)

    for vname_u, vname_v in vector_pairs:
        u_data, v_data = _faces_to_latlon_vector(
            ds[vname_u].data, ds[vname_v].data, nfaces=nfaces
        )
        data_vars[vname_u] = xr.Variable(
            _drop_facedim(ds[vname_u].dims), u_data, ds[vname_u].attrs
        )
        data_vars[vname_v] = xr.Variable(
            _drop_facedim(ds[vname_v].dims), v_data, ds[vname_v].attrs
        )

    for vname_u, vname_v in metric_vector_pairs:
        u_data, v_data = _faces_to_latlon_vector(
            ds[vname_u].data, ds[vname_v].data, nfaces=nfaces, metric=True
        )
        data_vars[vname_u] = xr.Variable(
            _drop_facedim(ds[vname_u].dims), u_data, ds[vname_u].attrs
        )
        data_vars[vname_v] = xr.Variable(
            _drop_facedim(ds[vname_v].dims), v_data, ds[vname_v].attrs
        )

    ds_out = xr.merge([ds_new, xr.Dataset(data_vars)])
    ds_out = ds_out.set_coords([c for c in coord_vars if c in ds_out])
    return ds_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_group(ds_grid: xr.Dataset, var_names: list) -> xr.Dataset:
    """
    Convert a homogeneous-stagger subset of *ds_grid* to rectangular lat/lon.

    All variables in *var_names* must share the same staggered grid so that
    the face dimension is handled identically for each.  Grid variables are
    scalar fields, so metric_vector_pairs is always empty.
    """
    present = [v for v in var_names if v in ds_grid]
    if not present:
        return xr.Dataset()
    return _faces_dataset_to_latlon(ds_grid[present], metric_vector_pairs=[])


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def generate_grid_zarr(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    osn_endpoint: str = DEFAULT_OSN_ENDPOINT,
) -> None:
    """
    Load the LLC4320 grid, convert to rectangular lat/lon, and write to S3 Zarr.

    Parameters
    ----------
    s3_endpoint : str
        NRP S3 endpoint for writing output
        (e.g. 'https://s3-west.nrp-nautilus.io'). Requires AWS credentials.
    bucket : str
        S3 bucket name (e.g. 'dbof').
    folder : str
        S3 folder (e.g. 'native_grid_dbof_training_data').
    dataset_name : str
        Zarr store name within *folder* (default: 'llc4320_grid.zarr').
    osn_endpoint : str
        OSN endpoint for reading raw LLC4320 grid data (public, no credentials).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # ------------------------------------------------------------------
    # 1. Load grid from OSN
    # ------------------------------------------------------------------
    # Use the raw kerchunk dataset (co) directly — NOT the output of
    # process_llc4320_grid(), which calls reset_coords() and strips face/i/j
    # as proper coordinate variables. _faces_dataset_to_latlon needs those
    # coordinate values to detect the LLC4320 topology.
    logging.info(f"Fetching LLC4320 grid from OSN: {osn_endpoint}")
    co = get_raw_data.get_remote_gridfile(osn_endpoint)
    logging.info(f"Grid loaded. Variables: {list(co.data_vars)}")

    # ------------------------------------------------------------------
    # 2. Handle hFacC — select surface level if k dimension present
    # ------------------------------------------------------------------
    ds_hfacc = (
        co[[_HFACC_VAR]].isel(k=0, drop=True)
        if 'k' in co[_HFACC_VAR].dims
        else co[[_HFACC_VAR]]
    )
    if 'k' in co[_HFACC_VAR].dims:
        logging.info("hFacC has k dimension; selecting surface level k=0")

    # ------------------------------------------------------------------
    # 3. Convert each staggered-grid group to rectangular lat/lon
    # ------------------------------------------------------------------
    logging.info("Converting T-grid variables (XC, YC, rA, Depth, SN, CS)...")
    ds_t = _convert_group(co, _T_GRID_VARS)

    logging.info("Converting U-grid variables (dxC, dyG)...")
    ds_u = _convert_group(co, _U_GRID_VARS)

    logging.info("Converting V-grid variables (dyC, dxG)...")
    ds_v = _convert_group(co, _V_GRID_VARS)

    logging.info("Converting Z-grid variables (rAz)...")
    ds_z = _convert_group(co, _Z_GRID_VARS)

    logging.info("Converting hFacC (surface level)...")
    ds_h = _convert_group(ds_hfacc, [_HFACC_VAR])

    # ------------------------------------------------------------------
    # 4. Merge and attach variable attributes
    # ------------------------------------------------------------------
    ds_rect = xr.merge([ds_t, ds_u, ds_v, ds_z, ds_h])
    logging.info(f"Rectangular grid shape: {dict(ds_rect.dims)}")

    _attrs = {
        'XC':    {'long_name': 'longitude of T-cell centre',           'units': 'degrees_east'},
        'YC':    {'long_name': 'latitude of T-cell centre',            'units': 'degrees_north'},
        'rA':    {'long_name': 'T-cell area',                          'units': 'm^2'},
        'Depth': {'long_name': 'ocean depth (positive downward)',       'units': 'm'},
        'SN':    {'long_name': 'sine of grid-rotation angle',          'units': ''},
        'CS':    {'long_name': 'cosine of grid-rotation angle',        'units': ''},
        'dxC':   {'long_name': 'grid spacing in x at U-face',          'units': 'm'},
        'dyG':   {'long_name': 'grid spacing in y at U-face',          'units': 'm'},
        'dyC':   {'long_name': 'grid spacing in y at V-face',          'units': 'm'},
        'dxG':   {'long_name': 'grid spacing in x at V-face',          'units': 'm'},
        'rAz':   {'long_name': 'vorticity-cell area',                  'units': 'm^2'},
        'hFacC': {'long_name': 'fractional open cell thickness (k=0)',  'units': '',
                  'note': '0 = land, >0 = ocean (surface level only)'},
    }
    for var, attrs in _attrs.items():
        if var in ds_rect:
            ds_rect[var].attrs = {**attrs, **ds_rect[var].attrs}

    # ------------------------------------------------------------------
    # 5. Write to S3 Zarr
    # ------------------------------------------------------------------
    fs, _ = create_s3_filesystems(s3_endpoint)
    writer = GlobalGridZarrWriter(
        bucket=bucket,
        folder=folder,
        dataset_name=dataset_name,
        fs=fs,
    )
    store_path = (
        f"s3://{bucket.strip('/')}/{folder.strip('/')}/{dataset_name}"
    )
    logging.info(f"Writing grid to S3 Zarr: {store_path}")
    writer.write(ds_rect)
    logging.info(f"Done. Variables written: {writer.root.attrs.get('variables', [])}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Extract LLC4320 static grid and save to S3 Zarr "
            "(alongside the snapshot data from generate_fronts_global.py)."
        )
    )
    p.add_argument(
        "--s3-endpoint", required=True,
        help="NRP S3 endpoint, e.g. https://s3-west.nrp-nautilus.io. "
             "Requires AWS credentials in environment.",
    )
    p.add_argument(
        "--bucket", required=True,
        help="S3 bucket name, e.g. dbof",
    )
    p.add_argument(
        "--folder", required=True,
        help="S3 folder, e.g. native_grid_dbof_training_data",
    )
    p.add_argument(
        "--dataset-name", default=DEFAULT_DATASET_NAME,
        help=f"Zarr store name within --folder (default: {DEFAULT_DATASET_NAME})",
    )
    p.add_argument(
        "--osn-endpoint", default=DEFAULT_OSN_ENDPOINT,
        help=(
            f"OSN endpoint for reading raw LLC4320 grid data "
            f"(default: {DEFAULT_OSN_ENDPOINT}). Public, no credentials needed."
        ),
    )

    args = p.parse_args()
    generate_grid_zarr(
        s3_endpoint=args.s3_endpoint,
        bucket=args.bucket,
        folder=args.folder,
        dataset_name=args.dataset_name,
        osn_endpoint=args.osn_endpoint,
    )


if __name__ == "__main__":
    main()
