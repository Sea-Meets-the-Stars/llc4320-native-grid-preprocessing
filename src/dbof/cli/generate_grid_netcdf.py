"""
generate_grid_netcdf.py
-----------------------
Extracts LLC4320 static grid variables, converts them from
the native face format (13, 4320, 4320) to a rectangular lat/lon image
(12960, 17280), and saves the result as a self-describing NetCDF file.

This is intentionally separate from the main data pipeline: the grid is static
(never changes between timesteps) so it only needs to be generated once.

Grid variables saved
--------------------
  T-grid  (face, j, i)    → (j, i):  XC, YC, rA, Depth, hFacC, SN, CS
  U-grid  (face, j, i_g)  → (j, i):  dxC, dyG
  V-grid  (face, j_g, i)  → (j, i):  dyC, dxG
  Z-grid  (face, j_g, i_g)→ (j, i):  rAz

  Convenience aliases: lat = YC (degrees north), lon = XC (degrees east)

hFacC note
----------
  hFacC is 3-D in the raw grid file (face, j, i, k); only the surface level
  (k=0) is saved here. hFacC == 0 means land; hFacC > 0 means ocean.

Two separate S3 endpoints
--------------------------
  This script interacts with two distinct storage systems:

  1. OSN (input) — raw LLC4320 model data:
       https://mghp.osn.xsede.org
       This is a public read-only endpoint. No credentials required.
       The raw LLC4320 kerchunk grid file lives here.

  2. NRP S3 (optional output) — processed dataset storage:
       https://s3-west.nrp-nautilus.io
       This is the same bucket where generate_fronts_global.py writes Zarr data.
       Requires AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
       Pass --s3-endpoint / --bucket / --folder to enable upload after local write.

Usage
-----
  # Write to local HPC path only:
  generate-global-grid --output /scratch/llc4320_grid.nc

  # Write to local HPC path AND upload to NRP S3:
  generate-global-grid \\
      --output /scratch/llc4320_grid.nc \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data

  # Override the OSN source endpoint (rarely needed):
  generate-global-grid --output /scratch/llc4320_grid.nc \\
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

# Source: raw LLC4320 kerchunk files on OSN (public, read-only, no credentials needed)
DEFAULT_OSN_ENDPOINT = "https://mghp.osn.xsede.org"

def _faces_dataset_to_latlon(ds, metric_vector_pairs):
    """
    Xarray-version-safe replacement for llcmodel.faces_dataset_to_latlon.
    See generate_fronts_global.py for full explanation. The upstream function
    uses `ds_new = ds_new.update(data_vars)` which returns None in xarray < 0.17.
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


# Variables grouped by their native staggered grid.
# faces_dataset_to_latlon needs homogeneous stagger per call when mixing
# i/j with i_g/j_g, so we convert each group separately then merge.
_T_GRID_VARS  = ['XC', 'YC', 'rA', 'Depth', 'SN', 'CS']  # (face, j, i)
_U_GRID_VARS  = ['dxC', 'dyG']                             # (face, j, i_g)
_V_GRID_VARS  = ['dyC', 'dxG']                             # (face, j_g, i)
_Z_GRID_VARS  = ['rAz']                                    # (face, j_g, i_g)
_HFACC_VAR    = 'hFacC'                                    # (face, j, i [, k])


def _convert_group(ds_grid: xr.Dataset, var_names: list) -> xr.Dataset:
    """
    Convert a subset of ds_grid variables through faces_dataset_to_latlon.

    All variables in *var_names* must share the same staggered grid (so that
    the face dimension is handled identically for each).  Grid variables are
    scalar fields, so metric_vector_pairs is always empty.
    """
    present = [v for v in var_names if v in ds_grid]
    if not present:
        return xr.Dataset()
    sub = ds_grid[present]
    return _faces_dataset_to_latlon(sub, metric_vector_pairs=[])


def generate_grid_netcdf(
    output_path: str,
    osn_endpoint: str = DEFAULT_OSN_ENDPOINT,
    s3_endpoint: str = None,
    bucket: str = None,
    folder: str = None,
) -> None:
    """
    Load the LLC4320 grid, convert to rectangular lat/lon, and write to NetCDF.

    Parameters
    ----------
    output_path : str
        Local HPC file path for the output NetCDF (e.g. '/scratch/llc4320_grid.nc').
    osn_endpoint : str
        OSN endpoint for reading the raw LLC4320 kerchunk grid file.
        Default: https://mghp.osn.xsede.org  (public, no credentials needed).
    s3_endpoint : str, optional
        NRP S3 endpoint for optional upload after local write
        (e.g. 'https://s3-west.nrp-nautilus.io'). Requires AWS credentials.
        If None, no S3 upload is performed.
    bucket : str, optional
        S3 bucket name for optional upload (e.g. 'dbof').
    folder : str, optional
        S3 folder for optional upload (e.g. 'native_grid_dbof_training_data').
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # ------------------------------------------------------------------
    # 1. Load grid from OSN (raw LLC4320 source, public read-only)
    # ------------------------------------------------------------------
    # IMPORTANT: we use co (the raw consolidated kerchunk dataset) directly for
    # faces_dataset_to_latlon, NOT the output of process_llc4320_grid().
    # process_llc4320_grid() calls reset_coords() which strips face/i/j as proper
    # coordinate variables, leaving them as bare dimension names.
    # faces_dataset_to_latlon needs those coordinate values to detect the LLC4320
    # topology and stitch faces correctly; without them it silently returns None.
    logging.info(f"Fetching LLC4320 grid file from OSN: {osn_endpoint}")
    co = get_raw_data.get_remote_gridfile(osn_endpoint)
    # co is the raw xarray Dataset with full coordinate structure from the kerchunk reader.
    # ds_grid_raw is kept only for the variable name list and hFacC k-selection.
    ds_grid_raw = preproc_llc_core_data.process_llc4320_grid(co)
    logging.info(f"Grid variables to extract: {list(ds_grid_raw.data_vars)}")

    # ------------------------------------------------------------------
    # 2. Handle hFacC — select surface level if k dimension is present
    # ------------------------------------------------------------------
    # Use co (raw) for hFacC since it retains proper coordinate structure.
    hfacc = co[_HFACC_VAR]
    if 'k' in hfacc.dims:
        logging.info("hFacC has k dimension; selecting surface level k=0")
        hfacc = hfacc.isel(k=0, drop=True)
    # Build a single-variable dataset that still carries all the original coords
    # (face, i, j) so faces_dataset_to_latlon can detect the LLC4320 topology.
    ds_hfacc = co[[_HFACC_VAR]].isel(k=0, drop=True) if 'k' in co[_HFACC_VAR].dims \
               else co[[_HFACC_VAR]]

    # ------------------------------------------------------------------
    # 3. Convert each staggered-grid group to rectangular lat/lon
    # ------------------------------------------------------------------
    # All _convert_group calls use co (raw dataset) so that face/i/j coordinate
    # values are present and faces_dataset_to_latlon can stitch correctly.
    logging.info("Converting T-grid variables to rectangular lat/lon...")
    ds_t = _convert_group(co, _T_GRID_VARS)

    logging.info("Converting U-grid variables to rectangular lat/lon...")
    ds_u = _convert_group(co, _U_GRID_VARS)

    logging.info("Converting V-grid variables to rectangular lat/lon...")
    ds_v = _convert_group(co, _V_GRID_VARS)

    logging.info("Converting Z-grid (vorticity) variables to rectangular lat/lon...")
    ds_z = _convert_group(co, _Z_GRID_VARS)

    logging.info("Converting hFacC to rectangular lat/lon...")
    ds_h = _convert_group(ds_hfacc, [_HFACC_VAR])

    # ------------------------------------------------------------------
    # 4. Merge all groups and add convenience aliases
    # ------------------------------------------------------------------
    ds_rect = xr.merge([ds_t, ds_u, ds_v, ds_z, ds_h])
    logging.info(f"Rectangular grid shape: {ds_rect.dims}")

    # Convenience aliases so downstream code can use 'lat' / 'lon' directly.
    if 'YC' in ds_rect:
        ds_rect['lat'] = ds_rect['YC']
        ds_rect['lat'].attrs.update({'long_name': 'latitude',  'units': 'degrees_north'})
    if 'XC' in ds_rect:
        ds_rect['lon'] = ds_rect['XC']
        ds_rect['lon'].attrs.update({'long_name': 'longitude', 'units': 'degrees_east'})

    # Attach informative attributes to each variable where not already set.
    _attrs = {
        'XC':    {'long_name': 'longitude of T-cell center',          'units': 'degrees_east'},
        'YC':    {'long_name': 'latitude of T-cell center',           'units': 'degrees_north'},
        'rA':    {'long_name': 'T-cell area',                         'units': 'm^2'},
        'Depth': {'long_name': 'ocean depth (positive downward)',      'units': 'm'},
        'SN':    {'long_name': 'sine of grid angle (rotation SN)',     'units': ''},
        'CS':    {'long_name': 'cosine of grid angle (rotation CS)',   'units': ''},
        'dxC':   {'long_name': 'grid spacing in x at U-face',         'units': 'm'},
        'dyG':   {'long_name': 'grid spacing in y at U-face',         'units': 'm'},
        'dyC':   {'long_name': 'grid spacing in y at V-face',         'units': 'm'},
        'dxG':   {'long_name': 'grid spacing in x at V-face',         'units': 'm'},
        'rAz':   {'long_name': 'vorticity-cell area',                  'units': 'm^2'},
        'hFacC': {'long_name': 'fractional open cell thickness (k=0)', 'units': '',
                  'note': '0 = land, >0 = ocean (surface level only)'},
    }
    for var, attrs in _attrs.items():
        if var in ds_rect:
            existing = ds_rect[var].attrs
            ds_rect[var].attrs = {**attrs, **existing}

    ds_rect.attrs['description'] = (
        'LLC4320 static grid variables converted from native face format to '
        'rectangular lat/lon image via xmitgcm.llcreader.faces_dataset_to_latlon. '
        'Shape: (12960, 17280) = 3×4320 × 4×4320. No geographic interpolation.'
    )

    # ------------------------------------------------------------------
    # 5. Write to local HPC path
    # ------------------------------------------------------------------
    logging.info(f"Writing grid NetCDF to {output_path} ...")
    ds_rect.to_netcdf(output_path)
    logging.info(f"Local write complete: {output_path}")

    # ------------------------------------------------------------------
    # 6. Optional: upload to NRP S3 (same bucket as processed Zarr data)
    # ------------------------------------------------------------------
    if s3_endpoint is not None:
        if bucket is None or folder is None:
            raise ValueError(
                "--bucket and --folder are required when --s3-endpoint is set."
            )
        import fsspec
        bucket  = bucket.strip().strip('/')
        folder  = folder.strip().strip('/')
        fname   = output_path.rsplit('/', 1)[-1]  # basename of local file
        s3_path = f"s3://{bucket}/{folder}/{fname}"
        logging.info(f"Uploading grid file to NRP S3: {s3_path}")
        _, fs_synch = create_s3_filesystems(s3_endpoint)
        fs_synch.put(output_path, s3_path)
        logging.info(f"S3 upload complete: {s3_path}")

    logging.info("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Extract LLC4320 static grid and save as rectangular NetCDF."
    )

    p.add_argument(
        "--output", required=True,
        help="Local HPC output path, e.g. /scratch/llc4320_grid.nc",
    )
    p.add_argument(
        "--osn-endpoint", default=DEFAULT_OSN_ENDPOINT,
        help=(
            f"OSN endpoint for reading raw LLC4320 grid data "
            f"(default: {DEFAULT_OSN_ENDPOINT}). Public read-only, no credentials needed."
        ),
    )

    # Optional NRP S3 upload
    s3_group = p.add_argument_group(
        "NRP S3 upload (optional)",
        "If --s3-endpoint is given, the NetCDF will also be uploaded to the NRP S3 "
        "bucket after the local write. Requires AWS credentials in environment."
    )
    s3_group.add_argument('--s3-endpoint', default=None,
                          help="NRP S3 endpoint, e.g. https://s3-west.nrp-nautilus.io")
    s3_group.add_argument('--bucket',      default=None,
                          help="S3 bucket name, e.g. dbof")
    s3_group.add_argument('--folder',      default=None,
                          help="S3 folder, e.g. native_grid_dbof_training_data")

    args = p.parse_args()
    generate_grid_netcdf(
        output_path=args.output,
        osn_endpoint=args.osn_endpoint,
        s3_endpoint=args.s3_endpoint,
        bucket=args.bucket,
        folder=args.folder,
    )


if __name__ == "__main__":
    main()
