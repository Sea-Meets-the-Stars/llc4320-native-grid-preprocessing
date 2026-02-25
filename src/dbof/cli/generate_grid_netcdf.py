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

Usage
-----
  generate-global-grid --output /path/to/llc4320_grid.nc [--endpoint-url URL]

  Or from Python:
      from dbof.cli.generate_grid_netcdf import generate_grid_netcdf
      generate_grid_netcdf(output_path="/scratch/llc4320_grid.nc")
"""

import argparse
import logging
import sys

import numpy as np
import xarray as xr
from xmitgcm.llcreader import llcmodel

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data

DEFAULT_ENDPOINT = "https://mghp.osn.xsede.org"

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
    return llcmodel.faces_dataset_to_latlon(sub, metric_vector_pairs=[])


def generate_grid_netcdf(
    output_path: str,
    endpoint_url: str = DEFAULT_ENDPOINT,
) -> None:
    """
    Load the LLC4320 grid, convert to rectangular lat/lon, and write to NetCDF.

    Parameters
    ----------
    output_path : str
        Local file path for the output NetCDF (e.g. '/scratch/llc4320_grid.nc').
    endpoint_url : str
        OSN/S3 endpoint for the raw LLC4320 kerchunk grid file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # ------------------------------------------------------------------
    # 1. Load grid
    # ------------------------------------------------------------------
    logging.info("Fetching LLC4320 grid file...")
    co = get_raw_data.get_remote_gridfile(endpoint_url)
    ds_grid_raw = preproc_llc_core_data.process_llc4320_grid(co)
    logging.info(f"Grid variables loaded: {list(ds_grid_raw.data_vars)}")

    # ------------------------------------------------------------------
    # 2. Handle hFacC — select surface level if k dimension is present
    # ------------------------------------------------------------------
    hfacc = ds_grid_raw[_HFACC_VAR]
    if 'k' in hfacc.dims:
        logging.info("hFacC has k dimension; selecting surface level k=0")
        hfacc = hfacc.isel(k=0, drop=True)
    ds_hfacc = xr.Dataset({_HFACC_VAR: hfacc})

    # ------------------------------------------------------------------
    # 3. Convert each staggered-grid group to rectangular lat/lon
    # ------------------------------------------------------------------
    logging.info("Converting T-grid variables to rectangular lat/lon...")
    ds_t = _convert_group(ds_grid_raw, _T_GRID_VARS)

    logging.info("Converting U-grid variables to rectangular lat/lon...")
    ds_u = _convert_group(ds_grid_raw, _U_GRID_VARS)

    logging.info("Converting V-grid variables to rectangular lat/lon...")
    ds_v = _convert_group(ds_grid_raw, _V_GRID_VARS)

    logging.info("Converting Z-grid (vorticity) variables to rectangular lat/lon...")
    ds_z = _convert_group(ds_grid_raw, _Z_GRID_VARS)

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
    # 5. Write to NetCDF
    # ------------------------------------------------------------------
    logging.info(f"Writing grid NetCDF to {output_path} ...")
    ds_rect.to_netcdf(output_path)
    logging.info("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Extract LLC4320 static grid and save as rectangular NetCDF."
    )
    p.add_argument(
        "--output", required=True,
        help="Local output path, e.g. /scratch/llc4320_grid.nc",
    )
    p.add_argument(
        "--endpoint-url", default=DEFAULT_ENDPOINT,
        help=f"OSN/S3 endpoint URL for LLC4320 grid file (default: {DEFAULT_ENDPOINT})",
    )
    args = p.parse_args()
    generate_grid_netcdf(output_path=args.output, endpoint_url=args.endpoint_url)


if __name__ == "__main__":
    main()
