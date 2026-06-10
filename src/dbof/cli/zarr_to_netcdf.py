"""
zarr_to_netcdf.py
-----------------
Two modes (selected via --mode):

  snapshots (default)
  -------------------
  Reads global LLC4320 snapshots from the S3 Zarr store written by
  generate_global.py and writes one NetCDF
  file per timestep to a local directory on the HPC machine.

  All zarr stores are assumed to live under a date_prefix subdirectory::

      s3://{bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}

  Output file naming (default):
    <output_dir>/<run_id>_<date_prefix>.nc
    e.g.  global_depth_test00_20121109_120000.nc

  Each file is a self-describing xr.Dataset with:
    - One data variable per channel (dims: y, x)
    - Global attributes: date_prefix, run_id, channel_names, etc.

  grid
  ----
  Reads the static LLC4320 grid Zarr store written by generate_grid_global.py
  and writes a single NetCDF file containing all grid variables (XC, YC,
  Depth, hFacC, rA, SN, CS, dxC, dyG, dyC, dxG, rAz, lat, lon).

  Output file naming:
    <output_dir>/<grid_dataset_name>.nc   (default: llc4320_grid.nc)

  The grid file is stored at s3://{bucket}/{folder}/{grid_dataset_name}
  (no run_id in the path).

Grid note
---------
Geographic coordinates (lat/lon, grid spacings, etc.) are NOT included in
the per-timestep snapshot files. Load the companion grid file for that
information.

Usage — snapshots
-----------------
  # All channels, one date:
  zarr-to-netcdf \\
      --mode snapshots \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder depth_fields \\
      --run-id global_depth_test00 \\
      --dataset-name stratification.zarr \\
      --dates '2012-11-09 12:00:00' \\
      --output-dir /scratch/llc4320_netcdf/

  # Single channel with custom output filename:
  zarr-to-netcdf \
      --mode snapshots \
      --s3-endpoint https://s3-west.nrp-nautilus.io \
      --bucket dbof \
      --folder depth_fields \
      --run-id global_DEPTH_test01 \
      --dataset-name stratification.zarr \
      --dates '2012-11-09 12:00:00' \
      --channels N2_sfc \
      --output-dir /mnt/tank/Oceanography/data/OGCM/LLC/Fronts/vtest/20121109_120000 \
      --output-filename LLC4320_2012-11-09T12_00_00_N2_sfc.nc

  # All dates (reads all date_prefix subdirectories from the store):
  zarr-to-netcdf \\
      --mode snapshots \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder depth_fields \\
      --run-id global_depth_test00 \\
      --dataset-name stratification.zarr \\
      --output-dir /scratch/llc4320_netcdf/

  # With ice mask — mask all points where SIarea > 0 (from icearea.zarr):
  zarr-to-netcdf \\
      --mode snapshots \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder depth_fields \\
      --run-id global_depth_test00 \\
      --dataset-name stratification.zarr \\
      --dates '2012-11-09 12:00:00' \\
      --output-dir /scratch/llc4320_netcdf/ \\
      --ice-mask

  # Ice mask with a custom SIarea store name:
  zarr-to-netcdf \\
      --mode snapshots \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder depth_fields \\
      --run-id global_depth_test00 \\
      --dataset-name stratification.zarr \\
      --dates '2012-11-09 12:00:00' \\
      --output-dir /scratch/llc4320_netcdf/ \\
      --ice-mask --ice-mask-dataset-name my_siarea.zarr

Usage — grid
------------
  zarr-to-netcdf \\
      --mode grid \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data \\
      --grid-dataset-name llc4320_grid.zarr \\
      --output-dir /scratch/llc4320_netcdf/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from dbof.io.filesystems import create_s3_filesystems
from dbof.global_dataset_creation.iterations import (
    date_to_run_id as _date_to_prefix,
    prefix_to_display as _prefix_to_display,
)
import dbof.global_dataset_creation.zarr_dataset_global as zarr_dataset_global
import dbof.global_dataset_creation.zarr_grid_global as zarr_grid_global
from dbof.preprocessing.ice_mask import load_siarea_mask, apply_ice_mask


# ---------------------------------------------------------------------------
# S3 date-prefix discovery
# ---------------------------------------------------------------------------

def _discover_date_prefixes(bucket, folder, run_id, dataset_name, fs):
    """List available date_prefix subdirectories for a given run_id.

    Scans s3://{bucket}/{folder}/{run_id}/ for subdirectories that contain
    {dataset_name}, and returns sorted date_prefix strings.
    """
    run_prefix = f"{bucket}/{folder}/{run_id}"
    logging.info(f"Scanning for date_prefix subdirectories under: {run_prefix}/")

    try:
        entries = fs.ls(run_prefix, detail=False)
    except FileNotFoundError:
        logging.warning(f"Run prefix not found: {run_prefix}")
        return []

    prefixes = []
    for entry in entries:
        # entry looks like 'dbof/depth_fields/run_id/20121109_120000'
        candidate = entry.rstrip("/").split("/")[-1]
        # Check that this directory actually contains the target dataset
        dataset_path = f"{entry}/{dataset_name}"
        if fs.exists(dataset_path):
            prefixes.append(candidate)

    prefixes.sort()
    logging.info(f"Found {len(prefixes)} date_prefix(es): {prefixes}")
    return prefixes


# ---------------------------------------------------------------------------
# Mode: snapshots
# ---------------------------------------------------------------------------

def zarr_to_netcdf(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    run_id: str,
    dataset_name: str,
    output_dir: str,
    date_prefix: str,
    output_filename: str = None,
    channels: list = None,
    fs=None,
    ice_mask_dataset_name: str = None,
) -> None:
    """
    Convert a single date_prefix snapshot from S3 Zarr to a NetCDF file.

    Parameters
    ----------
    s3_endpoint : str
        NRP S3 endpoint, e.g. 'https://s3-west.nrp-nautilus.io'.
    bucket : str
        S3 bucket name, e.g. 'dbof'.
    folder : str
        S3 folder path, e.g. ``'surface_fields'`` or ``'depth_fields'``.
    run_id : str
        The run_id used when writing the Zarr store.
    dataset_name : str
        The dataset_name used when writing the Zarr store
        (e.g. 'stratification.zarr').
    output_dir : str
        Local directory where NetCDF files will be written.
        Created if it does not exist.
    date_prefix : str
        Date subdirectory under run_id (e.g. '20121109_120000').
    output_filename : str, optional
        Fixed output filename (stem + extension, e.g. 'N2_sfc.nc').
        If None, auto-generated as ``{run_id}_{date_prefix}.nc``.
    channels : list of str, optional
        Subset of channel names to include in the output file.
        If None, all channels are written.  Unknown names raise ValueError.
    fs : s3fs filesystem, optional
        Reuse an existing s3fs filesystem; created if None.
    ice_mask_dataset_name : str, optional
        If provided (e.g. ``"icearea.zarr"``), load SIarea from this
        zarr store (same bucket/folder/run_id/date_prefix) and mask all
        exported channels: points where SIarea > 0 are set to NaN.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Open Zarr reader
    # ------------------------------------------------------------------
    store_path = zarr_dataset_global.make_run_prefix(
        bucket, folder, run_id, dataset_name, date_prefix=date_prefix,
    )
    logging.info(f"Opening snapshot Zarr store: {store_path}")
    if fs is None:
        _, fs = create_s3_filesystems(s3_endpoint)
    reader = zarr_dataset_global.GlobalZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name=dataset_name,
        fs=fs,
        date_prefix=date_prefix,
    )

    n_total = len(reader)
    all_channel_names = reader.channel_names
    H, W = reader.rectangular_shape
    logging.info(
        f"Store contains {n_total} timestep(s), {len(all_channel_names)} channel(s): "
        f"{all_channel_names}  |  spatial shape: {H} × {W}"
    )

    # ------------------------------------------------------------------
    # 1b. Optionally load the SIarea ice mask
    # ------------------------------------------------------------------
    ice_mask = None
    if ice_mask_dataset_name:
        logging.info(
            f"Loading ice mask from {ice_mask_dataset_name} "
            f"(date_prefix={date_prefix})"
        )
        ice_mask = load_siarea_mask(
            bucket=bucket,
            folder=folder,
            run_id=run_id,
            date_prefix=date_prefix,
            fs=fs,
            dataset_name=ice_mask_dataset_name,
        )

    # Resolve channel subset
    if channels is None:
        channel_names = all_channel_names
    else:
        unknown = [c for c in channels if c not in all_channel_names]
        if unknown:
            raise ValueError(
                f"Requested channel(s) {unknown} not found in store. "
                f"Available: {all_channel_names}"
            )
        channel_names = channels
        logging.info(f"Writing subset of channels: {channel_names}")

    # ------------------------------------------------------------------
    # 2. Convert each timestep in this date_prefix store
    # ------------------------------------------------------------------
    for t in range(n_total):
        date_display = _prefix_to_display(date_prefix)
        logging.info(
            f"[{t + 1}/{n_total}] date_prefix={date_prefix} "
            f"({date_display}), t={t}"
        )

        n_y, n_x = H, W
        y_coord = np.arange(n_y, dtype=np.int32)
        x_coord = np.arange(n_x, dtype=np.int32)

        data_vars = {}
        for ch in channel_names:
            arr = reader.get_channel_snapshot(t, ch).astype(np.float32)
            if ice_mask is not None:
                arr = apply_ice_mask(arr, ice_mask)
            data_vars[ch] = xr.DataArray(
                arr, dims=['y', 'x'],
                coords={'y': y_coord, 'x': x_coord},
            )

        ds = xr.Dataset(
            data_vars,
            attrs={
                'date_prefix':       date_prefix,
                'model_time_utc':    date_display,
                'run_id':            run_id,
                'channel_names':     channel_names,
                'spatial_shape':     f'({n_y}, {n_x})',
                'source_zarr':       store_path,
                'description': (
                    'LLC4320 global snapshot in rectangular lat/lon format '
                    f'({n_y} × {n_x}). Produced by zarr_to_netcdf.py. '
                    'Load companion llc4320_grid.nc for XC/YC/Depth/etc.'
                ),
            },
        )

        if output_filename is not None:
            nc_filename = out_path / output_filename
        else:
            nc_filename = out_path / f"{run_id}_{date_prefix}.nc"
        ds.to_netcdf(nc_filename)
        logging.info(f"  → written: {nc_filename}")

    logging.info(f"Done. {n_total} file(s) written to {out_path}")


# ---------------------------------------------------------------------------
# Mode: grid
# ---------------------------------------------------------------------------

def grid_zarr_to_netcdf(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    grid_dataset_name: str = "llc4320_grid.zarr",
    output_dir: str = ".",
    output_filename: str = "llc4320_grid.nc",
) -> None:
    """
    Convert the static grid Zarr store on S3 to a single NetCDF file.

    Parameters
    ----------
    s3_endpoint : str
        NRP S3 endpoint, e.g. 'https://s3-west.nrp-nautilus.io'.
    bucket : str
        S3 bucket name, e.g. 'dbof'.
    folder : str
        S3 folder path, e.g. 'native_grid_dbof_training_data'.
    grid_dataset_name : str
        Zarr store name within ``folder`` (default: ``llc4320_grid.zarr``).
    output_dir : str
        Local directory where the output NetCDF will be written.
    output_filename : str
        Output filename (default: ``llc4320_grid.nc``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    store_path = f"{bucket}/{folder}/{grid_dataset_name}"
    logging.info(f"Opening grid Zarr store: {store_path}")
    _, fs = create_s3_filesystems(s3_endpoint)

    reader = zarr_grid_global.GlobalGridZarrReader(
        bucket=bucket,
        folder=folder,
        dataset_name=grid_dataset_name,
        fs=fs,
    )

    var_names = reader.variable_names
    H, W = reader.shape
    logging.info(f"Grid store: {len(var_names)} variable(s), shape: {H} × {W}")
    logging.info(f"  Variables: {var_names}")

    # Read all grid variables into an xarray Dataset.
    y_coord = np.arange(H, dtype=np.int32)
    x_coord = np.arange(W, dtype=np.int32)

    data_vars = {}
    for vname in var_names:
        logging.info(f"  Reading: {vname}")
        arr = reader.get_variable(vname).astype(np.float32)
        attrs = reader.get_variable_attrs(vname)
        data_vars[vname] = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': y_coord, 'x': x_coord},
            attrs=attrs,
        )

    # Promote XC/YC to coordinate variables (lat/lon).
    coords = {'y': y_coord, 'x': x_coord}
    if 'XC' in data_vars:
        coords['lon'] = data_vars.pop('XC')
    if 'YC' in data_vars:
        coords['lat'] = data_vars.pop('YC')

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            'source_zarr': store_path,
            'description': (
                'LLC4320 static grid variables in rectangular lat/lon format '
                f'(shape: {H} × {W} = 3×4320 × 4×4320). '
                'Produced by zarr_to_netcdf.py --mode grid from the S3 Zarr store. '
                'XC/lon = longitude (degrees east), YC/lat = latitude (degrees north). '
                'Depth = ocean depth (m), hFacC = fractional open thickness '
                '(0 = land). See variable attributes for details.'
            ),
        },
    )

    nc_path = out_path / output_filename
    ds.to_netcdf(nc_path)
    logging.info(f"Grid written to: {nc_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    output_dir: str = None,
    output_filename: str = None,
    mode: str = 'snapshots',
    s3_endpoint: str = 'https://s3-west.nrp-nautilus.io',
    bucket: str = 'dbof',
    folder: str = None,
    run_id: str = None,
    dataset_name: str = 'dataset_creation.zarr',
    dates: list = None,
    channels: list = None,
    grid_dataset_name: str = 'llc4320_grid.zarr',
    grid_output_filename: str = 'llc4320_grid.nc',
    date_prefix: str = None,
    ice_mask: bool = False,
    ice_mask_dataset_name: str = 'icearea.zarr',
) -> None:
    """Entry point callable from other modules or the CLI.

    Parameters
    ----------
    mode : str
        'snapshots' or 'grid'.
    s3_endpoint : str
        NRP S3 endpoint, e.g. 'https://s3-west.nrp-nautilus.io'.
    bucket : str
        S3 bucket name.
    folder : str
        S3 folder path.
    output_dir : str
        Local directory to write NetCDF file(s) into.
    run_id : str
        [snapshots] run_id used when writing the Zarr store.
    dataset_name : str
        [snapshots] Zarr dataset name (default: 'dataset_creation.zarr').
    dates : list of str, optional
        [snapshots] Model dates in 'YYYY-MM-DD HH:MM:SS' format.
        Each date is converted to a date_prefix and the corresponding
        zarr store is opened.  If None and date_prefix is None, all
        available date_prefixes are auto-discovered.
    channels : list of str, optional
        [snapshots] Channel name(s) to save. Default: all.
    date_prefix : str, optional
        [snapshots] Explicit date_prefix (e.g. '20121109_120000').
        Overrides --dates.
    grid_dataset_name : str
        [grid] Zarr store name within folder (default: 'llc4320_grid.zarr').
    grid_output_filename : str
        [grid] Output NetCDF filename (default: 'llc4320_grid.nc').
    ice_mask : bool
        [snapshots] If True, mask ice-covered points (SIarea > 0) with
        NaN before writing NetCDF.  Default: False.
    ice_mask_dataset_name : str
        [snapshots] Zarr store containing SIarea (default: 'icearea.zarr').
        Only used when *ice_mask* is True.
    """
    if output_dir is None:
        p = argparse.ArgumentParser(
            description=(
                "Convert S3 Zarr global LLC4320 stores to NetCDF.\n\n"
                "  --mode snapshots  Convert per-timestep snapshot data to per-file NetCDF.\n"
                "  --mode grid       Convert the static grid Zarr to a single NetCDF."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        p.add_argument('--mode', choices=['snapshots', 'grid'], default='snapshots')
        p.add_argument('--s3-endpoint', required=True)
        p.add_argument('--bucket', required=True)
        p.add_argument('--folder', required=True)
        p.add_argument('--output-dir', required=True)
        p.add_argument('--run-id')
        p.add_argument('--dataset-name', required=True)
        p.add_argument('--dates', nargs='+', metavar='YYYY-MM-DD HH:MM:SS',
                       help="Model dates to convert.  Each is mapped to a date_prefix.")
        p.add_argument('--date-prefix',
                       help="Explicit date_prefix (e.g. '20121109_120000'). Overrides --dates.")
        p.add_argument('--output-filename')
        p.add_argument('--channel', nargs='+', metavar='NAME', dest='channels')
        p.add_argument('--channels', nargs='+', metavar='NAME')
        p.add_argument('--grid-dataset-name', default='llc4320_grid.zarr')
        p.add_argument('--grid-output-filename', default='llc4320_grid.nc')
        p.add_argument('--ice-mask', action='store_true', default=False,
                       help=("[snapshots] Mask ice-covered points (SIarea > 0) "
                             "with NaN before writing NetCDF."))
        p.add_argument('--ice-mask-dataset-name', default='icearea.zarr',
                       help=("[snapshots] Zarr store containing SIarea "
                             "(default: icearea.zarr)."))
        args = p.parse_args()
        if args.mode == 'snapshots' and not args.run_id:
            p.error("--run-id is required when --mode snapshots")
        output_dir              = args.output_dir
        output_filename         = args.output_filename
        mode                    = args.mode
        s3_endpoint             = args.s3_endpoint
        bucket                  = args.bucket
        folder                  = args.folder
        run_id                  = args.run_id
        dataset_name            = args.dataset_name
        dates                   = args.dates
        channels                = args.channels or getattr(args, 'channel', None)
        date_prefix             = args.date_prefix
        grid_dataset_name       = args.grid_dataset_name
        grid_output_filename    = args.grid_output_filename
        ice_mask                = args.ice_mask
        ice_mask_dataset_name   = args.ice_mask_dataset_name

    if mode == 'grid':
        grid_zarr_to_netcdf(
            s3_endpoint=s3_endpoint,
            bucket=bucket,
            folder=folder,
            grid_dataset_name=grid_dataset_name,
            output_dir=output_dir,
            output_filename=grid_output_filename,
        )

    else:  # snapshots
        if not run_id:
            raise ValueError("run_id is required when mode='snapshots'")

        _, fs_synch = create_s3_filesystems(s3_endpoint)

        # Resolve which date_prefixes to convert.
        if date_prefix is not None:
            # Explicit single date_prefix.
            date_prefixes = [date_prefix]
        elif dates is not None:
            # Convert date strings to date_prefix strings.
            date_prefixes = [_date_to_prefix(d) for d in dates]
        else:
            # Auto-discover all date_prefixes for this run_id.
            date_prefixes = _discover_date_prefixes(
                bucket, folder, run_id, dataset_name, fs_synch)
            if not date_prefixes:
                raise ValueError(
                    f"No date_prefix subdirectories found for "
                    f"run_id={run_id!r}, dataset_name={dataset_name!r} "
                    f"under s3://{bucket}/{folder}/."
                )

        if output_filename is not None and len(date_prefixes) > 1:
            raise ValueError(
                f"--output-filename can only be used when converting a single "
                f"date, but {len(date_prefixes)} dates were requested."
            )

        logging.info(
            f"Converting {len(date_prefixes)} date_prefix(es) for "
            f"{dataset_name} (run_id={run_id})"
        )

        # Resolve ice mask: pass the dataset name when enabled, None when not.
        _ice_mask_ds = ice_mask_dataset_name if ice_mask else None

        for dp in date_prefixes:
            zarr_to_netcdf(
                s3_endpoint=s3_endpoint,
                bucket=bucket,
                folder=folder,
                run_id=run_id,
                dataset_name=dataset_name,
                output_dir=output_dir,
                date_prefix=dp,
                output_filename=output_filename,
                channels=channels,
                fs=fs_synch,
                ice_mask_dataset_name=_ice_mask_ds,
            )


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=(
            "Convert S3 Zarr global LLC4320 stores to NetCDF.\n\n"
            "  --mode snapshots  Convert per-timestep snapshot data to per-file NetCDF.\n"
            "  --mode grid       Convert the static grid Zarr to a single NetCDF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    p.add_argument(
        '--mode', choices=['snapshots', 'grid'], default='snapshots',
        help="Conversion mode (default: snapshots).",
    )

    # Shared S3 args
    p.add_argument('--s3-endpoint', required=True,
                   help="NRP S3 endpoint, e.g. https://s3-west.nrp-nautilus.io")
    p.add_argument('--bucket', required=True,
                   help="S3 bucket name, e.g. dbof")
    p.add_argument('--folder', required=True,
                   help="S3 folder, e.g. surface_fields or depth_fields")

    # Output
    p.add_argument('--output-dir', required=True,
                   help="Local directory to write NetCDF file(s) into")

    # --- snapshots-mode args ---
    p.add_argument('--run-id',
                   help="[snapshots] run_id used when writing the Zarr store")
    p.add_argument('--dataset-name', required=True,
                   help="[snapshots] dataset_name (e.g. stratification.zarr)")

    p.add_argument('--dates', nargs='+', metavar='DATE',
                   help=("[snapshots] Model dates in 'YYYY-MM-DD HH:MM:SS' format. "
                         "Each is mapped to a date_prefix subdirectory."))
    p.add_argument('--date-prefix',
                   help=("[snapshots] Explicit date_prefix subdirectory under run_id "
                         "(e.g. '20121109_120000'). Overrides --dates."))

    p.add_argument('--output-filename',
                   help=("[snapshots] Override the auto-generated output filename "
                         "(e.g. 'N2_sfc.nc'). "
                         "Only valid when converting a single date."))
    p.add_argument('--channels', nargs='+', metavar='NAME',
                   help=("[snapshots] Save only the named channel(s) "
                         "(e.g. --channels N2_sfc). Default: all channels."))

    # --- grid-mode args ---
    p.add_argument('--grid-dataset-name', default='llc4320_grid.zarr',
                   help="[grid] Zarr store name in --folder (default: llc4320_grid.zarr)")
    p.add_argument('--grid-output-filename', default='llc4320_grid.nc',
                   help="[grid] Output NetCDF filename (default: llc4320_grid.nc)")

    # --- ice mask ---
    p.add_argument('--ice-mask', action='store_true', default=False,
                   help=("[snapshots] Mask ice-covered points (SIarea > 0) "
                         "with NaN before writing NetCDF."))
    p.add_argument('--ice-mask-dataset-name', default='icearea.zarr',
                   help=("[snapshots] Zarr store containing SIarea "
                         "(default: icearea.zarr)."))

    args = p.parse_args()

    if args.mode == 'snapshots' and not args.run_id:
        p.error("--run-id is required when --mode snapshots")

    main(
        args.output_dir,
        output_filename=args.output_filename,
        mode=args.mode,
        s3_endpoint=args.s3_endpoint,
        bucket=args.bucket,
        folder=args.folder,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        dates=args.dates,
        channels=args.channels,
        grid_dataset_name=args.grid_dataset_name,
        grid_output_filename=args.grid_output_filename,
        date_prefix=args.date_prefix,
        ice_mask=args.ice_mask,
        ice_mask_dataset_name=args.ice_mask_dataset_name,
    )
