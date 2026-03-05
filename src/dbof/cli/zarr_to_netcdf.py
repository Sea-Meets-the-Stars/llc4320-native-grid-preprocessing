"""
zarr_to_netcdf.py
-----------------
Two modes (selected via --mode):

  snapshots (default)
  -------------------
  Reads global LLC4320 snapshots from the S3 Zarr store written by
  generate_fronts_global.py and writes one NetCDF file per timestep to a
  local directory on the HPC machine.

  Output file naming:
    <output_dir>/<run_id>_it<iteration:07d>.nc
    e.g.  year_1xglobal_20260226_043824_it0184320.nc

  Each file is a self-describing xr.Dataset with:
    - One data variable per channel (dims: y, x)
    - Global attributes: llc4320_iteration, run_id, channel_names, etc.

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
  zarr-to-netcdf \\
      --mode snapshots \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data \\
      --run-id year_1xglobal_20260226_043824 \\
      --output-dir /scratch/llc4320_netcdf/
    
      #all channels
  zarr-to-netcdf \
    --mode snapshots \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --run-id year_1xglobal_20260226_043824 \
    --dates 09112012-12:00:00 \
    --output-dir /mnt/tank/Oceanography/data/OGCM/LLC/Fronts/derived/DBOF_v1_test \
    --output-filename LLC4320_2012-11-09T12_00_00_props.nc

    #single channel
  zarr-to-netcdf \
    --mode snapshots \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --run-id year_1xglobal_20260226_043824 \
    --dates 09112012-12:00:00 \
    --channel log_gradb \
    --output-dir /mnt/tank/Oceanography/data/OGCM/LLC/Fronts/derived/DBOF_v1_test \
    --output-filename LLC4320_2012-11-09T12_00_00_Divb2.nc

  Optional: convert only specific timesteps:
      --iterations 184320 328320
      --dates 01012012-00:00:00 01042012-00:00:00
      --indices 0 2

Usage — grid
------------
  zarr-to-netcdf \\
      --mode grid \\
      --s3-endpoint https://s3-west.nrp-nautilus.io \\
      --bucket dbof \\
      --folder native_grid_dbof_training_data \\
      --grid-dataset-name llc4320_grid.zarr \\
      --output-dir /scratch/llc4320_netcdf/

zarr-to-netcdf \
    --mode grid \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --output-dir /mnt/tank/Oceanography/data/OGCM/LLC/Fronts/derived/DBOF_v1_test \
    --output-filename LLC4320_grid.nc
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from dbof.io.filesystems import create_s3_filesystems
import dbof.dataset_creation.zarr_dataset_global as zarr_dataset_global
import dbof.dataset_creation.zarr_grid_global as zarr_grid_global

# LLC4320 calendar constants (same as generate_fronts_global.py)
LLC4320_START_DATE    = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS = 25
DATE_FMT              = '%d%m%Y-%H:%M:%S'


def _date_to_iteration(date_str: str) -> int:
    """Convert 'DDMMYYYY-HH:MM:SS' → LLC4320 iteration number."""
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start "
            f"({LLC4320_START_DATE.date()}). "
            "Expected format: DDMMYYYY-HH:MM:SS, e.g. '01012012-00:00:00'."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def _iteration_to_datetime(iteration: int) -> datetime:
    """Convert LLC4320 iteration number → UTC datetime."""
    import datetime as _dt
    return LLC4320_START_DATE + _dt.timedelta(
        seconds=int(iteration) * LLC4320_TIMESTEP_SECS
    )


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
    target_indices: list = None,    # list of t-axis indices, or None for all
    output_filename: str = None,    # override auto-generated name (single timestep)
    channels: list = None,          # subset of channel names to save; None = all
) -> None:
    """
    Convert a GlobalZarrDataset on S3 to per-timestep NetCDF files locally.

    Parameters
    ----------
    s3_endpoint : str
        NRP S3 endpoint, e.g. 'https://s3-west.nrp-nautilus.io'.
    bucket : str
        S3 bucket name, e.g. 'dbof'.
    folder : str
        S3 folder path, e.g. 'native_grid_dbof_training_data'.
    run_id : str
        The run_id used when writing the Zarr store.
    dataset_name : str
        The dataset_name used when writing the Zarr store
        (e.g. 'dataset_creation.zarr').
    output_dir : str
        Local directory where NetCDF files will be written.
        Created if it does not exist.
    target_indices : list of int, optional
        Which t-axis indices to convert. If None, all timesteps are converted.
    output_filename : str, optional
        Fixed output filename (stem + extension, e.g. 'myfile.nc').
        Only valid when exactly one timestep is being written; raises an error
        if multiple timesteps are requested with a single filename.
    channels : list of str, optional
        Subset of channel names to include in the output file.
        If None, all channels are written.  Unknown names raise ValueError.
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
    store_path = f"s3://{bucket.strip('/')}/{folder.strip('/')}/{run_id}/{dataset_name}"
    logging.info(f"Opening snapshot Zarr store: {store_path}")
    _, fs_synch = create_s3_filesystems(s3_endpoint)
    reader = zarr_dataset_global.GlobalZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name=dataset_name,
        fs=fs_synch,
    )

    n_total = len(reader)
    all_channel_names = reader.channel_names
    H, W = reader.rectangular_shape
    logging.info(
        f"Store contains {n_total} timestep(s), {len(all_channel_names)} channel(s): "
        f"{all_channel_names}  |  spatial shape: {H} × {W}"
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

    if target_indices is None:
        target_indices = list(range(n_total))
    else:
        bad = [i for i in target_indices if i < 0 or i >= n_total]
        if bad:
            raise ValueError(
                f"Requested t-indices {bad} are out of range [0, {n_total - 1}]."
            )

    if output_filename is not None and len(target_indices) > 1:
        raise ValueError(
            f"--output-filename can only be used when converting a single timestep, "
            f"but {len(target_indices)} timesteps were requested."
        )

    # ------------------------------------------------------------------
    # 2. Convert each selected timestep
    # ------------------------------------------------------------------
    for t in target_indices:
        iteration = int(reader.time[t])

        logging.info(
            f"[{t + 1}/{len(target_indices)}] "
            f"t={t}, iteration={iteration} "
            f"({_iteration_to_datetime(iteration).strftime('%Y-%m-%d %H:%M UTC')})"
        )

        # Load only the required channels one at a time to limit peak RAM
        n_y, n_x = H, W
        y_coord = np.arange(n_y, dtype=np.int32)
        x_coord = np.arange(n_x, dtype=np.int32)

        data_vars = {
            ch: xr.DataArray(
                reader.get_channel_snapshot(t, ch).astype(np.float32),
                dims=['y', 'x'],
                coords={'y': y_coord, 'x': x_coord},
            )
            for ch in channel_names
        }

        ds = xr.Dataset(
            data_vars,
            attrs={
                'llc4320_iteration': iteration,
                'model_time_utc':    _iteration_to_datetime(iteration).isoformat(),
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
            nc_filename = out_path / f"{run_id}_it{iteration:07d}.nc"
        ds.to_netcdf(nc_filename)
        logging.info(f"  → written: {nc_filename}")

    logging.info(f"Done. {len(target_indices)} file(s) written to {out_path}")


# ---------------------------------------------------------------------------
# Mode: grid
# ---------------------------------------------------------------------------

def grid_zarr_to_netcdf(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    grid_dataset_name: str,
    output_dir: str,
    output_filename: str = "llc4320_grid.nc",
) -> None:
    """
    Convert the LLC4320 static grid Zarr store (written by generate_grid_global.py)
    to a single NetCDF file saved locally on HPC.

    The grid Zarr is stored at:
        s3://{bucket}/{folder}/{grid_dataset_name}
    (note: no run_id in the path — the grid is shared across all runs).

    Parameters
    ----------
    s3_endpoint : str
        NRP S3 endpoint, e.g. 'https://s3-west.nrp-nautilus.io'.
    bucket : str
        S3 bucket name, e.g. 'dbof'.
    folder : str
        S3 folder, e.g. 'native_grid_dbof_training_data'.
    grid_dataset_name : str
        Name of the grid Zarr store within *folder*
        (e.g. 'llc4320_grid.zarr').
    output_dir : str
        Local directory where the NetCDF file will be written.
        Created if it does not exist.
    output_filename : str
        Name of the output NetCDF file (default: 'llc4320_grid.nc').
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    store_path = (
        f"s3://{bucket.strip('/')}/{folder.strip('/')}/{grid_dataset_name}"
    )
    logging.info(f"Opening grid Zarr store: {store_path}")

    _, fs_synch = create_s3_filesystems(s3_endpoint)
    grid_reader = zarr_grid_global.GlobalGridZarrReader(
        bucket=bucket,
        folder=folder,
        dataset_name=grid_dataset_name,
        fs=fs_synch,
    )

    H, W = grid_reader.grid_shape
    logging.info(
        f"Grid shape: {H} × {W}  |  variables: {grid_reader.variables}"
    )

    # ------------------------------------------------------------------
    # Load all variables into an xarray Dataset (dims: y, x)
    # ------------------------------------------------------------------
    y_coord = np.arange(H, dtype=np.int32)
    x_coord = np.arange(W, dtype=np.int32)

    data_vars = {}
    for vname in grid_reader.variables:
        arr  = grid_reader[vname]               # (H, W) numpy float32
        attrs = dict(grid_reader.root[vname].attrs)
        data_vars[vname] = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': y_coord, 'x': x_coord},
            attrs=attrs,
        )

    ds = xr.Dataset(
        data_vars,
        coords={'y': y_coord, 'x': x_coord},
        attrs={
            'grid_shape':  f'{H} x {W}',
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

    # Promote XC/YC to 2D coordinate variables so downstream tools can use them
    if 'XC' in ds and 'YC' in ds:
        ds = ds.assign_coords(
            lon=ds['XC'],
            lat=ds['YC'],
        )

    nc_filename = out_path / output_filename
    ds.to_netcdf(nc_filename)
    size_gb = nc_filename.stat().st_size / 1e9
    logging.info(f"Grid written: {nc_filename}  ({size_gb:.2f} GB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    output_dir: str,
    output_filename: str = None,
    mode: str = 'snapshots',
    s3_endpoint: str = 'https://s3-west.nrp-nautilus.io',
    bucket: str = 'dbof',
    folder: str = None,
    run_id: str = None,
    dataset_name: str = 'dataset_creation.zarr',
    indices: list = None,
    iterations: list = None,
    dates: list = None,
    channel: list = None,
    grid_dataset_name: str = 'llc4320_grid.zarr',
    grid_output_filename: str = 'llc4320_grid.nc',
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
    indices : list of int, optional
        [snapshots] t-axis indices to convert (0-based). Default: all.
    iterations : list of int, optional
        [snapshots] LLC4320 iteration numbers to convert.
    dates : list of str, optional
        [snapshots] Model dates in 'DDMMYYYY-HH:MM:SS' format.
    output_filename : str, optional
        [snapshots] Override auto-generated output filename (single timestep only).
    channel : list of str, optional
        [snapshots] Channel name(s) to save. Default: all.
    grid_dataset_name : str
        [grid] Zarr store name within folder (default: 'llc4320_grid.zarr').
    grid_output_filename : str
        [grid] Output NetCDF filename (default: 'llc4320_grid.nc').
    """
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

        target_indices = None

        if iterations is not None or dates is not None:
            _, fs_synch = create_s3_filesystems(s3_endpoint)
            reader = zarr_dataset_global.GlobalZarrDatasetReader(
                bucket=bucket,
                folder=folder,
                run_id=run_id,
                dataset_name=dataset_name,
                fs=fs_synch,
            )
            if dates is not None:
                iters = [_date_to_iteration(d) for d in dates]
            else:
                iters = iterations
            target_indices = [reader.iteration_to_index(it) for it in iters]
        elif indices is not None:
            target_indices = indices

        zarr_to_netcdf(
            s3_endpoint=s3_endpoint,
            bucket=bucket,
            folder=folder,
            run_id=run_id,
            dataset_name=dataset_name,
            output_dir=output_dir,
            target_indices=target_indices,
            output_filename=output_filename,
            channels=channel,
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
                   help="S3 folder, e.g. native_grid_dbof_training_data")

    # Output
    p.add_argument('--output-dir', required=True,
                   help="Local directory to write NetCDF file(s) into")

    # --- snapshots-mode args ---
    p.add_argument('--run-id',
                   help="[snapshots] run_id used when writing the Zarr store")
    p.add_argument('--dataset-name', default='dataset_creation.zarr',
                   help="[snapshots] dataset_name (default: dataset_creation.zarr)")

    sel = p.add_mutually_exclusive_group()
    sel.add_argument('--indices', nargs='+', type=int, metavar='T',
                     help="[snapshots] t-axis indices to convert (0-based). Default: all.")
    sel.add_argument('--iterations', nargs='+', type=int, metavar='IT',
                     help="[snapshots] LLC4320 iteration numbers to convert.")
    sel.add_argument('--dates', nargs='+', metavar='DDMMYYYY-HH:MM:SS',
                     help="[snapshots] Model dates to convert, e.g. 01012012-00:00:00")

    p.add_argument('--output-filename',
                   help=("[snapshots] Override the auto-generated output filename "
                         "(e.g. 'LLC4320_2012-01-01_props.nc'). "
                         "Only valid when converting a single timestep."))
    p.add_argument('--channel', nargs='+', metavar='NAME',
                   help=("[snapshots] Save only the named channel(s) "
                         "(e.g. --channel log_gradb). Default: all channels."))

    # --- grid-mode args ---
    p.add_argument('--grid-dataset-name', default='llc4320_grid.zarr',
                   help="[grid] Zarr store name in --folder (default: llc4320_grid.zarr)")
    p.add_argument('--grid-output-filename', default='llc4320_grid.nc',
                   help="[grid] Output NetCDF filename (default: llc4320_grid.nc)")

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
        indices=args.indices,
        iterations=args.iterations,
        dates=args.dates,
        channel=args.channel,
        grid_dataset_name=args.grid_dataset_name,
        grid_output_filename=args.grid_output_filename,
    )
