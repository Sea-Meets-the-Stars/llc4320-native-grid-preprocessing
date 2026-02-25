# FIX [1]: Removed stale imports no longer needed in the global pipeline.
# - metadata: no Parquet metadata writer in this flow (time/channel info lives in Zarr)
# - dask_pipeline: no patch creation
# - relative_vorticity direct import: already accessible via calculate_additional_fields
# TODO METAdata has extra id column unused. Get rid of it.

# stdlib
import sys
import logging
from pathlib import Path

# numerical / compute
import numpy as np
import xgcm
import zarr

# grid
import ecco_v4_py as ecco

# distributed / IO
from dask.distributed import Client

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields

import dbof.llc4320_ingestion.get_raw_data as get_raw_data

import dbof.dataset_creation.zarr_dataset_global as zarr_dataset

import dbof.dataset_creation.config as config

# Constants --------------------------
# NOTE these are constants for the LLC 4320 model. If we look to support other models in the future
# this will need to be updated or configurable.
TS_PER_HOUR = 144 # model cadence: 25 s → 144 steps/hr
MAX_ITER = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368
LLC_FACES = range(13)
LLC_NATIVE_GRID_DEG = 1 / 48   # ≈ 0.0208°

# url of our raw data - this may need to be an input in the future
endpoint_url = 'https://mghp.osn.xsede.org'


def generate_logging(cfg: config.JobConfig):
    log_root = Path(cfg.run.log_dir).expanduser().resolve()
    run_dir = log_root / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log_file = run_dir / "generate_front_training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def calculate_iterations_for_llc(cfg: config.JobConfig):
    # Calculate iterations based on input
    iter_step = cfg.data.sampling_step * TS_PER_HOUR  # iteration Δ between samples
    start_iter = FIRST_WIND_RECORD_OFFSET + cfg.data.start_record * TS_PER_HOUR

    if (cfg.data.timestep_hours is None):
        end_iter = MAX_ITER  # to the end of data
    else:
        end_iter = start_iter + cfg.data.timestep_hours * TS_PER_HOUR

    return np.arange(start_iter, end_iter, iter_step)


def set_up_grid_data_and_masks(cfg: config.JobConfig):
    logging.info("Fetching grid file")
    co = get_raw_data.get_remote_gridfile(cfg.data.endpoint_url)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    logging.info("Calculating land mask")
    land_mask = native_grid_masks.generate_static_land_mask_for_sampling(ds_grid, cfg.output.target_km_res)

    return ds_grid, land_mask



def process_time_snapshot(
    cfg: config.JobConfig,
    zarr_ds,
    ds_merge,
    grid,
    land_mask,
    model_feature_channels,
    computed_feature_channels,
    it,                         
):
    # NOTE The ordering of the following steps matters.

    # --- Ice mask (optional) ---
    logging.info(f"Calculating ice mask")
    ice_mask = ~(ds_merge.Theta <= 0.0)
    ice_mask_np = ice_mask.values
    merged_mask = ice_mask_np & land_mask

    # --- Computed fields ---
    calculated_fields = {}

    log_gradb = calculate_additional_fields.log_grad_b(ds_merge, grid)

    if "relative_vorticity" in computed_feature_channels:
        rv = calculate_additional_fields.relative_vorticity(ds_merge, grid)
        calculated_fields["relative_vorticity"] = rv

    # --- Staggered → tracer interpolation ---
    # U lives on X cell faces, V on Y cell faces. Interpolating to tracer points
    # (cell centers) means all channels share the same XC/YC coordinates, which
    # is required before passing to ecco.resample_to_latlon.
    # NOTE: W is on vertical cell faces but shares the same horizontal position as
    # tracers (XC/YC), so no horizontal interpolation is needed for W.
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    # Materialise log_gradb into memory now. This breaks a problematic Dask graph
    # split caused by xmitgcm internals. Do not remove the .values call here.
    log_gradb_np = log_gradb.values  # protected line do not modify
    calculated_fields["log_gradb_np"] = log_gradb_np

    # --- Resample each channel to regular lat/lon grid ---
    #
    # ecco.resample_to_latlon takes:
    #   source_lons, source_lats  : native grid coordinate arrays (face, j, i)
    #   source_field              : data array of same shape, as numpy
    #   new_grid_delta_lat        : output lat spacing in degrees
    #   new_grid_delta_lon        : output lon spacing in degrees
    #
    # It returns:
    #   new_grid_lon_centers : 1-D array of output lon bin centers
    #   new_grid_lat_centers : 1-D array of output lat bin centers
    #   resampled_field      : 2-D array of shape (n_lat, n_lon)
    #
    # .values calls here are intentional — we are in a sequential Python for-loop,
    # NOT inside a dask.delayed task. This is the correct place to materialise data.

    XC = ds_merge.XC.values
    YC = ds_merge.YC.values

    channel_arrays = []

    for ch in model_feature_channels:
        _, _, field = ecco.resample_to_latlon(
            XC, YC,
            ds_merge[ch].values,
            LLC_NATIVE_GRID_DEG, LLC_NATIVE_GRID_DEG,
            fill_value=np.nan,
        )
        channel_arrays.append(field)   # shape: (n_lat, n_lon)

    for ch in computed_feature_channels:
        # computed fields are already numpy arrays in calculated_fields
        _, _, field = ecco.resample_to_latlon(
            XC, YC,
            calculated_fields[ch],
            LLC_NATIVE_GRID_DEG, LLC_NATIVE_GRID_DEG,
            fill_value=np.nan,
        )
        channel_arrays.append(field)

    # log_gradb is already numpy (materialised above)
    _, _, log_gradb_field = ecco.resample_to_latlon(
        XC, YC,
        log_gradb_np,
        LLC_NATIVE_GRID_DEG, LLC_NATIVE_GRID_DEG,
        fill_value=np.nan,
    )
    channel_arrays.append(log_gradb_field)

    # Stack into (C, n_lat, n_lon) and write.
    data = np.stack(channel_arrays, axis=0)   # (C, n_lat, n_lon)
    zarr_ds.write_snapshot(data, it)

    logging.info(f"Snapshot written for iteration {it}")

    # --- Cleanup ---
    ds_merge = None
    del ds_merge

    grid = None
    del grid

    merged_mask = None
    del merged_mask


def main():
    """
    Entry point for global LLC4320 dataset generation.

    Orchestrates argument parsing, Dask setup, filesystem initialization,
    and iteration over time snapshots.
    """

    cli = config.parse_args()
    cfg = config.load_config(cli.config)

    # override run_id if passed in through cli
    if cli.run_id is not None:
        cfg = config.JobConfig(
            run=config.RunConfig(run_id=cli.run_id, log_dir=cfg.run.log_dir),
            data=cfg.data,
            sampling=cfg.sampling,
            output=cfg.output,
            features=cfg.features,
            runtime=cfg.runtime,
        )

    generate_logging(cfg)

    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    model_feature_channels = [c.strip() for c in cfg.features.model_data_feature_channels if c.strip()]
    computed_feature_channels = [c.strip() for c in cfg.features.compute_features_channels if c.strip()]

    # Set concurrency for zarr ds writes
    zarr.config.set({'async.concurrency': cfg.runtime.zarr_async_concurrency})

    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores
    logging.info(f"Dask Client {dask_client}")

    iter_range = calculate_iterations_for_llc(cfg)
    logging.info(f"Processing: {iter_range} time snapshots")

    # Set up filesystem
    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)

    # Get our grid and static masks once. These never change across timesteps.
    ds_grid, land_mask = set_up_grid_data_and_masks(cfg)
    grid = xgcm.Grid(ds_grid, periodic=False)

    # Determine output lat/lon grid BEFORE constructing the Zarr writer.
    #
    # We do one "dry run" through ecco.resample_to_latlon with a dummy zero array
    # to get the output bin centers (new_grid_lat, new_grid_lon). These define the
    # shape and coordinates of the output grid and must be passed to GlobalZarrDataset
    # at construction time so it can initialise its arrays correctly.
    #
    # This must come AFTER set_up_grid_data_and_masks so ds_grid is available,
    # and BEFORE constructing zarr_ds so the lat/lon arrays exist.
    logging.info("Computing output lat/lon grid via dry-run ecco resample...")
    new_grid_lon, new_grid_lat, _ = ecco.resample_to_latlon(
        ds_grid.XC.values,
        ds_grid.YC.values,
        np.zeros_like(ds_grid.XC.values, dtype=np.float32),
        LLC_NATIVE_GRID_DEG, LLC_NATIVE_GRID_DEG,
        fill_value=np.nan,
    )
    logging.info(f"Output grid: lat {new_grid_lat.shape}, lon {new_grid_lon.shape}")

    # Construct GlobalZarrDataset
    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        channel_names=model_feature_channels + computed_feature_channels + ["log_gradb"],
        lats=new_grid_lat,
        lons=new_grid_lon,
    )

    logging.info(f"Zarr dataset created.")

    for it in tqdm.tqdm(iter_range):
        # grab raw data for this iteration
        ds = get_raw_data.get_remote_llc_data(endpoint_url, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        logging.info(f"Data loaded for iteration: {it}")

        # now process this iteration of data
        process_time_snapshot(
            cfg,
            zarr_ds,
            ds_merge,
            grid,
            land_mask,
            model_feature_channels,
            computed_feature_channels,
            it,
        )

        ds_merge = None
        del ds_merge

        ds = None
        del ds


if __name__ == "__main__":
    main()
