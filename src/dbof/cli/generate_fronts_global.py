# TODO METAdata has extra id column unused. Get rid of it.

# stdlib
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# numerical / compute
import numpy as np
import xgcm
import zarr

# grid / face-to-latlon stitching
import xarray as xr
from xmitgcm.llcreader import llcmodel

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
# LLC_NATIVE_GRID_DEG = 1 / 48   # only needed if using resample_to_latlon; not used here

# LLC4320 calendar reference.
# Iteration 0 corresponds to 2011-09-13 00:00:00 UTC; each step is 25 seconds.
# Used to convert human-readable dates ('DDMMYYYY-HH:MM:SS') → iteration numbers.
LLC4320_START_DATE      = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS   = 25          # seconds per model step
DATE_FMT                = '%d%m%Y-%H:%M:%S'   # expected format: DDMMYYYY-HH:MM:SS

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

def _date_to_iteration(date_str: str) -> int:
    """
    Convert a date string in 'DDMMYYYY-HH:MM:SS' format to an LLC4320 iteration number.

    The LLC4320 model starts at 2011-09-13 00:00:00 UTC (iteration 0) with a
    25-second timestep. The returned iteration is rounded to the nearest step.

    Examples
    --------
    _date_to_iteration('13092011-00:00:00')  ->  0
    _date_to_iteration('01012012-00:00:00')  ->  ~1,011,456
    """
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start ({LLC4320_START_DATE.date()}). "
            "Check your DDMMYYYY-HH:MM:SS format — e.g. '13092011-00:00:00'."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def calculate_iterations_for_llc(cfg: config.JobConfig):
    """
    Return the list of LLC4320 iteration numbers to process.

    Three modes, in priority order:

    1. Date list  (cfg.data.date_iterations is set in YAML, e.g.
       ``date_iterations: ['01012012-00:00:00', '01042012-00:00:00']``):
       Each date string is converted to the nearest LLC4320 iteration number
       using the model's 25-second timestep and start date (2011-09-13 00:00 UTC).

    2. Range mode (default, backwards-compatible):
       A uniformly-spaced range is derived from start_record, sampling_step, and
       timestep_hours.  If timestep_hours is None the range runs to MAX_ITER.
    """
    if cfg.data.date_iterations is not None:
        iterations = [_date_to_iteration(d) for d in cfg.data.date_iterations]
        logging.info(
            f"Using date-derived iteration list: "
            + ", ".join(f"'{d}' → {it}" for d, it in zip(cfg.data.date_iterations, iterations))
        )
        return np.array(iterations, dtype=int)

    # Range mode: convert hours → model iteration numbers
    iter_step  = cfg.data.sampling_step * TS_PER_HOUR
    start_iter = FIRST_WIND_RECORD_OFFSET + cfg.data.start_record * TS_PER_HOUR
    end_iter   = MAX_ITER if cfg.data.timestep_hours is None \
                 else start_iter + cfg.data.timestep_hours * TS_PER_HOUR

    return np.arange(start_iter, end_iter, iter_step)

def set_up_grid_data_and_masks(cfg: config.JobConfig, use_halo: bool = False):
    """
    Load the LLC4320 grid and compute a static land mask.

    Parameters
    ----------
    cfg : JobConfig
    use_halo : bool, default False
        If True, expand the land mask outward by a halo of cfg.output.target_km_res km
        using the fast-marching method. This is useful for the cutout pipeline where
        patch centres must be far from land, but is generally not needed for global output
        (it would unnecessarily NaN out coastal ocean in the compact image).
        If False, the raw land mask (hFacC == 0) is used with no expansion.
    """
    logging.info("Fetching grid file")
    co = get_raw_data.get_remote_gridfile(cfg.data.endpoint_url)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    logging.info(f"Calculating land mask (use_halo={use_halo})")
    if use_halo:
        land_mask = native_grid_masks.generate_static_land_mask_for_sampling(
            ds_grid, cfg.output.target_km_res
        )
    else:
        land_mask = (ds_grid.hFacC == 0).values  # raw land mask, no coastal buffer

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
    # NOTE The ordering of the following steps matters

    # --- Calculate Ice Mask ---
    logging.info(f"Calculating ice mask")
    ice_mask = ~(ds_merge.Theta <= 0.0)
    ice_mask_np = ice_mask.values
    merged_mask = ice_mask_np & land_mask

    # --- Calculated Fields ---
    calculated_fields = {}

    # This must be included so long as we are sampling using it.
    # If we support additional sampling methods in the future, this becomes optional.
    log_gradb = calculate_additional_fields.log_grad_b(ds_merge, grid)

    if "relative_vorticity" in computed_feature_channels:
        relative_vorticity = calculate_additional_fields.relative_vorticity(ds_merge, grid)
        calculated_fields["relative_vorticity"] = relative_vorticity

    # Move non tracer values to tracer points. This allows us to stack images for our final patches.
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    '''
    Here we compute the calculated gradients into memory before creating our patches. 
    While this is arguably inefficient if we do not do this our Dask graph splits and we will run into difficult errors
    or warnings to fix. 
    The cause of this is either xmitgcm code calculating the gradients on the native grid or that we are using 
    the gradient in our sampling logic. I believe it is the first but I am not sure yet. - Jake 
    '''

    # Materialize log_gradb into memory before we lose the ds_merge reference. 
    log_gradb_np = log_gradb.values #protected line do not modify
    calculated_fields["log_gradb_np"] = log_gradb_np

    # --- Convert from LLC faces (face, j, i) → rectangular lat/lon (lat, lon) ---
    #
    # xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon() stitches the 13 LLC
    # faces into a single coherent 2D rectangular image. This is NOT interpolation
    # — values are pixel-shifted and some faces are rotated to tile correctly, but
    # we remain on the native LLC grid.
    # Input:  xr.Dataset with (face, j, i) dimensions, shape (13, 4320, 4320) per var
    # Output: xr.Dataset with (lat, lon) dimensions, shape (12960, 17280) per var

    # log_gradb is already a numpy array (materialised above via the protected
    # .values call). Wrap it back as a DataArray so it can be passed to
    # faces_dataset_to_latlon alongside the other xarray variables.
    log_gradb_da = xr.DataArray(
        log_gradb_np,
        dims=ds_merge['Theta'].dims,
        coords=ds_merge['Theta'].coords,
        name='log_gradb',
    )

    # Assemble all channels into a single Dataset for a single conversion pass.
    all_channels_ds = xr.Dataset(
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
        | {'log_gradb': log_gradb_da}
    )

    # metric_vector_pairs tells faces_dataset_to_latlon to also rotate the
    # direction of vector fields (U, V) when Arctic-cap faces are transposed.
    # This is needed in addition to the staggered --> tracer interpolation above,
    # which only co-locates U/V on cell centres but does not correct directions.
    has_uv = ('U' in model_feature_channels and 'V' in model_feature_channels)
    metric_vector_pairs = [('U', 'V')] if has_uv else []

    logging.info("Converting from LLC faces to rectangular lat/lon...")
    ds_rect = llcmodel.faces_dataset_to_latlon(
        all_channels_ds,
        metric_vector_pairs=metric_vector_pairs,
    )

    # Extract channels in a consistent order and stack into (C, H, W).
    channel_order = model_feature_channels + computed_feature_channels + ['log_gradb']
    channel_arrays = [ds_rect[ch].values for ch in channel_order]
    data = np.stack(channel_arrays, axis=0)   # shape: (C, compact_h, compact_w)

    # write to zarr ds
    zarr_ds.write_snapshot(data, it)

    # for dask
    ds_merge = None
    del ds_merge

    grid = None
    del grid

    merged_mask = None
    del merged_mask


def main():
    """
    Entry point for native-grid LLC dataset_creation generation.

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
    zarr.config.set({'async.concurrency':  cfg.runtime.zarr_async_concurrency})

    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores
    logging.info(f"Dask Client {dask_client}")

    iter_range = calculate_iterations_for_llc(cfg)
    logging.info(f"Processing: {iter_range} time snapshots")

    # Set up meta and zarr data writers
    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)

    # Get our grid and static masks once ever. These never change.
    # use_halo=False for global 
    ds_grid, land_mask = set_up_grid_data_and_masks(cfg, use_halo=False)
    grid = xgcm.Grid(ds_grid, periodic=False)

    # Determine the rectangular output shape BEFORE constructing the Zarr writer.
    # Dry-run on a single dummy channel to find out what shape
    # faces_dataset_to_latlon produces for this grid. Runs once at startup.
    logging.info("Computing rectangular output shape via dry-run...")
    _dummy_ds = xr.Dataset({'_dummy': ds_grid.XC.astype(np.float32)})
    _dummy_rect = llcmodel.faces_dataset_to_latlon(_dummy_ds, metric_vector_pairs=[])
    rectangular_shape = tuple(_dummy_rect['_dummy'].shape)   # (12960, 17280)
    logging.info(f"LLC rectangular output shape: {rectangular_shape}")

    # Construct GlobalZarrDataset.
    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        channel_names=model_feature_channels + computed_feature_channels + ["log_gradb"],
        rectangular_shape=rectangular_shape,
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