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
# LLC_NATIVE_GRID_DEG = 1 / 48   # only needed if using resample_to_latlon; not used here

# url of our raw data - this may need to be an input in the future
endpoint_url = 'https://mghp.osn.xsede.org'

# feature_channels_lazy = ["Eta", "Salt", "Theta", "U", "V", "W"] #,
# feature_channels_computed = []

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

    # --- Compact each channel from LLC faces (13, j, i) → 2D (compact_h, compact_w) ---
    #
    # ecco.llc_tiles_to_compact() stitches the 13 LLC faces into a single coherent
    # 2D array. This is NOT interpolation — values are pixel-shifted and some faces
    # are rotated to tile correctly, but we stay on the native LLC grid.
    # Input shape:  (13, 4320, 4320)  — one value per native grid cell per face
    # Output shape: (12960, 17280)  — 3×4320 × 4×4320, same values rearranged into 2D
    #
    # .values calls here are intentional — we are in a sequential Python for-loop,
    # NOT inside a dask.delayed task. This is the correct place to materialise data.

    channel_arrays = []

    for ch in model_feature_channels:
        logging.info(f"Compacting channel {ch} to LLC compact format")
        field = ecco.llc_tiles_to_compact(ds_merge[ch].values)
        channel_arrays.append(field)   # shape: (compact_h, compact_w)

    # Computed features (e.g. relative_vorticity) — computed on native face grid,
    # compacted the same way as model channels.
    for ch in computed_feature_channels:
        logging.info(f"Compacting computed channel {ch}")
        # calculated_fields values are xarray DataArrays; materialise before compacting
        field = ecco.llc_tiles_to_compact(calculated_fields[ch].values)
        channel_arrays.append(field)

    # log_gradb is already numpy (materialised above via the protected .values call)
    log_gradb_field = ecco.llc_tiles_to_compact(log_gradb_np)
    channel_arrays.append(log_gradb_field)

    # stack channels into single array (C, n_lat, n_lon) for zarr dataset_creation
    data = np.stack(channel_arrays, axis=0)

    # write to zarr ds 
    zarr_ds.write_snapshot(data,it)

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
    # use_halo=False: global output doesn't need the coastal buffer used by the cutout pipeline.
    ds_grid, land_mask = set_up_grid_data_and_masks(cfg, use_halo=False)
    grid = xgcm.Grid(ds_grid, periodic=False)

    # Determine compact output shape BEFORE constructing the Zarr writer.
    # We do a dry-run with a zero array to find out what shape llc_tiles_to_compact
    # produces for this grid. This runs once at startup and is cheap.
    logging.info("Computing compact output shape via dry-run ecco compact...")
    dummy_compact = ecco.llc_tiles_to_compact(
        np.zeros_like(ds_grid.XC.values, dtype=np.float32)
    )
    compact_shape = dummy_compact.shape   # (compact_h, compact_w)
    logging.info(f"LLC compact output shape: {compact_shape}")

    # Construct GlobalZarrDataset.
    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        channel_names=model_feature_channels + computed_feature_channels + ["log_gradb"],
        compact_shape=compact_shape,
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