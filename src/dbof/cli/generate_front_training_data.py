# TODO METAdata has extra id column unused. Get rid of it.

# stdlib
import sys
import logging
from pathlib import Path

# numerical / compute
import numpy as np
import xgcm
import zarr

# distributed / IO
from dask.distributed import Client

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
import dbof.preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling

import dbof.llc4320_ingestion.get_raw_data as get_raw_data

import dbof.dataset_creation.zarr_dataset as zarr_dataset

import dbof.dataset_creation.metadata as metadata
import dbof.dataset_creation.dask_pipeline as dask_pipeline
import dbof.dataset_creation.config as config


# Constants --------------------------
# NOTE these are constants for the LLC 4320 model. If we look to support other models in the future
# this will need to be updated or configurable.
TS_PER_HOUR = 144 # model cadence: 25 s → 144 steps/hr
MAX_ITER = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368
LLC_FACES = range(13)

# url of our raw data - this may need to be an input in the future
endpoint_url = 'https://mghp.osn.xsede.org'

# feature_channels_lazy = ["Eta", "Salt", "Theta", "U", "V", "W"] #,
# feature_channels_computed = ["log_gradb"]

metadata_cols = [
    "id",
    "dataset_index",
    "native_grid",
    "center_grid_face",
    "center_grid_j",
    "center_grid_i",
    "center_lat",
    "center_lon",
    "target_km_res",
    "real_km_w",
    "real_km_h",
    "pre_interp_res",
    "log_grad_b_2_center",
    "time_snapshot"
]

# def parse_args(p):
#     #data selection-----
#     p.add_argument("--sampling_step", required=True,
#                    help="Stride in timesteps", type=int)
#
#     p.add_argument("--start_record", default=1180, type=int,
#                    help="Starting record. Default is first record with wind forcing.")
#
#     p.add_argument("--timestep_hours", default=None, type=int,
#                    help="How many total hours to load between start iteration and end iteration."
#                         "If not given, the script will proces at provided sampling_step until the end of the data.")
#
#     #sampling options--------
#     p.add_argument("--bias_to_high_gradients", default=2, type=float,
#                    help="Bias to high B in sampled data")
#
#     p.add_argument("--sample_points_per_snapshot", default=100, type=int,
#                    help="How many points per snapshot to sample")
#
#     #s3 config----------
#     p.add_argument("--s3_endpoint", default="https://s3-west.nrp-nautilus.io",
#                    help="nrp s3 endpoint. Likely leave default.")
#
#     p.add_argument("--bucket", default="llc/",
#                    help="NRP s3 bucket to save data")
#
#     p.add_argument("--folder", default="native_grid_dbof_training_data/",
#                    help="NRP s3 bucket to save data")
#
#     p.add_argument("--run_id", required=True,
#                    help="Ensure run Id is unique inside s3://bucket/folder/ ")
#
#     #logging -------------
#     p.add_argument("--log_dir", default="./logs",
#         help="Directory where logs for this run will be written."
#     )
#
#     #return data-------
#     p.add_argument("--target_km_res", default=150, type=int,
#                    help="Target physical resolution in km. Default is 150.")
#
#     p.add_argument("--down_sample_res", default=64, type=int,
#                    help="Downsampling resolution of W and H in pixels. Default is 64.")
#
#     p.add_argument(
#         "--model_data_feature_channels",
#         type=str,
#         default="Eta,Salt,Theta,U,V,W",
#         help="Comma-separated list of lazy-loaded feature channels"
#     )
#
#     p.add_argument(
#         "--compute_features_channels",
#         type=str,
#         default="log_gradb",
#         help="Comma-separated list of computed feature channels"
#     )
#
#     args = p.parse_args()
#     return args

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

    logging.info("Calculating land and face masks")
    land_face_mask = native_grid_masks.generate_static_land_face_masks_for_sampling(ds_grid, cfg.output.target_km_res)

    return ds_grid, land_face_mask

def process_time_snapshot(cfg: config.JobConfig, metadata_writer, zarr_ds, ds_merge, grid, land_face_mask, non_computed_feature_channels):
    # NOTE The ordering of the following steps matters

    # Calculate Ice Mask
    logging.info(f"Calculating ice mask")
    ice_mask = ~(ds_merge.Theta <= 0.0)
    ice_mask_np = ice_mask.values
    merged_mask = ice_mask_np & land_face_mask

    # calculate gradients
    ds_merge, log_gradb = calculate_additional_fields.calculate_gradients(ds_merge, grid)

    logging.info(f"Sampling patch center points")
    # todo this should be updated to take in a numpy array since we compute loggradb later anyway
    indices = weighted_coordinate_sampling.weighted_sample_on_grid(cfg.sampling.sample_points_per_snapshot, cfg.sampling.bias_to_high_gradients,
                                                                   log_gradb, merged_mask)

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
    log_gradb_np = log_gradb.values #protected line do not modify

    # grow zarr ds to fit at most len of indices
    zarr_ds.grow_array(len(indices))

    #process data and write to s3
    images = dask_pipeline.run_patch_creation(zarr_ds, metadata_writer, cfg.output.down_sample_res,
                                                          indices, ds_merge, log_gradb_np, cfg.output.target_km_res,
                                                          non_computed_feature_channels, metadata_cols)

    # flush metada
    metadata_writer.close()

    # for dask
    ds_merge = None
    del ds_merge

    grid = None
    del grid

    merged_mask = None
    del merged_mask


def main():
    """
    Entry point for native-grid LLC patch dataset_creation generation.

    Orchestrates argument parsing, Dask setup, filesystem initialization,
    and iteration over time snapshots.
    """

    # Script arguments
    # p = argparse.ArgumentParser()
    # args = parse_args(p)

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

    non_computed_feature_channels = [c.strip() for c in cfg.features.model_data_feature_channels if c.strip()]
    feature_channels_computed = [c.strip() for c in cfg.features.compute_features_channels if c.strip()]

    # Set concurrency for zarr ds writes
    zarr.config.set({'async.concurrency':  cfg.runtime.zarr_async_concurrency})

    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores
    logging.info(f"Dask Client {dask_client}")

    iter_range = calculate_iterations_for_llc(cfg)
    logging.info(f"Processing: {iter_range} time snapshots")

    # Set up meta and zarr data writers
    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)
    metadata_writer = metadata.create_metadata_writer(
        bucket=cfg.output.bucket,
        folder=cfg.output.folder,
        run_id=cfg.run.run_id,
        fs_sync=fs_synch,
        flush_every=10_000,
    )

    # # Zarr Dataset
    # dataset_name = f"dataset_creation.zarr"
    # zarr_ds = zarr_dataset.ZarrDataset(args.bucket, args.folder, args.run_id, dataset_name, fs=fs,
    #                                    feature_channels=non_computed_feature_channels+feature_channels_computed,
    #                                    down_sample_res=args.down_sample_res)

    zarr_ds = zarr_dataset.ZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        feature_channels=non_computed_feature_channels + feature_channels_computed,
        down_sample_res=cfg.output.down_sample_res,
    )

    logging.info(f"Zarr dataset_creation created.")

    # Get our grid and static masks once ever. These never change.
    ds_grid, land_face_mask = set_up_grid_data_and_masks(cfg)
    grid = xgcm.Grid(ds_grid, periodic=False)

    for it in tqdm.tqdm(iter_range):
        # grab raw data for this iteration
        ds = get_raw_data.get_remote_llc_data(endpoint_url, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        logging.info(f"Data loaded for iteration: {it}")

        # now process this iteration of data
        process_time_snapshot(cfg, metadata_writer, zarr_ds, ds_merge, grid, land_face_mask, non_computed_feature_channels)

        ds_merge = None
        del ds_merge

        ds = None
        del ds


if __name__ == "__main__":
    main()