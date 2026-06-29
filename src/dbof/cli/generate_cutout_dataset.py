# TODO METAdata has extra id column unused. Get rid of it.

# stdlib
import logging

# numerical / compute
import zarr

# distributed / IO
from dask.distributed import Client

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.cutout_dataset_creation.metadata as metadata
import dbof.cutout_dataset_creation.dask_pipeline as dask_pipeline
import dbof.cutout_dataset_creation.config as config
import dbof.cutout_dataset_creation.processing as processing
from dbof.cutout_dataset_creation.global_input import resolve_input_locations, resolve_date_prefixes, verify_feature_channels, verify_required_channels
from dbof.utils.logging import generate_logging

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

def process_time_snapshot(cfg: config.JobConfig, metadata_writer, zarr_ds, ds_merge, land_face_mask, feature_channels):

    logging.info("Calculating sampling mask (ice)")
    merged_mask = processing.build_sampling_mask(ds_merge, land_face_mask, cfg.output.target_km_res)

    logging.info("Sampling patch center points")
    indices, log_gradb_np = processing.sample_cutout_centers_with_loggradb(
        ds_merge, merged_mask,
        cfg.sampling.sample_points_per_snapshot, cfg.sampling.bias_to_high_gradients,
    )

    # # Move non tracer values to tracer points. This allows us to stack images for our final patches.
    # ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    # ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    # grow zarr ds to fit at most len of indices
    zarr_ds.grow_array(len(indices))

    #process data and write to s3
    images = dask_pipeline.run_patch_creation(zarr_ds, metadata_writer, cfg.output.down_sample_res, indices,
                                              ds_merge,
                                              cfg.output.target_km_res,
                                              metadata_cols,
                                              log_gradb_np
                                              )

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
    Entry point for native-grid LLC patch cutout_dataset_creation generation.

    Orchestrates argument parsing, Dask setup, filesystem initialization,
    and iteration over time snapshots.
    """
    cli = config.parse_args()
    cfg = config.load_config(cli.config)
    print(cfg)

    # override run_id if passed in through cli
    if cli.run_id is not None:
        cfg = config.JobConfig(
            run=config.RunConfig(run_id=cli.run_id, log_dir=cfg.run.log_dir),
            input=cfg.input,
            sampling=cfg.sampling,
            output=cfg.output,
            features=cfg.features,
            runtime=cfg.runtime,
        )

    generate_logging(cfg.run, log_filename="generate_cutout_dataset.log")

    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    input_base, grid_uri = resolve_input_locations(cfg.input)
    logging.info(f"Input source : {input_base}")
    logging.info(f"Grid store   : {grid_uri}")

    fs_in, fs_in_sync = create_s3_filesystems(cfg.input.s3_endpoint)

    date_prefixes = resolve_date_prefixes(cfg.input, fs_in_sync)
    logging.info(f"Date prefixes : {date_prefixes}")

    feature_channels = [c.strip() for c in cfg.features.feature_channels if c.strip()]
    logging.info(f"Feature Channels to load   : {feature_channels}")

    verify_feature_channels(cfg.input, date_prefixes[0], feature_channels, fs_in, fs_in_sync)
    logging.info(f"All requested feature channels present in {date_prefixes[0]}")

    verify_required_channels(cfg.input, date_prefixes[0], fs_in, fs_in_sync)

    # Set concurrency for zarr ds writes
    zarr.config.set({'async.concurrency':  cfg.runtime.zarr_async_concurrency})
    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores
    logging.info(f"Dask Client {dask_client}")

    # Set up meta and zarr data writers
    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)
    metadata_writer = metadata.create_metadata_writer(
        bucket=cfg.output.bucket,
        folder=cfg.output.folder,
        run_id=cfg.run.run_id,
        fs_sync=fs_synch,
        flush_every=10_000,
    )

    zarr_ds = zarr_dataset.ZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        num_channels=len(feature_channels),
        down_sample_res=cfg.output.down_sample_res,
    )

    logging.info(f"Zarr dataset created.")

    ds_grid, land_face_mask = processing.set_up_grid_data_and_land_masks(cfg, fs_in)

    for snapshot in tqdm.tqdm(date_prefixes):

        ds_merge = processing.load_snapshot(cfg, snapshot, feature_channels, ds_grid, fs_in, fs_in_sync)

        process_time_snapshot(cfg, metadata_writer, zarr_ds, ds_merge, land_face_mask, feature_channels)

        ds_merge = None
        del ds_merge


if __name__ == "__main__":
    main()