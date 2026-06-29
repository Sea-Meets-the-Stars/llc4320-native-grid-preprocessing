# TODO METAdata has extra id column unused. Get rid of it.

# stdlib
import logging

# numerical / compute
import numpy as np
import xgcm
import zarr
import xarray as xr

# distributed / IO
from dask.distributed import Client

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.static_masks as static_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
import dbof.preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling

import dbof.llc4320_ingestion.get_raw_data as get_raw_data

import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset

import dbof.cutout_dataset_creation.metadata as metadata
import dbof.cutout_dataset_creation.dask_pipeline as dask_pipeline
import dbof.cutout_dataset_creation.config as config
from dbof.cutout_dataset_creation.global_input import resolve_input_locations, resolve_date_prefixes, verify_feature_channels, verify_required_channels
from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
from dbof.utils.logging import generate_logging
from dbof.preprocessing.calculate_additional_fields import relative_vorticity
from dbof.cutout_dataset_creation.global_input import load_snapshot_features
from dbof.preprocessing.ice_mask import generate_siarea_mask, generate_halo_ice_mask

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

def set_up_grid_data_and_masks(cfg: config.JobConfig, fs):
    logging.info("Fetching global stitched grid file")
    grid = cfg.input.grid_access
    grid_reader = GlobalGridZarrReader(
        bucket=grid.bucket,
        folder=grid.folder,
        dataset_name=grid.dataset_name,
        fs=fs,
    )
    ds_grid = grid_reader.to_dataset_lazy()

    logging.info("Calculating land and face masks")
    land_halo_mask = static_masks.generate_halo_land_mask(ds_grid, cfg.output.target_km_res, stitched=True)

    return ds_grid, land_halo_mask


def process_time_snapshot(cfg: config.JobConfig, metadata_writer, zarr_ds, ds_merge, land_face_mask, feature_channels):

    logging.info(f"Calculating ice mask")
    # Ice mask from the already-loaded SIarea field, then buffered with a halo the same way we do for land.
    ice_mask = generate_siarea_mask(ds_merge["SIarea"].values)
    halo_ice_mask = generate_halo_ice_mask(
        ds_merge, ice_mask, cfg.output.target_km_res, stitched=True)
    merged_mask = halo_ice_mask & land_face_mask

    # Calculated Fields
    calculated_fields = {}

    # This must be included so long as we are sampling using it.
    # If we support additional sampling methods in the future, this becomes optional.
    # log_gradb = calculate_additional_fields.log_grad_b(ds_merge, grid)
    #
    # if "relative_vorticity" in computed_feature_channels:
    #     relative_vorticity = calculate_additional_fields.relative_vorticity(ds_merge, grid)
    #     calculated_fields["relative_vorticity"] = relative_vorticity

    logging.info(f"Sampling patch center points")
    # Sample patch centers weighted by log10 of the buoyancy gradient (gradb2),
    # one of the already-loaded frontal_structure feature channels.
    log_gradb = np.log10(ds_merge["gradb2"])
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
    calculated_fields["log_gradb_np"] = log_gradb_np

    # grow zarr ds to fit at most len of indices
    zarr_ds.grow_array(len(indices))

    #process data and write to s3
    images = dask_pipeline.run_patch_creation(zarr_ds, metadata_writer, cfg.output.down_sample_res, indices,
                                              ds_merge,
                                              cfg.output.target_km_res,
                                              metadata_cols,
                                              calculated_fields,
                                              model_feature_channels,
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

    ds_grid, land_face_mask = set_up_grid_data_and_masks(cfg, fs_in)


    for snapshot in tqdm.tqdm(date_prefixes):

        ds = load_snapshot_features(cfg.input, snapshot, feature_channels, fs_in, fs_in_sync)
        ds_merge = xr.merge([ds, ds_grid])

        process_time_snapshot(cfg, metadata_writer, zarr_ds, ds_merge, land_face_mask, feature_channels)

        ds_merge = None
        del ds_merge

        ds = None
        del ds


if __name__ == "__main__":
    main()