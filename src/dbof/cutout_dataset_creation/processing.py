"""Reusable per-run and per-snapshot steps for the cutout pipeline.

Shared by generate_cutout_dataset (the CLI) and the integration tests so both
exercise identical logic instead of re-implementing it.
"""
import logging

import numpy as np
import xarray as xr
import zarr
import tqdm
from dask.distributed import Client

import dbof.preprocessing.static_masks as static_masks
import dbof.preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling
import dbof.cutout_dataset_creation.config as config
import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.cutout_dataset_creation.metadata as metadata
import dbof.cutout_dataset_creation.dask_pipeline as dask_pipeline
from dbof.cutout_dataset_creation.global_input import load_snapshot_features
from dbof.cutout_dataset_creation.global_input import (
    resolve_input_locations, resolve_date_prefixes,
    verify_feature_channels, verify_required_channels,
)
from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
from dbof.io.filesystems import create_s3_filesystems
from dbof.preprocessing.ice_mask import generate_siarea_mask, generate_halo_ice_mask

metadata_cols = [
    "image_id",
    "native_grid",
    "center_grid_j",
    "center_grid_i",
    "center_lat",
    "center_lon",
    "target_km_res",
    "real_km_w",
    "real_km_h",
    "pre_interp_res",
    "log_grad_b_2_center",
    "time_snapshot",
]

# Grid coordinate channels appended to every cutout after the feature channels.
# XC = longitude, YC = latitude (the grid store's variable names).
COORD_CHANNELS = ["XC", "YC"]


def set_up_grid_data_and_land_masks(cfg: config.JobConfig, fs):
    """Load the stitched global grid and the static land halo mask (once per run)."""
    logging.info("Fetching global stitched grid file")
    grid = cfg.input.grid_access
    grid_reader = GlobalGridZarrReader(
        bucket=grid.bucket, folder=grid.folder, dataset_name=grid.dataset_name, fs=fs,
    )
    ds_grid = grid_reader.to_dataset_lazy()

    logging.info("Calculating land and face masks")
    land_halo_mask = static_masks.generate_halo_land_mask(
        ds_grid, cfg.output.target_km_res, stitched=True)
    return ds_grid, land_halo_mask


def load_snapshot(cfg: config.JobConfig, date_prefix, feature_channels, ds_grid, fs, fs_sync):
    """Load one snapshot's feature channels and merge them with the grid."""
    ds = load_snapshot_features(cfg.input, date_prefix, feature_channels, fs, fs_sync)
    return xr.merge([ds, ds_grid])


def build_sampling_mask(ds_merge, land_halo_mask, target_km_res):
    """Per-snapshot ice halo mask AND-ed with the static land halo mask.

    Returns a boolean (j, i) array; True = ocean point eligible for sampling.
    """
    ice_mask = generate_siarea_mask(ds_merge["SIarea"].values)
    halo_ice_mask = generate_halo_ice_mask(ds_merge, ice_mask, target_km_res, stitched=True)
    return halo_ice_mask & land_halo_mask


def sample_cutout_centers_with_loggradb(ds_merge, sampling_mask, sample_points_per_snapshot,
                                        bias_to_high_gradients):
    """Weighted-sample cutout-center (j, i) indices using log10 of gradb2.

    Returns ``(indices, log_gradb_np)``.  log_gradb is materialized to numpy
    before sampling (the sampler works on numpy) and returned so the caller can
    reuse it (e.g. stored as a cutout channel) without recomputing.
    """
    log_gradb = np.log10(ds_merge["gradb2"])

    # todo, gradb2 is materialized later. We could do this only once here but maybe is not worth the complexity
    log_gradb_np = log_gradb.values  # materialize once; sampler works on numpy
    indices = weighted_coordinate_sampling.weighted_sample_on_grid(
        sample_points_per_snapshot, bias_to_high_gradients, log_gradb_np, sampling_mask,
    )
    return indices, log_gradb_np


def process_time_snapshot(cfg: config.JobConfig, metadata_writer, zarr_ds, ds_merge, land_face_mask, channels, date_prefix):

    logging.info("Calculating sampling mask (ice)")
    merged_mask = build_sampling_mask(ds_merge, land_face_mask, cfg.output.target_km_res)

    logging.info("Sampling cutout center points")
    indices, log_gradb_np = sample_cutout_centers_with_loggradb(
        ds_merge, merged_mask,
        cfg.sampling.sample_points_per_snapshot, cfg.sampling.bias_to_high_gradients,
    )

    # grow zarr ds to fit at most len of indices
    zarr_ds.grow_array(len(indices))

    #process data and write to s3
    dask_pipeline.run_cutout_creation(zarr_ds, metadata_writer, cfg.output.down_sample_res, indices,
                                     ds_merge,
                                     cfg.output.target_km_res,
                                     metadata_cols,
                                     log_gradb_np,
                                     date_prefix,
                                     channels,
                                     )

    # flush metada
    metadata_writer.close()

    # for dask
    ds_merge = None
    del ds_merge

    merged_mask = None
    del merged_mask


def run(cfg: config.JobConfig):
    """Generate the cutout dataset for a fully-resolved config."""
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

    # Grid lat/lon (XC/YC) are written as extra channels after the requested features.
    output_channels = feature_channels + COORD_CHANNELS

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
        channel_names=output_channels,
        down_sample_res=cfg.output.down_sample_res,
        target_km_res=cfg.output.target_km_res,
    )

    logging.info(f"Zarr dataset created.")

    ds_grid, land_face_mask = set_up_grid_data_and_land_masks(cfg, fs_in)

    for snapshot in tqdm.tqdm(date_prefixes):

        ds_merge = load_snapshot(cfg, snapshot, feature_channels, ds_grid, fs_in, fs_in_sync)

        process_time_snapshot(cfg, metadata_writer, zarr_ds, ds_merge, land_face_mask, output_channels, snapshot)

        ds_merge = None
        del ds_merge
