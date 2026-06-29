"""Reusable per-run and per-snapshot steps for the cutout pipeline.

Shared by generate_cutout_dataset (the CLI) and the integration tests so both
exercise identical logic instead of re-implementing it.
"""
import logging

import numpy as np
import xarray as xr

import dbof.preprocessing.static_masks as static_masks
import dbof.preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling
import dbof.cutout_dataset_creation.config as config
from dbof.cutout_dataset_creation.global_input import load_snapshot_features
from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
from dbof.preprocessing.ice_mask import generate_siarea_mask, generate_halo_ice_mask


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
    reuse it (e.g. stored as a patch channel) without recomputing.
    """
    log_gradb = np.log10(ds_merge["gradb2"])

    '''
    Here we compute the calculated loggradb2 into memory before creating our patches.
    While this is arguably inefficient if we do not do this our Dask graph splits and we will run into difficult errors
    or warnings to fix.
    The cause of this is either xmitgcm code calculating the gradients on the native grid or that we are using
    the gradient in our sampling logic. I believe it is the first but I am not sure yet. - Jake

    Todo with new code this is probably safe to remove. Needs to be tested.
    '''
    log_gradb_np = log_gradb.values  # materialize once; sampler works on numpy
    indices = weighted_coordinate_sampling.weighted_sample_on_grid(
        sample_points_per_snapshot, bias_to_high_gradients, log_gradb_np, sampling_mask,
    )
    return indices, log_gradb_np
