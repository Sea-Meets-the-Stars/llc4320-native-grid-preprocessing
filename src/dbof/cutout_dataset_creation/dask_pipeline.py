from dask import delayed
import torch
import dask
import uuid
import numpy as np
import dask.array as da
import logging
from datetime import datetime

import dbof.cutout_dataset_creation.spatial_patches as spatial_patches

@delayed
def downsample_image_and_write_image_and_metadata_lazy(zarr_ds, metadata_writer, patch_data, image, patch, down_sample_res, target_km_res, metadata_cols):
    index = patch_data["index"]

    patch_metadata = dict.fromkeys(metadata_cols)
    patch_metadata["id"] = str(uuid.uuid4())
    patch_metadata["native_grid"] = "LLC4320"
    patch_metadata["center_grid_face"] = index[0]
    patch_metadata["center_grid_j"] = index[1]
    patch_metadata["center_grid_i"] = index[2]
    patch_metadata["target_km_res"] = target_km_res
    patch_metadata["center_lat"] = patch_data["center_lat"]
    patch_metadata["center_lon"] = patch_data["center_lon"]
    patch_metadata["log_grad_b_2_center"] = patch_data["log_grad_b_2_center"]

    patch_metadata["time_snapshot"] = patch_data["time_snapshot"]

    patch_metadata["real_km_w"] = patch["real_km_w"]
    patch_metadata["real_km_h"] = patch["real_km_h"]

    # img_patch = patch_data["img_patch"]
    img_patch = image

    patch_metadata["pre_interp_res"] = img_patch[0].shape

    data_sample = spatial_patches.downsample_image(img_patch, target_dim=down_sample_res)
    image_id = zarr_ds.append_image(data_sample)

    patch_metadata["dataset_index"] = image_id
    metadata_writer.add(patch_metadata)

    return data_sample, patch_metadata

def create_image_cutout_lazy(ds_merge, feature_channels, cutout):
    '''
    Lazy (dask) extraction of one cutout's image stack.

    Slices every requested feature channel from ds_merge over the cutout's
    (j, i) extent and stacks them into a (C, H, W) array.  ds_merge holds
    exactly the user-requested feature channels (plus grid vars), so we iterate
    feature_channels directly -- no per-feature special-casing or faces.
    '''
    channels_array = []

    for channel in feature_channels:
        feature = ds_merge[channel].isel(
            j=slice(cutout["j_start"], cutout["j_end"] + 1),
            i=slice(cutout["i_start"], cutout["i_end"] + 1),
        )
        channels_array.append(feature.data)

    # Stack lazily (Dask only)
    img = da.stack(channels_array, axis=0).astype("float32")  # (C, H, W)

    return img

def create_image_cutouts_batch_as_tensors_dask(ds_merge, feature_channels, cutouts, scheduler='threads'):
    """Extract each cutout's requested feature channels and return them as
    a list of (C, H, W) torch tensors."""
    cutout_arrays = [
        create_image_cutout_lazy(ds_merge, feature_channels, cutout)
        for cutout in cutouts
    ]
    computed_arrays = dask.compute(*cutout_arrays, scheduler=scheduler) # todo does this create problems materializing log_grad_np? If not we don't need to materialize it or pass it around
    return [torch.from_numpy(np.asarray(arr)) for arr in computed_arrays]

def extract_patch_extents_and_metadata_in_series(index, XC, YC, log_gradb_np,
                                                 dxC, dyC, target_km_res, time_snapshot):
    """
    Build patch metadata + spatial extents for one ``(j, i)`` center.

    All grid inputs are numpy (materialized once by the caller) so this stays a
    cheap in-memory operation.
    """

    patch_meta_data = {}
    patch_meta_data["index"] = index
    patch_meta_data["center_lat"] = float(YC[index])
    patch_meta_data["center_lon"] = float(XC[index])
    patch_meta_data["log_grad_b_2_center"] = log_gradb_np[index]
    patch_meta_data["time_snapshot"] = time_snapshot

    patch = spatial_patches.get_lat_lon_extents_of_patch(index, dxC, dyC, XC.shape, target_km_res)

    if patch is None:
        return None, None

    return patch, patch_meta_data

def run_patch_creation(zarr_ds, metadata_writer, down_sample_res,
                 indices, ds_merge, target_km_res, metadata_cols, log_gradb_np,
                 date_prefix, feature_channels, logger=None):

    logger = logger or logging.getLogger(__name__)

    # Materialize grid arrays once (they are single-chunk dask arrays over S3);
    # per-cutout indexing would otherwise re-read the full field every iteration.
    XC = np.asarray(ds_merge["XC"].values)
    YC = np.asarray(ds_merge["YC"].values)
    dxC = np.asarray(ds_merge["dxC"].values)
    dyC = np.asarray(ds_merge["dyC"].values)
    time_snapshot = np.datetime64(datetime.strptime(date_prefix, "%Y%m%d_%H%M%S"), "ns")

    patch_meta_data_list = []
    patches = []

    logger.info(f"Starting patch extents cutout_dataset_creation")
    for index in indices:
        # Calculate the extents of each patch
        patch, patch_meta_data = extract_patch_extents_and_metadata_in_series(
            index, XC, YC, log_gradb_np, dxC, dyC, target_km_res, time_snapshot)

        # Occurs from some error in the patch, likely extends past the grid boundaries (off the face of the world)
        if patch_meta_data is not None:
            patch_meta_data_list.append(patch_meta_data)
            patches.append(patch)

    logger.info("Starting batched image cutout_dataset_creation")
    images = create_image_cutouts_batch_as_tensors_dask(ds_merge, feature_channels, patches, scheduler='threads')

    # todo left off here

    # Downsample images and get metadata ----------------------------------------------------
    logger.info("Downsampling Images")
    tasks = []
    for pd, image, patch in zip(patch_meta_data_list, images, patches):
        tasks.append(downsample_image_and_write_image_and_metadata_lazy(zarr_ds, metadata_writer, pd, image, patch, down_sample_res, target_km_res, metadata_cols))

    # Writing data is only thread safe for now. Not processes safe.
    dask.compute(*tasks, scheduler='threads')

    return images