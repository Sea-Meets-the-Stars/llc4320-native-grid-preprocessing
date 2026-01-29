from dask import delayed
import torch
import dask
import uuid
import numpy as np
import dask.array as da
import logging

import dbof.dataset_creation.spatial_patches as spatial_patches

# todo this file deserves a document to explain

@delayed
def downsample_image_and_write_image_and_metadata(zarr_ds, metadata_writer, patch_data, image, patch, down_sample_res, target_km_res, metadata_cols):
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

def create_image_patch_lazy(ds, channels, patch):
    '''
    This is a lazy function meant to be called by dask.
    dask.compute()
    '''
    channels_array = []

    for channel in channels:
        feature = ds[channel].isel(
            face=patch["face"],
            j=slice(patch["j_start"], patch["j_end"] + 1),
            i=slice(patch["i_start"], patch["i_end"] + 1),
        )
        channels_array.append(feature.data)

    # Stack lazily (Dask only)
    img = da.stack(channels_array, axis=0).astype("float32")  # (C, H, W)

    return img


def create_image_patch_numpy(array, patch):
    '''
    For use on already computed data that is in memory as a numpy array.
    '''
    log_slice = array[
            patch["face"],
            patch["j_start"] : patch["j_end"] + 1,
            patch["i_start"] : patch["i_end"] + 1,
        ].astype("float32")

    return log_slice

# todo future proof for more numpy arrays like loggradb - needed for wind curl in the near future
# todo the lazy channels thing is really confusing to read. Lets make that clear
def create_image_patches_batch_as_tensors_dask(ds, log_gradb_np, lazy_channels, patches, scheduler='threads'):
    # 1. Build lazy dask arrays (DS channels only)
    patch_arrays = [
        create_image_patch_lazy(ds, lazy_channels, patch)
        for patch in patches
    ]

    # 2. Compute all images in parallel
    computed_arrays = dask.compute(*patch_arrays, scheduler=scheduler)

    tensors = []
    # 3. Append log_gradb (NumPy) per patch, these are from already computed data
    for arr, patch in zip(computed_arrays, patches):

        log_slice = create_image_patch_numpy(log_gradb_np, patch)

        log_slice = np.expand_dims(log_slice, axis=0)
        img = np.concatenate((arr, log_slice), axis=0)
        tensors.append(torch.from_numpy(img))

    return tensors

def extract_patch_extents_and_metadata_in_series(index, ds_merge, log_gradb_np, target_km_res):
    """
    This function extracts metadata for the patch from ds_merge
    it then gets the spatial extents for the patch

    todo in the future we could probably figure out how to parallelize this.
    Currently there are issues with computing ds_merge down stream that break parallel code
    the ds_merge.YC[index].values.item() particularly
    """

    patch_meta_data = {}
    patch_meta_data["index"] = index
    patch_meta_data["center_lat"] = ds_merge.YC[index].values.item()
    patch_meta_data["center_lon"] = ds_merge.XC[index].values.item()
    patch_meta_data["log_grad_b_2_center"] = log_gradb_np[index]
    patch_meta_data["time_snapshot"] = np.datetime64(ds_merge.time.item(), 'ns')

    patch = spatial_patches.get_lat_lon_extents_of_patch(index, ds_merge, target_km_res)

    if patch is None:
        return None, None

    return patch, patch_meta_data

def run_patch_creation(zarr_ds, metadata_writer, down_sample_res,
                 indices, ds_merge, log_gradb_np, target_km_res, feature_channels_lazy, metadata_cols, logger=None):

    logger = logger or logging.getLogger(__name__)

    patch_meta_data_list = []
    patches = []

    logger.info(f"Starting patch extents dataset_creation")
    for index in indices:
        patch, patch_meta_data = extract_patch_extents_and_metadata_in_series(index, ds_merge, log_gradb_np, target_km_res)

        if patch_meta_data is not None:
            patch_meta_data_list.append(patch_meta_data)
            patches.append(patch)

    logger.info("Starting batched image dataset_creation")
    images = create_image_patches_batch_as_tensors_dask(ds_merge, log_gradb_np, feature_channels_lazy, patches, scheduler='threads')

    # Downsample images and get metadata ----------------------------------------------------
    logger.info("Downsampling Images")
    tasks = []
    for pd, image, patch in zip(patch_meta_data_list, images, patches):
        tasks.append(downsample_image_and_write_image_and_metadata(zarr_ds, metadata_writer, pd, image, patch, down_sample_res, target_km_res, metadata_cols))

    # Writing data is only thread safe for now. Not processes safe.
    dask.compute(*tasks, scheduler='threads')

    return images