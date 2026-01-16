from dask import delayed
import torch
import dask
import logging
import uuid
import numpy as np
import dask.array as da

import data_preprocessing.spatial_patches as spatial_patches
import utils.physical_calculations as physical_calculations
import utils.native_gradient as ng

def calculate_gradients(ds_merge, grid):
    """
    Compute log10 of squared buoyancy gradients on the native LLC grid.

    The resulting log-gradient field is added to the
    merged dataset as a new variable.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing fields.
    grid : xgcm.Grid
        XGCM grid object.

    Returns
    -------
    ds_merge : xarray.Dataset
        Dataset augmented with `log_gradb`.
    log_gradb : dask.array.Array
        log10(|∇b|^2) field used for weighted sampling.
    """

    buoyancy = physical_calculations.buoyancy_of_field(ds_merge)

    # gradient of b
    zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_merge, grid=grid)

    #zonal_grad_b, merid_grad_b = dask.persist(zonal_grad_b, merid_grad_b)

    # zonal_grad_b = zonal_grad_b.persist()
    # merid_grad_b = merid_grad_b.persist()

    gradb2 = physical_calculations.grad_squared(zonal_grad_b, merid_grad_b)
    # gradb2 = gradb2.persist()

    log_gradb = da.log10(gradb2)

    #log_gradb = log_gradb.persist()

    #log_gradb_ds = log_gradb.to_dataset(name="log_gradb")

    #ds_merge = xr.merge([ds_merge, log_gradb_ds])

    # ds_merge["log_gradb"] = ds_merge["log_gradb"].persist()

    return ds_merge, log_gradb

@delayed
def write_image_and_metadata(zarr_ds, metadata_writer, patch_data, image, patch, down_sample_res, target_km_res, metadata_cols):
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


    # todo this should be safe here but test
    image_id = zarr_ds.append_image(data_sample)
    patch_metadata["dataset_index"] = image_id
    metadata_writer.add(patch_metadata)

    return data_sample, patch_metadata


def create_image_patch_lazy(ds, channels, patch):
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
    log_slice = array[
            patch["face"],
            patch["j_start"] : patch["j_end"] + 1,
            patch["i_start"] : patch["i_end"] + 1,
        ].astype("float32")

    return log_slice


# todo future proof for more dask arrays like loggradb
# todo quadruple check that dask respects order
def create_image_patches_batch_as_tensors_dask(ds, log_gradb_np, lazy_channels, patches, scheduler='threads'):

    # 1. Build lazy dask arrays (DS channels only)
    patch_arrays = [
        create_image_patch_lazy(ds, lazy_channels, patch)
        for patch in patches
    ]

    # 2. Compute in parallel
    computed_arrays = dask.compute(*patch_arrays, scheduler=scheduler)

    tensors = []
    # 3. Append log_gradb (NumPy) per patch
    for arr, patch in zip(computed_arrays, patches):

        log_slice = create_image_patch_numpy(log_gradb_np, patch)

        log_slice = np.expand_dims(log_slice, axis=0)
        img = np.concatenate((arr, log_slice), axis=0)
        tensors.append(torch.from_numpy(img))

    return tensors

# not parallelized.
def extract_patch_extents(index, ds_merge, log_gradb_np, target_km_res):

    """Extract all needed data from ds_merge for one patch"""
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

# computed dask_arrays must be numpy
# todo remove loggging add to main
def run_patch_creation(zarr_ds, metadata_writer, down_sample_res,
                 indices, ds_merge, log_gradb_np, target_km_res, feature_channels_lazy, metadata_cols):


    # Extract all patch data extents ---------------------------------------------------------
    patch_meta_data_list = []
    patches = []
    logging.info(f"Starting patch extents creation") # this takes about 1.5s per patch

    # todo in the future we could probably figure out how to parallelize this
    # currently there are issues with computing ds_merge down stream that break parallization code
    for index in indices:
        patch, patch_meta_data = extract_patch_extents(index, ds_merge, log_gradb_np, target_km_res)

        if patch_meta_data is not None:
            patch_meta_data_list.append(patch_meta_data)
            patches.append(patch)


    # Create images of patches ----------------------------------------------------------------------
    logging.info("Starting batched image creation")
    images = create_image_patches_batch_as_tensors_dask(ds_merge, log_gradb_np, feature_channels_lazy, patches, scheduler='threads')


    # Downsample images and get metadata ----------------------------------------------------
    logging.info("Downsampling Images")
    tasks = []
    for pd, image, patch in zip(patch_meta_data_list, images, patches):
        tasks.append(write_image_and_metadata(zarr_ds, metadata_writer, pd, image, patch, down_sample_res, target_km_res, metadata_cols))

    dask.compute(*tasks, scheduler='threads')

    return images
