from dask import delayed
import torch
import dask
import numpy as np
import dask.array as da
import logging
from datetime import datetime

import dbof.cutout_dataset_creation.spatial_cutouts as spatial_cutouts

@delayed
def downsample_and_write_cutout_lazy(zarr_ds, metadata_writer, cutout_data, cutout_img, cutout, down_sample_res, target_km_res, metadata_cols, channels):
    index = cutout_data["index"]

    cutout_metadata = dict.fromkeys(metadata_cols)
    cutout_metadata["native_grid"] = "LLC4320"
    cutout_metadata["center_grid_j"] = index[0]
    cutout_metadata["center_grid_i"] = index[1]
    cutout_metadata["target_km_res"] = target_km_res
    cutout_metadata["center_lat"] = cutout_data["center_lat"]
    cutout_metadata["center_lon"] = cutout_data["center_lon"]
    cutout_metadata["log_grad_b_2_center"] = cutout_data["log_grad_b_2_center"]

    cutout_metadata["time_snapshot"] = cutout_data["time_snapshot"]

    cutout_metadata["real_km_w"] = cutout["real_km_w"]
    cutout_metadata["real_km_h"] = cutout["real_km_h"]

    cutout_metadata["pre_interp_res"] = cutout_img[0].shape

    data_sample = spatial_cutouts.downsample_image(cutout_img, channels, target_dim=down_sample_res)
    image_id = zarr_ds.append_image(data_sample)

    cutout_metadata["image_id"] = image_id
    metadata_writer.add(cutout_metadata)

    return data_sample, cutout_metadata

def create_image_cutout_lazy(ds_merge, channels, cutout):
    '''
    Lazy (dask) extraction of one cutout's image stack.

    Slices every channel from ds_merge over the cutout's (j, i) extent and
    stacks them into a (C, H, W) array.  ds_merge holds the requested feature
    channels plus the grid vars, so we iterate channels directly -- no
    per-channel special-casing or faces.
    '''
    # Stack in channels order; this matches the channel_names attr stored with the
    # dataset, so channel c of every cutout is channel_names[c].
    channels_array = []

    for channel in channels:
        feature = ds_merge[channel].isel(
            j=slice(cutout["j_start"], cutout["j_end"] + 1),
            i=slice(cutout["i_start"], cutout["i_end"] + 1),
        )
        channels_array.append(feature.data)

    # Stack lazily (Dask only)
    img = da.stack(channels_array, axis=0).astype("float32")  # (C, H, W)

    return img

def create_image_cutouts_batch_as_tensors_dask(ds_merge, channels, cutouts, scheduler='threads'):
    """Extract each cutout's channels and return them as a list of
    (C, H, W) torch tensors."""
    cutout_arrays = [
        create_image_cutout_lazy(ds_merge, channels, cutout)
        for cutout in cutouts
    ]
    computed_arrays = dask.compute(*cutout_arrays, scheduler=scheduler)
    return [torch.from_numpy(np.asarray(arr)) for arr in computed_arrays]

def extract_cutout_extents_and_metadata_in_series(index, XC, YC, log_gradb_np,
                                                  dxC, dyC, target_km_res, time_snapshot):
    """
    Build cutout metadata + spatial extents for one ``(j, i)`` center.

    All grid inputs are numpy (materialized once by the caller) so this stays a
    cheap in-memory operation.
    """

    cutout_meta_data = {}
    cutout_meta_data["index"] = index
    cutout_meta_data["center_lat"] = float(YC[index])
    cutout_meta_data["center_lon"] = float(XC[index])
    cutout_meta_data["log_grad_b_2_center"] = log_gradb_np[index]
    cutout_meta_data["time_snapshot"] = time_snapshot

    cutout = spatial_cutouts.get_lat_lon_extents_of_cutout(index, dxC, dyC, XC.shape, target_km_res)

    if cutout is None:
        return None, None

    return cutout, cutout_meta_data

def run_cutout_creation(zarr_ds, metadata_writer, down_sample_res,
                 indices, ds_merge, target_km_res, metadata_cols, log_gradb_np,
                 date_prefix, channels, logger=None):

    logger = logger or logging.getLogger(__name__)

    # Materialize grid arrays once (they are single-chunk dask arrays over S3);
    # per-cutout indexing would otherwise re-read the full field every iteration.
    XC = np.asarray(ds_merge["XC"].values)
    YC = np.asarray(ds_merge["YC"].values)
    dxC = np.asarray(ds_merge["dxC"].values)
    dyC = np.asarray(ds_merge["dyC"].values)
    time_snapshot = np.datetime64(datetime.strptime(date_prefix, "%Y%m%d_%H%M%S"), "ns")

    cutout_meta_data_list = []
    cutouts = []

    logger.info("Starting cutout extents + metadata")
    for index in indices:
        cutout, cutout_meta_data = extract_cutout_extents_and_metadata_in_series(
            index, XC, YC, log_gradb_np, dxC, dyC, target_km_res, time_snapshot)

        # None when the cutout extends past the grid boundaries (off the edge)
        if cutout_meta_data is not None:
            cutout_meta_data_list.append(cutout_meta_data)
            cutouts.append(cutout)

    logger.info("Starting batched cutout image creation")
    images = create_image_cutouts_batch_as_tensors_dask(ds_merge, channels, cutouts, scheduler='threads')

    logger.info(f"Generated {len(images)} cutouts")

    # Downsample cutouts and write metadata --------------------------------------------------
    logger.info("Downsampling cutouts and writing in parallel")
    tasks = []
    for meta, image, cutout in zip(cutout_meta_data_list, images, cutouts):
        tasks.append(downsample_and_write_cutout_lazy(zarr_ds, metadata_writer, meta, image, cutout, down_sample_res, target_km_res, metadata_cols, channels))

    # Writing data is only thread safe for now. Not processes safe.
    dask.compute(*tasks, scheduler='threads')
