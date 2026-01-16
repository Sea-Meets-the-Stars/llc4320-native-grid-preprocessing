# stdlib
import argparse
import uuid
from concurrent.futures import ThreadPoolExecutor
import sys
import logging
from pathlib import Path

# numerical / compute
import numpy as np
import dask.array as da
import xarray as xr
import xgcm
import zarr

# distributed / IO
import dask
from dask.distributed import Client
import fsspec

# progress
import tqdm

# internal
import data_ingestion.get_raw_data as get_raw_data
import data_preprocessing.preproc_llc_core_data as preproc_llc_core_data
import data_preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling
import data_preprocessing.halo_mask as halo_mask
import data_preprocessing.spatial_patches as spatial_patches
import dataset_creation.metadata as metadata
import dataset_creation.zarr_dataset as zarr_dataset
from dataset_creation import dbof_front_training_utils


# import utils.native_gradient as ng
# import utils.physical_calculations as physical_calculations

# Constants --------------------------
TS_PER_HOUR = 144 # model cadence: 25 s → 144 steps/hr
MAX_ITER = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368
LLC_FACES = range(13)

# url of our raw data - this may need to be an input in the future
endpoint_url = 'https://mghp.osn.xsede.org'

# # input features
# feature_channels = ["Eta", "Salt", "Theta", "U", "V", "W", "log_gradb"]

# todo lets make this more clear
feature_channels_lazy = ["Eta", "Salt", "Theta", "U", "V", "W"] #,
feature_channels_computed = ["log_gradb"]

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

def parse_args(p):
    """
    Parse command-line arguments

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser instance to populate.

    Returns
    -------
    argparse.Namespace
    """
    #data selection-----
    p.add_argument("--sampling_step", required=True,
                   help="Stride in timesteps", type=int)

    p.add_argument("--start_record", default=1180, type=int,
                   help="Starting record. Default is first record with wind forcing.")

    p.add_argument("--timestep_hours", default=None, type=int,
                   help="How many total hours to load between start iteration and end iteration."
                        "If not given, the script will proces at provided sampling_step until the end of the data.")

    #sampling options--------
    p.add_argument("--bias_to_high_gradients", default=2, type=float,
                   help="Bias to high B in sampled data")

    p.add_argument("--sample_points_per_snapshot", default=100, type=int,
                   help="How many points per snapshot to sample")

    #s3 config----------
    p.add_argument("--s3_endpoint", default="https://s3-west.nrp-nautilus.io",
                   help="nrp s3 endpoint. Likely leave default.")

    p.add_argument("--bucket", default="llc/",
                   help="NRP s3 bucket to save data")

    p.add_argument("--folder", default="native_grid_dbof_training_data/",
                   help="NRP s3 bucket to save data")

    p.add_argument("--run_id", required=True,
                   help="Ensure run Id is unique inside s3://bucket/folder/ ")

    #return data-------
    p.add_argument("--target_km_res", default=150, type=int,
                   help="Target physical resolution in km. Default is 150.")

    p.add_argument("--down_sample_res", default=64, type=int,
                   help="Downsampling resolution of W and H in pixels. Default is 64.")

    # parallel options-------
    p.add_argument("--num_workers", default=1, type=int,
                   help="How many threads would you like to create? Note : Should be <= sample_points_per_snapshot")

    args = p.parse_args()
    return args

# def calculate_gradients(ds_merge, grid):
#     """
#     Compute log10 of squared buoyancy gradients on the native LLC grid.
#
#     The resulting log-gradient field is added to the
#     merged dataset as a new variable.
#
#     Parameters
#     ----------
#     ds_merge : xarray.Dataset
#         Dataset containing fields.
#     grid : xgcm.Grid
#         XGCM grid object.
#
#     Returns
#     -------
#     ds_merge : xarray.Dataset
#         Dataset augmented with `log_gradb`.
#     log_gradb : dask.array.Array
#         log10(|∇b|^2) field used for weighted sampling.
#     """
#
#     buoyancy = physical_calculations.buoyancy_of_field(ds_merge)
#
#     # gradient of b
#     zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_merge, grid=grid)
#
#     zonal_grad_b, merid_grad_b = dask.persist(zonal_grad_b, merid_grad_b)
#
#     # zonal_grad_b = zonal_grad_b.persist()
#     # merid_grad_b = merid_grad_b.persist()
#
#     gradb2 = physical_calculations.grad_squared(zonal_grad_b, merid_grad_b)
#     # gradb2 = gradb2.persist()
#
#     log_gradb = da.log10(gradb2)
#     log_gradb = log_gradb.persist()
#
#     log_gradb_ds = log_gradb.to_dataset(name="log_gradb")
#     ds_merge = xr.merge([ds_merge, log_gradb_ds])
#
#     ds_merge["log_gradb"] = ds_merge["log_gradb"].persist()
#
#     return ds_merge #, log_gradb

def generate_filesystems(s3_endpoint):
    """
    Create asynchronous and synchronous S3 filesystems.

    The asynchronous filesystem is used for Zarr writes,
    while the synchronous filesystem is used for Parquet metadata.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL.

    Returns
    -------
    fs : fsspec.AbstractFileSystem
        Asynchronous S3 filesystem.
    fs_synch : fsspec.AbstractFileSystem
        Synchronous S3 filesystem.
    """

    fs = fsspec.filesystem(
        "s3",  #
        asynchronous=True,
        client_kwargs={
            "endpoint_url": s3_endpoint,

        },

        # These become botocore.client.Config(...)
        config_kwargs={
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "s3": {
                "addressing_style": "path",
                "payload_signing_enabled": False,
                "use_accelerate_endpoint": False,
                "use_dualstack_endpoint": False,
            },
        },
    )

    fs_synch = fsspec.filesystem(
        "s3",  #
        asynchronous=False,
        client_kwargs={
            "endpoint_url": s3_endpoint,

        },

        # These become botocore.client.Config(...)
        config_kwargs={
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "s3": {
                "addressing_style": "path",
                "payload_signing_enabled": False,
                "use_accelerate_endpoint": False,
                "use_dualstack_endpoint": False,
            },
        },
    )

    return fs, fs_synch

def generate_metadata_writer(bucket, folder, run_id, fs_synch):
    meda_data_folder_path = "s3://" + bucket + folder + run_id + f"metadata/"
    return metadata.MetadataWriter(meda_data_folder_path, flush_every=10000, fs = fs_synch)

# def worker_task(zarr_ds, metadata_writer, down_sample_res,
#                 indices, ds_merge, target_km_res):
#     """
#     Process a subset of sampled grid indices and write patches + metadata.
#
#     Each worker:
#       - extracts spatial patches
#       - downsamples to target resolution
#       - appends image data to the Zarr dataset
#       - records per-patch metadata
#
#     Parameters
#     ----------
#     zarr_ds : ZarrDataset
#         Thread-safe Zarr dataset writer.
#     metadata_writer : MetadataWriter
#         Thread-safe metadata writer.
#     down_sample_res : int
#         Output spatial resolution in pixels.
#     indices : iterable
#         Iterable of (face, j, i) grid indices. Center points of our patches.
#     ds_merge : xarray.Dataset
#         Dataset containing fields and gradients.
#     target_km_res : float
#         Target physical resolution in km.
#     """
#
#     for index in indices:
#
#         index = tuple(index)
#
#         if (index is None):
#             continue
#
#         patch_metadata = dict.fromkeys(metadata_cols)
#         patch_metadata["id"] = str(uuid.uuid4())
#         patch_metadata["native_grid"] = "LLC4320" # todo all supported for now
#         patch_metadata["center_grid_face"] = index[0]
#         patch_metadata["center_grid_j"] = index[1]
#         patch_metadata["center_grid_i"] = index[2]
#         patch_metadata["target_km_res"] = target_km_res
#         patch_metadata["center_lat"] = ds_merge.YC[index].values.item()
#         patch_metadata["center_lon"] = ds_merge.XC[index].values.item()
#         patch_metadata["log_grad_b_2_center"] = (ds_merge.log_gradb[index].values.item())
#         patch_metadata["time_snapshot"] = np.datetime64(ds_merge.time.item(), 'ns')
#
#         patch = spatial_patches.get_lat_lon_extents_of_patch(index, ds_merge, target_km_res)
#
#         if(patch is None):
#             logging.warning(f"Skipping index {index}")
#             continue
#
#         patch_metadata["real_km_w"] = patch["real_km_w"]
#         patch_metadata["real_km_h"] = patch["real_km_h"]
#
#         img_patch = spatial_patches.create_image_patch(ds_merge, feature_channels, patch)
#
#         patch_metadata["pre_interp_res"] = img_patch[0].shape
#         data_sample = spatial_patches.downsample_image(img_patch, target_dim=down_sample_res)
#
#         # add data to zarr
#         image_id = zarr_ds.append_image(data_sample)
#         patch_metadata["dataset_index"] = image_id
#
#         # add metadata
#         metadata_writer.add(patch_metadata)



# def run_patch_creation(zarr_ds, metadata_writer, down_sample_res,
#                  indices, ds_merge, target_km_res):
#     """
#     Patch extraction
#
#     Parameters
#     ----------
#     zarr_ds : ZarrDataset
#         Shared Zarr dataset writer.
#     metadata_writer : MetadataWriter
#         Shared metadata writer.
#     down_sample_res : int
#         Output resolution in pixels.
#     indices : array-like
#         Sampled grid indices to process.
#     ds_merge : xarray.Dataset
#         Dataset containing data fields.
#     target_km_res : float
#         Target physical resolution.
#
#     """
#
#     # split indices evenly among workers
#     # num_workers = min(num_workers, len(indices)) # ensure no empty splits
#     # indices_split = np.array_split(indices, num_workers)
#
#     logging.info(f"Starting  patch creation")
#
#     # with ThreadPoolExecutor(max_workers=num_workers) as executor:
#     #     futures = []
#     #     for wid in range(num_workers):
#     #         futures.append(
#     #             executor.submit(
#     #                 worker_task,
#     #                 zarr_ds, metadata_writer,
#     #                 wid, down_sample_res,
#     #                 indices_split[wid], ds_merge,
#     #                 target_km_res
#     #             )
#     #         )
#     #     # wait for all workers to finish
#     #     for f in futures:
#     #         f.result()
#
#     # 1 worker task for now
#     worker_task(zarr_ds, metadata_writer, down_sample_res,
#                     indices, ds_merge,
#                     target_km_res)
#
#     metadata_writer.close()
#
#     logging.info(f"Finished patch creation of {len(indices)} indices")

def process_time_snapshot(it, args, metadata_writer, zarr_ds, num_workers, ds_merge, grid, land_face_mask):
    """
    Process a single LLC time snapshot.

    This includes:
      - loading raw data
      - preprocessing and masking
      - gradient computation
      - weighted sampling
      - parallel patch extraction

    Parameters
    ----------
    it : int
        LLC iteration number.
    face_range : iterable
        Faces to load.
    args : argparse.Namespace
        Parsed command-line arguments.
    metadata_writer : MetadataWriter
        Metadata writer instance.
    zarr_ds : ZarrDataset
        Zarr dataset writer.
    num_workers : int
        Number of parallel workers.
    """

    # NOTE The ordering of the following steps matters

    # Calculate Ice Mask
    logging.info(f"Calculating ice mask")
    ice_mask = ~(ds_merge.Theta <= 0.0)
    ice_mask_np = ice_mask.values
    merged_mask = ice_mask_np & land_face_mask
    # merged_mask = merged_mask.values

    # calculate gradients
    # ds_merge = dbof_front_training_utils.calculate_gradients(ds_merge, grid)
    ds_merge, log_gradb = dbof_front_training_utils.calculate_gradients(ds_merge, grid)

    # U = ds_merge["U"].persist()
    # V = ds_merge["V"].persist()
    # U_interp = grid.interp(U, "X", boundary="fill")
    # V_interp = grid.interp(V, "Y", boundary="fill")
    # ds_merge = ds_merge.assign(
    #     U=U_interp,
    #     V=V_interp,
    # )

    # find indices with mask
    logging.info(f"Sampling patch center points")
    # indices = weighted_coordinate_sampling.weighted_sample_on_grid(args.sample_points_per_snapshot, args.bias_to_high_gradients,
    #                                                                ds_merge["log_gradb"], merged_mask)

    # todo this should be updated to take in a numpy array since we compute loggradb later anyway
    indices = weighted_coordinate_sampling.weighted_sample_on_grid(args.sample_points_per_snapshot, args.bias_to_high_gradients,
                                                                   log_gradb, merged_mask)

    # Move non tracer values to tracer points. This allows us to stack images for our final patches.
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    # grow zarr ds to fit at most len of indices
    zarr_ds.grow_array(len(indices))

    # create our patches
    # compute gradient into memory
    log_gradb_np = log_gradb.values

    #process data and write to s3
    images = dbof_front_training_utils.run_patch_creation(zarr_ds, metadata_writer, args.down_sample_res,
                                                          indices, ds_merge, log_gradb_np, args.target_km_res,
                                                          feature_channels_lazy, metadata_cols)

    # flush metada
    metadata_writer.close()


    # run_patch_creation(zarr_ds, metadata_writer, args.down_sample_res,
    #              indices, ds_merge, args.target_km_res)
    #

    # for dask
    ds_merge = None
    del ds_merge

    grid = None
    del grid

    merged_mask = None
    del merged_mask

def generate_land_face_masks(ds_grid, target_km_res):
    """
    Construct a composite sampling mask for the LLC native grid. These are the unchanging masks, so not ice.

    The mask excludes:
      - land points (via hFacC)
      - grid-face perimeter cells
    and applies a halo buffer based on the target physical resolution to land and face perimeter cells.

    Parameters
    ----------
    ds_grid : xarray.Dataset
        LLC grid dataset containing metric terms.
    target_km_res : float
        Target physical resolution (km) used to define halo width.

    Returns
    -------
    xarray.DataArray
        Boolean mask indicating valid sampling locations.
        True = sample
        False = mask
    """

    halo_km = target_km_res  # buffer to account for mean usage

    DXC = ds_grid["dxC"].persist()
    DYC = ds_grid["dyC"].persist()
    land_mask = (ds_grid.hFacC == 0).persist()

    halo_land_mask = halo_mask.llc_halo_mask(
        mask=land_mask,
        dxC=DXC,
        dyC=DYC,
        halo_km=halo_km
    )

    faces_perimeter_mask = xr.zeros_like(ds_grid.XC).astype(bool)
    faces_perimeter_mask.loc[dict(j=0)] = True
    faces_perimeter_mask.loc[dict(j=(faces_perimeter_mask.coords.sizes["j"] - 1))] = True
    faces_perimeter_mask.loc[dict(i=0)] = True
    faces_perimeter_mask.loc[dict(i=(faces_perimeter_mask.coords.sizes["i"] - 1))] = True

    halo_faces_perimeter_mask = halo_mask.llc_halo_mask(
        mask=faces_perimeter_mask,
        dxC=DXC,
        dyC=DYC,
        halo_km=halo_km
    )

    merged_mask = halo_land_mask & halo_faces_perimeter_mask

    return merged_mask

def set_up_grid_data_and_masks(args):
    logging.info("Fetching grid file")
    co = get_raw_data.get_remote_gridfile(endpoint_url)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    logging.info("Calculating land and face masks")
    land_face_mask = generate_land_face_masks(ds_grid, args.target_km_res)

    return ds_grid, land_face_mask

def generate_logging(args):
    # Logging -------------
    base = Path(__file__).resolve().parent # Relative to this script
    base_dir = base / "dbof_logs" # todo do we want to make this an input option?
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_name = Path(args.run_id).name
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    log_file = run_dir / "generate_front_training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

def calculate_iterations_for_llc(args):
    # Calculate iterations based on input
    iter_step = args.sampling_step * TS_PER_HOUR  # iteration Δ between samples
    start_iter = 10368 + args.start_record * TS_PER_HOUR

    if (args.timestep_hours is None):
        end_iter = 1495008  # to the end of data
    else:
        end_iter = start_iter + args.timestep_hours * TS_PER_HOUR

    return np.arange(start_iter, end_iter, iter_step)


def main():
    """
    Entry point for native-grid LLC patch dataset generation.

    Orchestrates argument parsing, Dask setup, filesystem initialization,
    and iteration over time snapshots.
    """

    # Script arguments
    p = argparse.ArgumentParser()
    args = parse_args(p)

    generate_logging(args)

    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    # Set concurrency for zarr ds writes
    zarr.config.set({'async.concurrency': 128})

    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores
    logging.info(f"Dask Client {dask_client}")

    iter_range = calculate_iterations_for_llc(args)
    logging.info(f"Processing: {iter_range} time snapshots")

    # Set up meta and zarr data writers
    fs, fs_synch = generate_filesystems(args.s3_endpoint)
    metadata_writer = generate_metadata_writer(args.bucket, args.folder, args.run_id, fs_synch)

    # Zarr Dataset
    dataset_name = f"dataset.zarr"
    zarr_ds = zarr_dataset.ZarrDataset(args.bucket, args.folder, args.run_id, dataset_name, fs=fs,
                                       feature_channels=feature_channels_lazy+feature_channels_computed,
                                       down_sample_res=args.down_sample_res)

    logging.info(f"Zarr dataset created.")

    # Get data grid once
    ds_grid, land_face_mask = set_up_grid_data_and_masks(args)
    grid = xgcm.Grid(ds_grid, periodic=False)

    for it in tqdm.tqdm(iter_range):
        # grab raw data for this iteration
        ds = get_raw_data.get_remote_llc_data(endpoint_url, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)

        logging.info(f"Data loaded for iteration: {it}")
        process_time_snapshot(it, args, metadata_writer, zarr_ds, args.num_workers, ds_merge, grid, land_face_mask)

        # For dask
        ds_merge = None
        del ds_merge

        ds = None
        del ds


if __name__ == "__main__":
    main()