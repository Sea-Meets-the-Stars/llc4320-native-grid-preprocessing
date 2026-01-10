import argparse
import numpy as np
import cmocean
import xgcm
import matplotlib.colors as colors
import ecco_v4_py as ecco
import xmitgcm.llcreader as llcreader

import dask.array as da
from dask.distributed import Client

import tqdm

import xarray as xr
import pandas as pd

import uuid

import zarr
import fsspec





# Internal Modules
import data_ingestion.get_raw_data as get_raw_data
import utils.native_gradient as ng
import plotting.llc_plotting as llc_plotting

import data_preprocessing.preproc_llc_core_data as preproc_llc_core_data
import utils.physical_calculations as physical_calculations

import data_preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling

import data_preprocessing.halo_mask as halo_mask
import data_preprocessing.spatial_patches as spatial_patches
import dataset_creation.metadata as metadata



def parse_args(p):

    #data selection-----
    p.add_argument("--sampling_step", required=True,
                   help="Stride in timesteps")

    p.add_argument("--start_record", default=1180,
                   help="Starting record. Default is first record with wind forcing.")

    p.add_argument("--timestep_hours", default=None,
                   help="How many total hours to load between start iteration and end iteration."
                        "If not given, the script will proces at provided sampling_step until the end of the data.")

    #sampling options--------
    p.add_argument("--bias_to_high_gradients", default=2,
                   help="Bias to high B in sampled data")

    p.add_argument("--sample_points_per_snapshot", default=100,
                   help="Bias to high B in sampled data")

    #s3 config
    p.add_argument("--s3_endpoint", default="https://s3-west.nrp-nautilus.io",
                   help="nrp s3 endpoing. Likely leave default.")

    p.add_argument("--bucket", default="llc/",
                   help="NRP s3 bucket to save data")

    p.add_argument("--folder", default="native_grid_dbof_training_data/",
                   help="NRP s3 bucket to save data")

    p.add_argument("--bucket", required=True,
                   help="Ensure run Id is unique inside s3://bucket/folder/ ")



    s3_endpoint = "https://s3-west.nrp-nautilus.io"

    #return data-------
    p.add_argument("--target_km_res", default=150,
                   help="Target physical resolution in km. Default is 150.")

    p.add_argument("--down_sample_res", default=64,
                   help="Downsampling resolution of W and H in pixels. Default is 64.")

    args = p.parse_args()
    return args

def generate_masks(ds_merge, ds_grid, target_km_res):
    halo_km = target_km_res  # buffer to account for mean usage

    ice_mask = ds_merge.Theta <= 0.0

    halo_land_mask = halo_mask.llc_halo_mask(
        mask=ds_grid.hFacC == 0,
        dxC=ds_grid["dxC"],
        dyC=ds_grid["dyC"],
        halo_km=halo_km
    )

    # halo_ice_mask = halo_mask.llc_halo_mask(
    #     mask=ice_mask,
    #     dxC=ds_grid["dxC"],
    #     dyC=ds_grid["dyC"],
    #     halo_km=halo_km
    # )
    halo_ice_mask = ice_mask #ice mask is already very aggressive. No need for halo.

    faces_perimeter_mask = xr.zeros_like(ds_merge.Theta).astype(bool)
    faces_perimeter_mask.loc[dict(j=0)] = True
    faces_perimeter_mask.loc[dict(j=(faces_perimeter_mask.coords.sizes["j"] - 1))] = True
    faces_perimeter_mask.loc[dict(i=0)] = True
    faces_perimeter_mask.loc[dict(i=(faces_perimeter_mask.coords.sizes["i"] - 1))] = True

    halo_faces_perimeter_mask = halo_mask.llc_halo_mask(
        mask=faces_perimeter_mask,
        dxC=ds_grid["dxC"],
        dyC=ds_grid["dyC"],
        halo_km=halo_km
    )

    merged_mask = halo_ice_mask & halo_land_mask & halo_faces_perimeter_mask

    return merged_mask

def caculate_gradients(ds_merge, ds_grid, grid):
    buoyancy = physical_calculations.buoyancy_of_field()

    # gradient of b
    zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_grid, grid=grid)

    zonal_grad_b = zonal_grad_b.persist()
    merid_grad_b = merid_grad_b.persist()

    gradb2 = physical_calculations.grad_squared(zonal_grad_b, merid_grad_b)
    gradb2 = gradb2.persist()

    log_gradb = da.log10(gradb2)
    log_gradb.persist()

    log_gradb_ds = log_gradb.to_dataset(name="log_gradb")
    ds_merge = xr.merge([ds_merge, log_gradb_ds])

    return ds_merge, log_gradb

def generate_filesystems(s3_endpoint):
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
    meda_data_file_path = "s3://" + bucket + folder + run_id + "metadata/metadata1.parquet"

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

    return metadata.MetadataWriter(meda_data_file_path, flush_every=10000, fs=fs_synch)

def loop_time_snapshots(it, face_range, args, metadata_writer): # target_km_res, bias_to_high_gradients, sample_points_per_snapshot, s3_endpoint):
    endpoint_url = 'https://mghp.osn.xsede.org'

    # grab raw data
    co = get_raw_data.get_remote_gridfile(endpoint_url)
    ds = get_raw_data.get_remote_llc_data(endpoint_url, it, face_range)
    ds_merge, ds_grid = preproc_llc_core_data.process_llc4320(ds, co)

    grid = xgcm.Grid(ds_grid, periodic=False)

    # Halo mask for sample points
    merged_mask = generate_masks(ds_merge, ds_grid, args.target_km_res)

    ds_merge, log_gradb = caculate_gradients(ds_merge, ds_grid, grid)

    merged_mask = xr.DataArray(merged_mask)
    indices = weighted_coordinate_sampling.weighted_sample_on_grid(args.sample_points_per_snapshot, args.bias_to_high_gradients,
                                                                   log_gradb, merged_mask)

    # Move non tracer values to tracer points
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')




def main():
    p = argparse.ArgumentParser()
    args = parse_args(p)


    feature_channels = ["Eta", "Salt", "Theta", "U", "V", "W", "log_gradb"]


    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores

    ts_per_hour = 144  # model cadence: 25 s → 144 steps/hr
    iter_step = args.sampling_step * ts_per_hour  # iteration Δ between samples
    face_range = range(13)

    # NOTE:
    # MAX iteration : 1495008
    # First valid wind/forcing record begins ~1180

    # Compute iter numbers
    start_iter = 10368 + args.start_record * ts_per_hour

    if (args.timestep_hours is None):
        end_iter = 1495008 #to the end of data
    else :
        end_iter = start_iter + args.timestep_hours * ts_per_hour

    iter_range = np.arange(start_iter, end_iter, iter_step)

    # Set up data writers
    fs, fs_synch = generate_filesystems(args.s3_endpoint)
    metadata_writer = generate_metadata_writer(args.bucket, args.folder, args.run_id, fs_synch)

    for it in tqdm.tqdm(iter_range):
        loop_time_snapshots(it, face_range, args, metadata_writer)

                            # args.target_km_res, args.bias_to_high_gradients, args.sample_points_per_snapshot,
                            # args.s3_endpoint)

