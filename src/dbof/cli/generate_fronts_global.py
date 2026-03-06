# TODO METAdata has extra id column unused. Get rid of it.

# stdlib
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# numerical / compute
import numpy as np
import xgcm
import zarr

# grid / face-to-latlon stitching
import xarray as xr
from xmitgcm.llcreader import llcmodel
from xmitgcm.llcreader.llcmodel import (
    _faces_coords_to_latlon,
    _faces_to_latlon_scalar,
    _faces_to_latlon_vector,
    _drop_facedim,
)

# distributed / IO
from dask.distributed import Client

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
from dbof.preprocessing import ice_mask as ice_masking

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
from dbof.llc4320_ingestion import grid as llc_grid

import dbof.dataset_creation.zarr_dataset_global as zarr_dataset

import dbof.dataset_creation.config as config

from IPython import embed

# Constants --------------------------
# NOTE these are constants for the LLC 4320 model. If we look to support other models in the future
# this will need to be updated or configurable.
TS_PER_HOUR = 144 # model cadence: 25 s → 144 steps/hr
MAX_ITER = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368
LLC_FACES = range(13)
# LLC_NATIVE_GRID_DEG = 1 / 48   # only needed if using resample_to_latlon; not used here

# LLC4320 calendar reference.
# Iteration 0 corresponds to 2011-09-13 00:00:00 UTC; each step is 25 seconds.
# Used to convert human-readable dates ('DDMMYYYY-HH:MM:SS') → iteration numbers.
LLC4320_START_DATE      = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS   = 25          # seconds per model step
DATE_FMT                = '%d%m%Y-%H:%M:%S'   # expected format: DDMMYYYY-HH:MM:SS

# url of our raw data - this may need to be an input in the future
endpoint_url = 'https://mghp.osn.xsede.org'


def _faces_dataset_to_latlon(ds, metric_vector_pairs):
    """
    This function is based on xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon
    but has been updated to be compatible with all xarray versions.

    The upstream function contains:
        ds_new = ds_new.update(data_vars)      # reassigns to None in xarray < 0.17
        ds_new = ds_new.set_coords(...)        # AttributeError: 'NoneType'...
    because Dataset.update() was in-place (returning None) in xarray < 0.17.
    This version uses xr.merge() instead, which has stable semantics across all
    xarray versions.
    """
    coord_vars = list(ds.coords)
    ds_new = _faces_coords_to_latlon(ds)    # skeleton Dataset with expanded coords
    nfaces = len(ds['face'])

    # Classify variables as scalars or vector pairs (same logic as upstream).
    vector_pairs = []
    vnames = list(ds.reset_coords().variables)
    for vname in list(vnames):
        try:
            mate = ds[vname].attrs['mate']
        except KeyError:
            mate = None
        if mate is not None:
            vector_pairs.append((vname, mate))
            try:
                vnames.remove(mate)
            except ValueError:
                raise ValueError(
                    f"If '{vname}' in varnames, '{mate}' must also be in varnames"
                )

    all_vector_components = [
        inner for outer in (vector_pairs + metric_vector_pairs) for inner in outer
    ]
    scalars = [v for v in vnames if v not in all_vector_components]

    data_vars = {}

    for vname in scalars:
        if vname == 'face' or vname in ds_new:
            continue
        if 'face' in ds[vname].dims:
            data = _faces_to_latlon_scalar(ds[vname].data, nfaces=nfaces)
            dims = _drop_facedim(ds[vname].dims)
        else:
            data = ds[vname].data
            dims = ds[vname].dims
        data_vars[vname] = xr.Variable(dims, data, ds[vname].attrs)

    for vname_u, vname_v in vector_pairs:
        u_data, v_data = _faces_to_latlon_vector(
            ds[vname_u].data, ds[vname_v].data, nfaces=nfaces
        )
        data_vars[vname_u] = xr.Variable(
            _drop_facedim(ds[vname_u].dims), u_data, ds[vname_u].attrs
        )
        data_vars[vname_v] = xr.Variable(
            _drop_facedim(ds[vname_v].dims), v_data, ds[vname_v].attrs
        )

    for vname_u, vname_v in metric_vector_pairs:
        u_data, v_data = _faces_to_latlon_vector(
            ds[vname_u].data, ds[vname_v].data, nfaces=nfaces, metric=True
        )
        data_vars[vname_u] = xr.Variable(
            _drop_facedim(ds[vname_u].dims), u_data, ds[vname_u].attrs
        )
        data_vars[vname_v] = xr.Variable(
            _drop_facedim(ds[vname_v].dims), v_data, ds[vname_v].attrs
        )

    # Use xr.merge instead of ds_new.update() to avoid the xarray < 0.17 issue
    # where Dataset.update() returned None rather than the modified dataset.
    ds_out = xr.merge([ds_new, xr.Dataset(data_vars)])
    ds_out = ds_out.set_coords([c for c in coord_vars if c in ds_out])
    return ds_out


def generate_logging(cfg: config.JobConfig):
    log_root = Path(cfg.run.log_dir).expanduser().resolve()
    run_dir = log_root / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "generate_front_training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

def _date_to_iteration(date_str: str) -> int:
    """
    Convert a date string in 'DDMMYYYY-HH:MM:SS' format to an LLC4320 iteration number.

    The LLC4320 model starts at 2011-09-13 00:00:00 UTC (iteration 0) with a
    25-second timestep. The returned iteration is rounded to the nearest step.

    Examples
    --------
    _date_to_iteration('13092011-00:00:00')  ->  0
    _date_to_iteration('01012012-00:00:00')  ->  ~1,011,456
    """
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start ({LLC4320_START_DATE.date()}). "
            "Check your DDMMYYYY-HH:MM:SS format — e.g. '13092011-00:00:00'."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def calculate_iterations_for_llc(cfg: config.JobConfig):
    """
    Return the list of LLC4320 iteration numbers to process.

    Three modes, in priority order:

    1. Date list  (cfg.data.date_iterations is set in YAML, e.g.
       ``date_iterations: ['01012012-00:00:00', '01042012-00:00:00']``):
       Each date string is converted to the nearest LLC4320 iteration number
       using the model's 25-second timestep and start date (2011-09-13 00:00 UTC).

    2. Range mode (default, backwards-compatible):
       A uniformly-spaced range is derived from start_record, sampling_step, and
       timestep_hours.  If timestep_hours is None the range runs to MAX_ITER.
    """
    if cfg.data.date_iterations is not None:
        iterations = [_date_to_iteration(d) for d in cfg.data.date_iterations]
        logging.info(
            f"Using date-derived iteration list: "
            + ", ".join(f"'{d}' → {it}" for d, it in zip(cfg.data.date_iterations, iterations))
        )
        return np.array(iterations, dtype=int)

    # Range mode: convert hours → model iteration numbers
    iter_step  = cfg.data.sampling_step * TS_PER_HOUR
    start_iter = FIRST_WIND_RECORD_OFFSET + cfg.data.start_record * TS_PER_HOUR
    end_iter   = MAX_ITER if cfg.data.timestep_hours is None \
                 else start_iter + cfg.data.timestep_hours * TS_PER_HOUR

    return np.arange(start_iter, end_iter, iter_step)

def set_up_grid_data_and_masks(cfg: config.JobConfig, use_halo: bool = False):
    """
    Load the LLC4320 grid and compute a static land mask.

    Parameters
    ----------
    cfg : JobConfig
    use_halo : bool, default False
        If True, expand the land mask outward by a halo.
            This is useful for the cutout pipeline where patch centres must be far from land, 
            but is generally not needed for global output (it would NaN out coastal ocean in the image).
        If False, the raw land mask (hFacC == 0) is used with no expansion.
    """
    logging.info("Fetching grid file")
    co = get_raw_data.get_remote_gridfile(cfg.data.endpoint_url)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    logging.info(f"Calculating land mask (use_halo={use_halo})")
    if use_halo:
        land_mask = native_grid_masks.generate_static_land_mask_for_sampling(
            ds_grid, cfg.output.target_km_res
        )
    else:
        land_mask = (ds_grid.hFacC == 0).values  # raw land mask, no coastal buffer

    return ds_grid, land_mask

def process_time_snapshot(
    cfg: config.JobConfig,
    zarr_ds,
    ds,
    ds_merge,
    grid,
    land_mask,
    model_feature_channels,
    computed_feature_channels,
    it,
):
    # NOTE The ordering of the following steps matters

    # --- Calculate Ice Mask ---
    logging.info(f"Calculating ice mask")
    #ice_mask = ~(ds_merge.Theta <= 0.0)
    # TODO -- TURN THIS BACK ON
    '''
    ice_mask = ice_masking.mask_by_theta(ds_merge)
    ice_mask_np = ice_mask.values
    merged_mask = ice_mask_np & land_mask
    '''

    # --- Calculated Fields ---
    calculated_fields = {}

    # This must be included for the front finding
    embed(header='276 of process_time_snapshot')
    gradb2 = calculate_additional_fields.grad_b2(ds_merge, grid)

    #if "relative_vorticity" in computed_feature_channels:
    #    relative_vorticity = calculate_additional_fields.relative_vorticity(ds_merge, grid)
    #    calculated_fields["relative_vorticity"] = relative_vorticity


    # Move non tracer values to tracer points. This allows us to stack images for our final patches.
    # TODO -- Could use original grid for this, as a hack of sorts
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    #vec_u_to_ij = grid.interp_2d_vector({'X': ds_merge['U'], 'Y': ds_merge['V']},boundary='fill')
    #vec_u_to_ij = grid.interp_2d_vector({'X': u_x, 'Y': v_y},boundary='fill')



    ## 
    #Here we compute the calculated gradients into memory before creating our patches. 
    #While this is arguably inefficient if we do not do this our Dask graph splits and we will run into difficult errors
    #or warnings to fix. 
    #The cause of this is either xmitgcm code calculating the gradients on the native grid or that we are using 
    #the gradient in our sampling logic. I believe it is the first but I am not sure yet. - Jake 
    #
    

    # Materialize gradb2 into memory before we lose the ds_merge reference. 
    gradb2_np = gradb2.values #protected line do not modify

    '''
    # Mask the edges as -999. where finite
    mask_val = -999.
    logging.info(f"Masking edges of gradb2 on each face with {mask_val} values")
    for face in range(gradb2_np.shape[0]):
        isf = np.isfinite(gradb2_np[face, 0, :])
        gradb2_np[face, 0, isf] = mask_val
        #
        isf = np.isfinite(gradb2_np[face, :, 0])
        gradb2_np[face, isf, 0] = mask_val
    '''

    calculated_fields["gradb2_np"] = gradb2_np


    #from IPython import embed
    #embed(header='288 of process_time_snapshot')

    # --- Stitch LLC faces (face, j, i) → 2D lat/lon (lat, lon) ---
    #
    # xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon() stitches the 13 LLC
    # faces into a single coherent 2D rectangular image. This is NOT interpolation
    # — values are pixel-shifted and some faces are rotated to tile correctly, but
    # we remain on the native LLC grid.
    # Input:  xr.Dataset with (face, j, i) dimensions, shape (13, 4320, 4320) per var
    # Output: xr.Dataset with (lat, lon) dimensions, shape (12960, 17280) per var

    # gradb2 is already a numpy array (materialised above via the protected
    # .values call). Wrap it back as a DataArray so it can be passed to
    # faces_dataset_to_latlon alongside the other xarray variables.
    gradb2_da = xr.DataArray(
        gradb2_np,
        dims=ds_merge['Theta'].dims,
        coords=ds_merge['Theta'].coords,
        name='gradb2',
    )

    # Assemble all channels into a single Dataset for a single conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels + ['gradb2']
    update_vars = (
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
        | {'gradb2': gradb2_da}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]

    # metric_vector_pairs tells faces_dataset_to_latlon to also rotate the
    # direction of vector fields (U, V) when Arctic-cap faces are transposed.
    has_uv = ('U' in model_feature_channels and 'V' in model_feature_channels)
    metric_vector_pairs = [('U', 'V')] if has_uv else []

    logging.info("Converting from LLC faces to rectangular lat/lon...")
    ds_rect = _faces_dataset_to_latlon(
        ds_to_convert,
        metric_vector_pairs=metric_vector_pairs,
    )

    # Extract channels in a consistent order and stack into (C, H, W).
    channel_arrays = [ds_rect[ch].values for ch in channels_to_convert]
    data = np.stack(channel_arrays, axis=0)   # shape: (C, compact_h, compact_w)

    # write to zarr ds
    logging.info("Writing snapshot to zarr dataset")
    zarr_ds.write_snapshot(data, it)

    # for dask
    ds_merge = None
    del ds_merge

    grid = None
    del grid

    merged_mask = None
    del merged_mask


def main(config_file: str):
    """
    Entry point for native-grid LLC dataset_creation generation.

    Orchestrates argument parsing, Dask setup, filesystem initialization,
    and iteration over time snapshots.
    """

    #cli = config.parse_args()
    cfg = config.load_config(config_file) #cli.config)

    # override run_id if passed in through cli
    #if cli.run_id is not None:
    #    cfg = config.JobConfig(
    #        run=config.RunConfig(run_id=cli.run_id, log_dir=cfg.run.log_dir),
    #        data=cfg.data,
    #        sampling=cfg.sampling,
    #        output=cfg.output,
    #        features=cfg.features,
    #        runtime=cfg.runtime,
    #    )

    generate_logging(cfg)

    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    model_feature_channels = [c.strip() for c in cfg.features.model_data_feature_channels if c.strip()]
    computed_feature_channels = [c.strip() for c in cfg.features.compute_features_channels if c.strip()]

    # Set concurrency for zarr ds writes
    zarr.config.set({'async.concurrency':  cfg.runtime.zarr_async_concurrency})

    # set up dask distributed client
    dask_client = Client()  # default: uses all local cores
    logging.info(f"Dask Client {dask_client}")

    iter_range = calculate_iterations_for_llc(cfg)
    logging.info(f"Processing: {iter_range} time snapshots")

    # Set up meta and zarr data writers
    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)

    # Get our grid and static masks once ever. These never change.
    # use_halo=False for global 
    ds_grid, land_mask = set_up_grid_data_and_masks(cfg, use_halo=False)

    #grid = xgcm.Grid(ds_grid, periodic=False)
    grid = xgcm.Grid(ds_grid,
                periodic=False,
                face_connections=llc_grid.face_connections,
        )

    # LLC4320 rectangular output shape is a model constant: 3×4320 rows, 4×4320 cols.
    # Because reset_coords() removed the face, i, and j coordinate values that 
    # faces_dataset_to_latlon needs to detect the LLC grid topology, 
    # the code cannot infer the output grid size from the dataset and 
    # instead we hardcode the known LLC4320 rectangular shape (3×4320 by 4×4320).
    rectangular_shape = (3 * 4320, 4 * 4320)   # (12960, 17280)
    logging.info(f"LLC rectangular output shape: {rectangular_shape}")

    # Construct GlobalZarrDataset.
    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        channel_names=model_feature_channels + computed_feature_channels + ["gradb2"],
        rectangular_shape=rectangular_shape,
    )

    logging.info(f"Zarr dataset created.")



    for it in tqdm.tqdm(iter_range):
        # grab raw data for this iteration
        ds = get_raw_data.get_remote_llc_data(endpoint_url, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        logging.info(f"Data loaded for iteration: {it}")

        # now process this iteration of data
        process_time_snapshot(
            cfg,
            zarr_ds,
            ds,        # raw kerchunk dataset — preserves LLC4320 topology for faces_dataset_to_latlon
            ds_merge,
            grid,
            land_mask,
            model_feature_channels,
            computed_feature_channels,
            it,
        )

        ds_merge = None
        del ds_merge

        ds = None
        del ds


#if __name__ == "__main__":
#    main()