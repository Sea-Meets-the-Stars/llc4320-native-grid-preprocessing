"""
Shared pipeline base for all global LLC4320 generate scripts.

This module contains everything that is common to generate_fronts_global,
generate_properties_global, and generate_frontogenesis_global:
  - LLC4320 model constants
  - Logging setup
  - ISO-format date → iteration conversion
  - Grid / land-mask setup
  - The main ``process_time_snapshot`` workhorse (takes a ``compute_fields_fn``
    callback for the mode-specific computed-field logic)
  - The ``run_global_pipeline`` orchestrator (called by each script's ``main()``)

Usage in a concrete script
--------------------------
    from dbof.cli._generate_global_base import run_global_pipeline

    def _compute_my_fields(ds_merge, grid, computed_feature_channels):
        # return a dict of {channel_name: DataArray/ndarray}
        ...

    def main(config_file: str = None, run_id: str = None):
        run_global_pipeline(config_file, run_id, _compute_my_fields)
"""

# stdlib
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# numerical / compute
import numpy as np
import zarr
import xarray as xr

# distributed / IO
from dask.distributed import Client

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
from dbof.llc4320_ingestion import grid as llc_grid

import dbof.dataset_creation.zarr_dataset_global as zarr_dataset
import dbof.dataset_creation.config as config

import dbof.utils.faces_to_latlon as faces_to_latlon


# ---------------------------------------------------------------------------
# LLC4320 model constants
# NOTE: these are constants for the LLC 4320 model.  If we look to support
# other models in the future this will need to be updated or configurable.
# ---------------------------------------------------------------------------
TS_PER_HOUR            = 144          # model cadence: 25 s → 144 steps/hr
MAX_ITER               = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368
LLC_FACES              = range(13)

# LLC4320 calendar reference.
# Iteration 0 corresponds to 2011-09-13 00:00:00 UTC; each step is 25 seconds.
# Used to convert human-readable dates → iteration numbers.
LLC4320_START_DATE     = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS  = 25           # seconds per model step

# ISO 8601-style format: 'YYYY-MM-DD HH:MM:SS'  (e.g. '2012-09-11 12:00:00')
DATE_FMT               = '%Y-%m-%d %H:%M:%S'

# URL of the raw LLC4320 data store
ENDPOINT_URL           = 'https://mghp.osn.xsede.org'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_logging(cfg: config.JobConfig) -> None:
    """Configure file + stdout logging for a pipeline run."""
    log_root = Path(cfg.run.log_dir).expanduser().resolve()
    run_dir  = log_root / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "generate_global.log"

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
    Convert a date string in 'YYYY-MM-DD HH:MM:SS' format to an LLC4320
    iteration number.

    The LLC4320 model starts at 2011-09-13 00:00:00 UTC (iteration 0) with a
    25-second timestep.  The returned iteration is rounded to the nearest step.

    Examples
    --------
    _date_to_iteration('2011-09-13 00:00:00')  ->  0
    _date_to_iteration('2012-01-01 00:00:00')  ->  ~1,011,456
    _date_to_iteration('2012-09-11 12:00:00')  ->  ~1,463,616
    """
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start ({LLC4320_START_DATE.date()}). "
            f"Expected format: YYYY-MM-DD HH:MM:SS  (e.g. '2011-09-13 00:00:00')."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def calculate_iterations_for_llc(cfg: config.JobConfig) -> np.ndarray:
    """
    Return the list of LLC4320 iteration numbers to process.

    Two modes, in priority order:

    1. **Date list** (``cfg.data.date_iterations`` is set in the YAML, e.g.
       ``date_iterations: ['2012-09-11 12:00:00', '2012-10-01 00:00:00']``):
       Each date string is converted to the nearest LLC4320 iteration number
       using the model's 25-second timestep and start date (2011-09-13 00:00 UTC).

    2. **Range mode** (default, backwards-compatible):
       A uniformly-spaced range derived from ``start_record``,
       ``sampling_step``, and ``timestep_hours``.  If ``timestep_hours`` is
       ``None`` the range runs to ``MAX_ITER``.
    """
    if cfg.data.date_iterations is not None:
        iterations = [_date_to_iteration(d) for d in cfg.data.date_iterations]
        logging.info(
            "Using date-derived iteration list: "
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
        If ``True``, expand the land mask outward by a halo.  This is useful
        for the cutout pipeline where patch centres must be far from land,
        but is generally not needed for global output (it would NaN out
        coastal ocean in the image).
        If ``False``, the raw land mask (``hFacC == 0``) is used with no
        expansion.
    """
    logging.info("Fetching grid file")
    co      = get_raw_data.get_remote_gridfile(cfg.data.endpoint_url)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    logging.info(f"Calculating land mask (use_halo={use_halo})")
    if use_halo:
        land_mask = native_grid_masks.generate_static_land_mask_for_sampling(
            ds_grid, cfg.output.target_km_res
        )
    else:
        land_mask = (ds_grid.hFacC == 0).values  # raw land mask, no coastal buffer

    return ds_grid, land_mask


# ---------------------------------------------------------------------------
# Core snapshot processor
# ---------------------------------------------------------------------------

def process_time_snapshot(
    cfg: config.JobConfig,
    zarr_ds,
    ds,
    ds_merge,
    grid,
    land_mask,
    model_feature_channels: list,
    computed_feature_channels: list,
    it: int,
    compute_fields_fn,
) -> None:
    """
    Process one time snapshot and write it to the zarr store.

    Parameters
    ----------
    cfg, zarr_ds, ds, ds_merge, grid, land_mask :
        Standard pipeline objects — see ``run_global_pipeline``.
    model_feature_channels :
        Raw model fields to include in the output (e.g. ``['Theta', 'Salt']``).
    computed_feature_channels :
        Names of mode-specific derived fields (e.g. ``['relative_vorticity']``).
        ``'gradb2'`` is always appended automatically and must NOT appear here.
    it : int
        LLC4320 iteration number (used only for logging).
    compute_fields_fn : callable
        ``(ds_merge, grid, computed_feature_channels) -> dict``
        Returns a mapping of ``{channel_name: DataArray | ndarray}`` for all
        channels listed in ``computed_feature_channels``.

    Notes
    -----
    The ordering of operations inside this function matters:
    1. ``gradb2`` is computed first (needs staggered U/V).
    2. ``compute_fields_fn`` is called (may also need staggered U/V).
    3. U and V are interpolated to tracer points.
    4. ``gradb2`` is materialised into a NumPy array.
    5. All channels are stitched face→latlon and written to zarr.
    """
    
    # gradb2 must always be computed (used for front finding)
    gradb2 = calculate_additional_fields.grad_b2(ds_merge, grid)

    # --- Computed (mode-specific) fields ---
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # Move non-tracer values to tracer points so all channels share the same
    # (face, j, i) grid before the face→latlon stitch.
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    # Materialise gradb2 into memory before we lose the ds_merge reference.
    # This must remain a plain .values call — do not modify.
    #gradb2_np = gradb2.values  # protected line do not modify
    #calculated_fields["gradb2_np"] = gradb2_np

    # --- Stitch LLC faces (face, j, i) → 2D lat/lon (lat, lon) ---
    #
    # Wrap the materialised gradb2 back as a DataArray so it can be passed to
    # faces_dataset_to_latlon alongside the other xarray variables.
    #gradb2_da = xr.DataArray(
    #    gradb2_np,
    #    dims=ds_merge['Theta'].dims,
    #    coords=ds_merge['Theta'].coords,
    #    name='gradb2',
    #)

    # Assemble all channels into a single Dataset for a single conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels #+ ['gradb2']
    update_vars = (
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]

    # metric_vector_pairs tells faces_dataset_to_latlon to rotate the direction
    # of vector fields (U, V) when Arctic-cap faces are transposed.
    has_uv = ('U' in model_feature_channels and 'V' in model_feature_channels)
    metric_vector_pairs = [('U', 'V')] if has_uv else []

    logging.info("Converting from LLC faces to rectangular lat/lon...")
    ds_rect = faces_to_latlon.faces_dataset_to_latlon(
        ds_to_convert,
        metric_vector_pairs=metric_vector_pairs,
    )

    # Extract channels in a consistent order and stack into (C, H, W).
    logging.info("Extracting channels and stacking into (C, H, W) format")
    channel_arrays = [ds_rect[ch].values for ch in channels_to_convert]
    data = np.stack(channel_arrays, axis=0)   # shape: (C, compact_h, compact_w)

    # Write to zarr store.
    logging.info("Writing snapshot to zarr dataset")
    zarr_ds.write_snapshot(data, it)

    # Release references so Dask can reclaim worker memory.
    ds_merge = None
    del ds_merge
    grid = None
    del grid


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_global_pipeline(
    config_file: str = None,
    run_id: str = None,
    compute_fields_fn = None,
    cfg: config.JobConfig = None,
) -> None:
    """
    Main orchestration loop shared by all global generate scripts.

    Parameters
    ----------
    config_file : str, optional
        Path to the YAML config file.  If ``None`` and ``cfg`` is also
        ``None``, the value is read from ``--config`` on the command line
        via ``config.parse_args()``.  Ignored when ``cfg`` is provided.
    run_id : str, optional
        Run-id override (takes precedence over the value in the YAML or
        the provided ``cfg``).
        If ``None`` and called from the CLI, ``--run_id`` is used if provided.
    compute_fields_fn : callable
        ``(ds_merge, grid, computed_feature_channels) -> dict``
        Mode-specific field computation.  See ``process_time_snapshot``.
    cfg : config.JobConfig, optional
        A fully-constructed ``JobConfig`` object.  When supplied, the
        ``config_file`` argument is ignored and no YAML file is read.
        Useful for callers that construct the config in memory (e.g.
        ``generate_combined_global``) to avoid writing a temporary file.
    """
    if cfg is None:
        if config_file is None:
            cli = config.parse_args()
            config_file = cli.config
            run_id = run_id or cli.run_id
        cfg = config.load_config(config_file)

    # override run_id if supplied by the caller
    if run_id is not None:
        cfg = config.JobConfig(
            run=config.RunConfig(run_id=run_id, log_dir=cfg.run.log_dir),
            data=cfg.data,
            sampling=cfg.sampling,
            output=cfg.output,
            features=cfg.features,
            runtime=cfg.runtime,
        )

    generate_logging(cfg)
    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    model_feature_channels    = [c.strip() for c in cfg.features.model_data_feature_channels    if c.strip()]
    computed_feature_channels = [c.strip() for c in cfg.features.compute_features_channels if c.strip()]

    # Set zarr async concurrency
    zarr.config.set({'async.concurrency': cfg.runtime.zarr_async_concurrency})

    # Start Dask distributed client (uses all local cores by default)
    dask_client = Client()
    logging.info(f"Dask Client {dask_client}")

    iter_range = calculate_iterations_for_llc(cfg)
    logging.info(f"Processing: {iter_range} time snapshots")

    # Set up S3 filesystem
    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)

    # Load the LLC4320 grid and land mask once — they never change across time.
    # use_halo=False: no coastal buffer needed for global rectangular output.
    ds_grid, land_mask = set_up_grid_data_and_masks(cfg, use_halo=False)
    grid = llc_grid.set_xgcm_grid(ds_grid, use_connections=True)

    # LLC4320 rectangular output shape is a model constant: 3×4320 rows, 4×4320 cols.
    rectangular_shape = (3 * 4320, 4 * 4320)   # (12960, 17280)
    logging.info(f"LLC rectangular output shape: {rectangular_shape}")

    # Construct the zarr output store
    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        channel_names=model_feature_channels + computed_feature_channels + ["gradb2"],
        rectangular_shape=rectangular_shape,
    )
    logging.info("Zarr dataset created.")

    for it in tqdm.tqdm(iter_range):
        # Fetch raw LLC4320 data for this iteration from S3/OSN
        ds       = get_raw_data.get_remote_llc_data(ENDPOINT_URL, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        logging.info(f"Data loaded for iteration: {it}")

        process_time_snapshot(
            cfg,
            zarr_ds,
            ds,        # raw kerchunk dataset — preserves LLC4320 topology for faces_to_latlon
            ds_merge,
            grid,
            land_mask,
            model_feature_channels,
            computed_feature_channels,
            it,
            compute_fields_fn,
        )

        ds_merge = None
        del ds_merge
        
        ds = None
        del ds
