"""
Global LLC4320 dataset generation.

Single-entry-point script for generating global LLC4320 property datasets.
The subset of properties to compute is selected at runtime via ``--subset``
(or the ``active_subset`` key in the YAML config).

Available subsets
-----------------
native_fields
    Raw model state variables only (Theta, Salt, Eta, U, V, W).
    No derived quantities are computed.

frontal_structure
    Scalar gradient magnitudes and the Turner angle characterising front
    intensity and water-mass structure: ``gradsalt2``, ``gradtheta2``,
    ``gradeta2``, ``gradrho2``, ``gradb2``, ``turner_angle``.
    The Turner angle reuses the gradient fields computed earlier in the
    same callback, so no gradient is evaluated twice.

kinematic
    Velocity-derived scalar fields computed from a single Jacobian pass:
    ``relative_vorticity``, ``strain_n``, ``strain_s``, ``strain_mag``,
    ``divergence``, ``coriolis_f``, ``rossby_number``, ``okubo_weiss``.

frontogenesis
    Kinematic frontogenesis tendency and geostrophic/ageostrophic decomposition:
    ``frontogenesis_tendency``, ``ug``, ``vg``,
    ``frontogenesis_geo``, ``frontogenesis_ageo``.

    *** Dask graph note ***
    This subset merges two large lazy lineages (velocity Jacobian gradients +
    tracer gradients for buoyancy and Eta).  To avoid the large-graph and
    run_spec scheduler warnings that appear when multiple frontogenesis arrays
    share the same lineage and are written together as lazy arrays, this
    callback materialises all selected fields with a *single* ``dask.compute()``
    call before returning them.  This fuses the shared subgraph in one scheduler
    round and returns NumPy arrays, so downstream zarr writes are decoupled from
    the Dask graph entirely.

CLI usage
---------
    generate-global \\
        --config configs/global_OSN.yaml \\
        --subset kinematic \\
        [--run_id my_run] \\
        [--no-icemask]

Config design
-------------
The YAML adds two top-level keys consumed here before the config object is
constructed:

    active_subset: kinematic       # default; overridden by --subset

    subsets:
      native_fields:
        dataset_name: "native_fields.zarr"
        model_data_feature_channels: [Theta, Salt, Eta, U, V, W]
        compute_features_channels: []
      frontal_structure:
        ...
      kinematic:
        ...
      frontogenesis:
        ...

Date format
-----------
All ``date_iterations`` entries in the YAML must use ISO format:
    'YYYY-MM-DD HH:MM:SS'  e.g. '2012-09-11 12:00:00'
"""

# stdlib
import sys
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

# numerical / compute
import numpy as np
import zarr
import xarray as xr

# distributed / IO
import dask
import yaml
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

from IPython import embed

# ---------------------------------------------------------------------------
# LLC4320 model constants
# ---------------------------------------------------------------------------
TS_PER_HOUR              = 144          # model cadence: 25 s → 144 steps/hr
MAX_ITER                 = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368
LLC_FACES                = range(13)

# LLC4320 calendar reference.
# Iteration 0 corresponds to 2011-09-13 00:00:00 UTC; each step is 25 seconds.
# Used to convert human-readable dates → iteration numbers.
LLC4320_START_DATE    = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS = 25           # seconds per model step

# ISO 8601-style format: 'YYYY-MM-DD HH:MM:SS'  (e.g. '2012-09-11 12:00:00')
DATE_FMT              = '%Y-%m-%d %H:%M:%S'

# URL of the raw LLC4320 data store
ENDPOINT_URL          = 'https://mghp.osn.xsede.org'


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
        land_mask = ds_grid.hFacC  # raw land mask, no coastal buffer

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
    apply_icemask: bool = True,
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
    it : int
        LLC4320 iteration number (used only for logging).
    compute_fields_fn : callable
        ``(ds_merge, grid, computed_feature_channels) -> dict``
        Returns a mapping of ``{channel_name: DataArray | ndarray}`` for all
        channels listed in ``computed_feature_channels``.
    apply_icemask : bool, default True
        When ``True`` (the default), pixels where ``Theta <= 0`` are treated as
        sea ice and set to NaN in the output.  Set to ``False`` to retain those
        values (e.g. when studying polar / sub-freezing surface waters).

    Notes
    -----
    The ordering of operations inside this function matters:
    1. ``compute_fields_fn`` is called (may also need staggered U/V).
    2. U and V are interpolated to tracer points.
    3. All channels are stitched face→latlon and written to zarr.
    """

    # --- Computed (mode-specific) fields ---
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # Move non-tracer values to tracer points so all channels share the same
    # (face, j, i) grid before the face→latlon stitch.
    ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    ds_merge["U"] = grid.interp(ds_merge["U"], 'X', boundary='fill')

    # Assemble all channels into a single Dataset for a single conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels
    update_vars = (
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]
    
    metric_vector_pairs = []
    if 'V' in ds_to_convert.variables:
        ds_to_convert['V'].attrs.pop('mate', None)
    if 'U' in ds_to_convert.variables:
        ds_to_convert['U'].attrs['mate'] = 'V'

    # land mask (always applied) + optional ice mask
    land_mask_da = (ds_merge.hFacC == 0)  # True where land (hFacC == 0)
    mask_vars = {'_land_mask': land_mask_da}
    if apply_icemask:
        logging.info("Calculating and applying ice mask (Theta <= 0) and land mask (hFacC == 0)")
        ice_mask_da = (ds_merge.Theta <= 0.0)  # True where ice
        mask_vars['_ice_mask'] = ice_mask_da
    else:
        logging.info("Calculating and applying land mask (hFacC == 0); ice mask disabled")

    ds_to_convert = ds_to_convert.assign(mask_vars)
    channels_to_convert_with_mask = channels_to_convert + list(mask_vars.keys())

    # stitch faces
    logging.info("Converting from LLC faces to rectangular lat/lon...")
    ds_rect = faces_to_latlon.faces_dataset_to_latlon(
        ds_to_convert[channels_to_convert_with_mask],
        metric_vector_pairs=metric_vector_pairs,
    )

    # Extract channels in a consistent order and stack into (C, H, W).
    logging.info("Extracting channels and stacking into (C, H, W) format")
    land_mask_rect = ds_rect['_land_mask'].values.astype(bool)  # shape: (H, W), True where land
    combined_mask = land_mask_rect
    if apply_icemask:
        ice_mask_rect = ds_rect['_ice_mask'].values.astype(bool)  # shape: (H, W)
        combined_mask = combined_mask | ice_mask_rect
    channel_arrays = [ds_rect[ch].values for ch in channels_to_convert]
    data = np.stack(channel_arrays, axis=0)   # shape: (C, compact_h, compact_w)
    data = np.where(combined_mask[np.newaxis], np.nan, data)

    # Write to zarr store.
    logging.info("Writing snapshot to zarr dataset")
    zarr_ds.write_snapshot(data, it)

    # Release references so Dask can reclaim worker memory.
    ds_merge = None
    del ds_merge
    grid = None
    del grid


# ---------------------------------------------------------------------------
# Per-subset compute callbacks
# ---------------------------------------------------------------------------

def _compute_native_fields(ds_merge, grid, computed_feature_channels: list) -> dict:
    """
    Compute callback for the ``native_fields`` subset.

    No derived quantities are computed here. These are raw
    model state variables specified in ``model_data_feature_channels`` in the
    config.

    Returns
    -------
    dict
        Always empty; present for interface consistency.
    """
    return {}


def _compute_frontal_structure_fields(
    ds_merge, grid, computed_feature_channels: list
) -> dict:
    """
    Compute callback for the ``frontal_structure`` subset.

    Computes scalar gradient-magnitude fields and the Turner angle.
    Gradient fields that the Turner angle depends on (``gradtheta2``,
    ``gradsalt2``, ``gradrho2``) are computed first and forwarded, so
    each gradient is only evaluated once.

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    computed_feature_channels : list of str

    Returns
    -------
    dict
        Mapping of ``{channel_name: DataArray}`` for each requested channel.
    """
    # --- gradient fields (computed first so turner_angle can reuse them) ---
    _GRAD_FNS = {
        "gradsalt2":  calculate_additional_fields.grad_salt2,
        "gradtheta2": calculate_additional_fields.grad_theta2,
        "gradeta2":   calculate_additional_fields.grad_eta2,
        "gradb2":     calculate_additional_fields.grad_b2,
        "gradrho2":   calculate_additional_fields.grad_rho2,
    }

    results = {
        name: fn(ds_merge, grid)
        for name, fn in _GRAD_FNS.items()
        if name in computed_feature_channels
    }

    # --- Turner angle (reuses already-computed gradients) ------------------
    results["turner_angle"] = calculate_additional_fields.turner_angle(
        ds_merge,
        grid,
        gradtheta2=results["gradtheta2"],
        gradsalt2=results["gradsalt2"],
        gradrho2=results["gradrho2"],
    )

    return results


def _compute_kinematic_fields(
    ds_merge, grid, computed_feature_channels: list
) -> dict:
    """
    Compute callback for the ``kinematic`` subset.

    All velocity-derived properties are obtained in a single Jacobian pass
    via ``all_velocity_properties``; only channels listed in
    ``computed_feature_channels`` are returned.

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    computed_feature_channels : list of str

    Returns
    -------
    dict
    """
    velocity_props = calculate_additional_fields.all_velocity_properties(ds_merge, grid)
    return {
        name: field
        for name, field in velocity_props.items()
        if name in computed_feature_channels
    }


def _compute_frontogenesis_fields(
    ds_merge, grid, computed_feature_channels: list
) -> dict:
    """
    Compute callback for the ``frontogenesis`` subset.

    Computes geostrophic velocities and geostrophic/ageostrophic frontogenesis
    via a single pass through ``all_frontogenesis_properties``.

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    computed_feature_channels : list of str

    Returns
    -------
    dict of str -> numpy.ndarray
        Materialised (not lazy) arrays for each requested channel.
    """
    props = calculate_additional_fields.all_frontogenesis_properties(ds_merge, grid)
    selected = {
        name: field
        for name, field in props.items()
        if name in computed_feature_channels
    }

    if not selected:
        return selected

    # Single dask.compute() call fuses the shared graph (Jacobian + tracer
    # gradients) into one scheduler submission, avoiding the run_spec warnings
    # that appear when multiple frontogenesis arrays are computed lazily later.
    # ** Claude suggestion **
    keys = list(selected.keys())
    materialised = dask.compute(*[selected[k] for k in keys])
    return dict(zip(keys, materialised))


# ---------------------------------------------------------------------------
# Subset registry: maps subset name → compute callback
# ---------------------------------------------------------------------------

SUBSET_COMPUTE_FNS = {
    "native_fields":     _compute_native_fields,
    "frontal_structure": _compute_frontal_structure_fields,
    "kinematic":         _compute_kinematic_fields,
    "frontogenesis":     _compute_frontogenesis_fields,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse --config, --run_id, and --subset from sys.argv."""
    parser = argparse.ArgumentParser(
        description=(
            "Global LLC4320 dataset generation. "
            "Select which property subset to compute with --subset."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Override the run_id defined in the config YAML.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        choices=list(SUBSET_COMPUTE_FNS),
        help=(
            "Property subset to compute. "
            f"One of: {', '.join(SUBSET_COMPUTE_FNS)}. "
            "If omitted, the value of 'active_subset' in the config YAML is used."
        ),
    )
    parser.add_argument(
        "--no-icemask",
        dest="apply_icemask",
        action="store_false",
        default=True,
        help=(
            "Disable the sea-ice mask (Theta <= 0).  By default the ice mask "
            "is applied and pixels where surface temperature is at or below "
            "freezing are set to NaN.  Pass --no-icemask to keep those values."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_global_pipeline(
    config_file: str = None,
    run_id: str = None,
    compute_fields_fn = None,
    cfg: config.JobConfig = None,
    apply_icemask: bool = True,
) -> None:
    """
    Main orchestration loop for global dataset generation.

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
        Subset-specific field computation.  See ``process_time_snapshot``.
    cfg : config.JobConfig, optional
        A fully-constructed ``JobConfig`` object.  When supplied,
        ``config_file`` is ignored.  Useful for callers that construct the
        config in memory (e.g. ``main()``) to avoid writing a temporary file.
    apply_icemask : bool, default True
        When ``True``, pixels where ``Theta <= 0`` are NaN-ed out as sea ice.
        Pass ``False`` to retain sub-freezing surface values.
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

    model_feature_channels    = [c.strip() for c in cfg.features.model_data_feature_channels if c.strip()]
    computed_feature_channels = [c.strip() for c in cfg.features.compute_features_channels   if c.strip()]

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
        channel_names=model_feature_channels + computed_feature_channels,
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
            apply_icemask=apply_icemask,
        )

        ds_merge = None
        del ds_merge

        ds = None
        del ds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    config_file: str = None,
    run_id: str = None,
    subset: str = None,
    apply_icemask: bool = None,
) -> None:
    """
    Entry point for the global dataset generation script.

    Can be called from the CLI (no arguments; reads ``--config``, ``--run_id``,
    ``--subset``, and ``--no-icemask`` from ``sys.argv``) or directly from
    Python by passing the arguments explicitly.

    Parameters
    ----------
    config_file : str, optional
        Path to the YAML config.  If ``None``, ``--config`` is read from
        ``sys.argv``.
    run_id : str, optional
        Override for the run identifier.  If ``None`` and called from the CLI,
        ``--run_id`` is used if provided.
    subset : str, optional
        One of the keys in ``SUBSET_COMPUTE_FNS``.  If ``None``, falls back to
        ``--subset`` from the CLI, then to the ``active_subset`` key in the
        YAML config.
    apply_icemask : bool or None, optional
        Whether to NaN-out pixels where ``Theta <= 0`` (sea-ice mask).
        ``True``  — ice mask on  (default when called from the CLI).
        ``False`` — ice mask off (equivalent to passing ``--no-icemask``).
        ``None``  — read from the CLI flag (``--no-icemask``); defaults to
                    ``True`` if not passed on the command line.
    """
    # --- Resolve arguments ---------------------------------------------------
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        run_id = run_id or cli.run_id
        subset = subset or cli.subset
        if apply_icemask is None:
            apply_icemask = cli.apply_icemask
    elif apply_icemask is None:
        apply_icemask = True  

    # --- Load raw YAML -------------------------------------------------------
    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # Determine active subset: CLI arg > YAML active_subset key > error
    if subset is None:
        subset = raw.get("active_subset")
    if subset is None:
        raise ValueError(
            "No subset specified.  Pass --subset on the command line "
            f"(one of: {', '.join(SUBSET_COMPUTE_FNS)}), "
            "or set 'active_subset' in the config YAML."
        )
    if subset not in SUBSET_COMPUTE_FNS:
        raise ValueError(
            f"Unknown subset '{subset}'.  "
            f"Valid options: {list(SUBSET_COMPUTE_FNS)}"
        )

    # --- Resolve subset entry ------------------------------------------------
    subsets_cfg  = raw.get("subsets", {})
    subset_entry = subsets_cfg.get(subset, {})

    if not subset_entry:
        raise ValueError(
            f"No entry found for subset '{subset}' under the 'subsets' key in "
            f"{config_file}.  Please add a 'subsets.{subset}' block."
        )

    # --- Build JobConfig in memory -------------------------------------------
    # The 'subsets' and 'active_subset' keys are top-level YAML keys that
    # config.load_config does not know about.  The JobConfig is built directly.
    output_dict = {**raw.get("output", {})}
    if "dataset_name" in subset_entry:
        output_dict["dataset_name"] = subset_entry["dataset_name"]

    cfg = config.JobConfig(
        run=config.RunConfig(**raw.get("run", {})),
        data=config.DataConfig(**raw.get("data", {})),
        sampling=config.SamplingConfig(**raw.get("sampling", {})),
        output=config.OutputConfig(**output_dict),
        features=config.FeaturesConfig(
            model_data_feature_channels=subset_entry.get(
                "model_data_feature_channels", []
            ),
            compute_features_channels=subset_entry.get(
                "compute_features_channels", []
            ),
        ),
        runtime=config.RuntimeConfig(**raw.get("runtime", {})),
    )


    run_global_pipeline(
        run_id=run_id,
        compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
        cfg=cfg,
        apply_icemask=apply_icemask,
    )
