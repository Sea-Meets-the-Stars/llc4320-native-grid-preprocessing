"""
Global LLC4320 dataset generation.

Single-entry-point script for generating global LLC4320 property datasets.
The subset of properties to compute is selected at runtime via ``--subset``
(or the ``active_subset`` key in the YAML config).

Available subsets
-----------------
native_fields
    Raw model state variables only (Theta, Salt, Eta, U, V, W, oceTAUX,
    oceTAUY, SIarea).
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
        --config configs/global.yaml \\
        --subset native_fields \\
        [--run_id my_run] \\
        [--icemask]

When ``date_iterations`` is set in the YAML, each date is processed as a
separate pipeline run, stored in a date subdirectory under the run_id::

    s3://dbof/surface_fields/{run_id}/20111209_120000/native_fields.zarr
    s3://dbof/surface_fields/{run_id}/20121109_120000/native_fields.zarr

When ``--run_id`` is provided it overrides the config value; otherwise the
run_id from the YAML is used.

Config design
-------------
The YAML adds two top-level keys consumed here before the config object is
constructed:

    active_subset: kinematic       # default; overridden by --subset

    subsets:
      native_fields:
        dataset_name: "native_fields.zarr"
        model_data_feature_channels: [Theta, Salt, Eta, U, V, W, ...]
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
import logging
import argparse

# distributed / IO
import yaml
# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.surface_subsets as surface_subsets
from dbof.utils.faces_to_latlon import (
    interp_staggered_to_tracer, set_vector_pair_attrs, stitch_and_mask,
)

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
from dbof.llc4320_ingestion import grid as llc_grid

import dbof.dataset_creation.zarr_dataset_global as zarr_dataset
import dbof.dataset_creation.config as config

from dbof.utils.logging import generate_logging
from dbof.utils.iterations import LLC_FACES, calculate_iterations_for_llc
from dbof.utils.variable_selection import required_model_variables
from dbof.utils.runtime import resolve_config, extract_feature_channels, create_dask_client
from dbof.utils.subset_config import resolve_subset, build_job_config, run_per_date

# URL of the raw LLC4320 data store
ENDPOINT_URL = 'https://mghp.osn.xsede.org'

# Variables available from the kerchunk endpoint; anything else needs S3.
_KERCHUNK_VARS = {'Theta', 'Salt', 'Eta', 'U', 'V', 'W'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Process one snapshot: compute fields → stagger interp → stitch → write."""

    # --- Computed (mode-specific) fields ---
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # Move non-tracer values to tracer points so all channels share the same
    # (face, j, i) grid before the face→latlon stitch.
    interp_staggered_to_tracer(ds_merge, grid)

    # Assemble all channels into a single Dataset for a single conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels
    update_vars = (
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]
    set_vector_pair_attrs(ds_to_convert)

    # Build mask dict: land (always) + optional ice mask.
    mask_dict = {'_land_mask': (ds_merge.hFacC == 0)}
    if apply_icemask:
        logging.info("Calculating and applying ice mask (Theta <= 0) and land mask (hFacC == 0)")
        mask_dict['_ice_mask'] = (ds_merge.Theta <= 0.0)
    else:
        logging.info("Calculating and applying land mask (hFacC == 0); ice mask disabled")

    data = stitch_and_mask(ds_to_convert, channels_to_convert, mask_dict)

    # Write to zarr store.
    logging.info("Writing snapshot to zarr dataset")
    zarr_ds.write_snapshot(data, it)

    # Release references so Dask can reclaim worker memory.
    ds_merge = None
    del ds_merge
    grid = None
    del grid


# ---------------------------------------------------------------------------
# Subset dispatch (callbacks live in surface_subsets.py)
# ---------------------------------------------------------------------------

SUBSET_COMPUTE_FNS = surface_subsets.SUBSET_COMPUTE_FNS


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
        "--icemask",
        dest="apply_icemask",
        action="store_true",
        default=False,
        help=(
            "Enable the sea-ice mask (Theta <= 0).  By default the ice mask "
            "is NOT applied.  Pass --icemask to NaN out pixels where surface "
            "temperature is at or below freezing."
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
    s3_source: dict = None,
    date_prefix: str | None = None,
) -> None:
    """Orchestration loop: load grid, iterate snapshots, write zarr.

    Kerchunk variables come from OSN; anything else (oceTAUX, SIarea, …)
    is loaded from *s3_source* timestep stores when provided.

    Parameters
    ----------
    date_prefix : str or None, optional
        Date subdirectory inserted between *run_id* and *dataset_name*
        in the S3 output path (e.g. ``'20121109_120000'``).
    """
    cfg = resolve_config(cfg, config_file, run_id, config_module=config)

    generate_logging(cfg, log_filename="generate_global.log")
    logging.info("Arguments parsed successfully. Logging set up. Running script.")

    model_feature_channels, computed_feature_channels = extract_feature_channels(cfg)
    dask_client = create_dask_client(cfg.runtime)

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

    # Variables not in the kerchunk endpoint must come from S3 timestep stores.
    all_needed = required_model_variables(model_feature_channels,
                                          computed_feature_channels)
    s3_vars = [v for v in all_needed if v not in _KERCHUNK_VARS]

    iter_to_date = {}
    if s3_source and cfg.data.date_iterations is not None:
        if s3_vars:
            logging.info(f"Will load {s3_vars} from S3 timestep stores")
        for date_str, it in zip(cfg.data.date_iterations, iter_range):
            iter_to_date[int(it)] = date_str
    elif s3_vars and s3_source is None:
        logging.warning(
            f"S3-only variables {s3_vars} requested but no s3_source configured; "
            "these will be missing from the output."
        )
    elif s3_vars and cfg.data.date_iterations is None:
        raise ValueError(
            f"S3-only variables {s3_vars} require 'date_iterations' in the "
            "config when 's3_source' is provided."
        )

    # Construct the zarr output store
    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket,
        cfg.output.folder,
        cfg.run.run_id,
        cfg.output.dataset_name,
        fs=fs,
        channel_names=model_feature_channels + computed_feature_channels,
        rectangular_shape=rectangular_shape,
        date_prefix=date_prefix,
    )
    logging.info("Zarr dataset created.")

    for it in tqdm.tqdm(iter_range):
        # Fetch raw LLC4320 data for this iteration from S3/OSN
        ds       = get_raw_data.get_remote_llc_data(ENDPOINT_URL, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        logging.info(f"Data loaded for iteration: {it}")

        # Load additional surface variables from S3 timestep stores.
        if s3_vars and int(it) in iter_to_date:
            ds_s3 = get_raw_data.get_s3_timestep_data(
                s3_source['s3_endpoint'],
                s3_source['bucket'],
                s3_source['folder'],
                iter_to_date[int(it)],
                face_range=LLC_FACES,
                vars_requested=s3_vars,
            )
            # S3 stores carry full depth; select surface before merging.
            for dim_name in ("k", "k_l"):
                if dim_name in ds_s3.dims:
                    ds_s3 = ds_s3.isel({dim_name: 0})
            for v in s3_vars:
                if v in ds_s3:
                    ds_merge[v] = ds_s3[v]
            logging.info(f"S3 variables merged: {[v for v in s3_vars if v in ds_s3]}")

        # Cross-check OSN vs. S3 timestamps for data integrity.
        if s3_source and int(it) in iter_to_date:
            get_raw_data.verify_osn_s3_timestamp(
                ds, s3_source, iter_to_date[int(it)], LLC_FACES,
            )

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
    """Entry point.  Reads CLI args when called with no arguments.

    When ``--run_id`` is omitted and ``date_iterations`` is set, each date
    gets its own output directory via ``run_per_date``.
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
        apply_icemask = False

    # --- Load raw YAML and resolve subset ------------------------------------
    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    subset, subset_entry = resolve_subset(raw, subset, SUBSET_COMPUTE_FNS)

    # S3 source: optional, for variables not in the kerchunk endpoint.
    s3_source_cfg = raw.get("s3_source") or None

    # --- Per-date looping or single run --------------------------------------
    date_iterations = raw.get("data", {}).get("date_iterations")
    pipeline_kwargs = dict(
        apply_icemask=apply_icemask,
        s3_source=s3_source_cfg,
    )

    if date_iterations is not None and len(date_iterations) > 0:
        run_per_date(
            raw, subset_entry, date_iterations,
            pipeline_fn=run_global_pipeline,
            compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
            run_id=run_id,
            **pipeline_kwargs,
        )
        return

    # Single run (no date_iterations).
    cfg = build_job_config(raw, subset_entry)
    run_global_pipeline(
        run_id=run_id,
        compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
        cfg=cfg,
        **pipeline_kwargs,
    )
