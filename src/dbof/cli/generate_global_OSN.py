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
from dbof.utils.iterations import (
    LLC_FACES,
    osn_date_to_iteration, calculate_iterations_for_llc as _calc_iters_shared,
)
from dbof.utils.runtime import resolve_config, extract_feature_channels, create_dask_client
from dbof.utils.subset_config import resolve_subset, build_global_job_config, run_per_date

# URL of the raw LLC4320 data store
ENDPOINT_URL          = 'https://mghp.osn.xsede.org'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# OSN pipeline uses OSN iterations (with OSN offset) in the date path.
_date_to_iteration = osn_date_to_iteration


def calculate_iterations_for_llc(cfg):
    """OSN version: dates use OSN iterations."""
    return _calc_iters_shared(cfg, use_osn_offset=True)


def set_up_grid_data_and_masks(cfg: config.GlobalJobConfig, use_halo: bool = False):
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
    cfg: config.GlobalJobConfig,
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
    # OSN endpoint has no wind stress variables, so only interpolate U/V.
    interp_staggered_to_tracer(ds_merge, grid,
                               stagger_map={'U': 'X', 'V': 'Y'})

    # Assemble all channels into a single Dataset for a single conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels
    update_vars = (
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]
    set_vector_pair_attrs(ds_to_convert,
                          vector_pairs=[('U', 'V')])

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
    cfg: config.GlobalJobConfig = None,
    apply_icemask: bool = True,
    date_prefix: str | None = None,
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
    date_prefix : str or None, optional
        Date subdirectory inserted between *run_id* and *dataset_name*
        in the S3 output path (e.g. ``'20121109_120000'``).
    """
    cfg = resolve_config(cfg, config_file, run_id, config_module=config)

    generate_logging(cfg, log_filename="generate_global_OSN.log")
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

    # --- Load raw YAML and resolve subset ------------------------------------
    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    subset, subset_entry = resolve_subset(raw, subset, SUBSET_COMPUTE_FNS)

    # --- Per-date looping or single run --------------------------------------
    date_iterations = raw.get("data", {}).get("date_iterations")
    pipeline_kwargs = dict(
        apply_icemask=apply_icemask,
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
    cfg = build_global_job_config(raw, subset_entry)
    run_global_pipeline(
        run_id=run_id,
        compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
        cfg=cfg,
        **pipeline_kwargs,
    )


if __name__ == "__main__":
    main()
