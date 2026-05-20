"""
Fully-lazy dask LLC4320 pipeline: depth diagnostics via xgcm + dask.

The entire computation graph stays lazy until a single ``dask.compute()``
at the end of each subset's compute function.  Horizontal gradients and
interpolation use xgcm natively on dask arrays — no per-k materialisation,
no per-face loops, no numpy stencils.

Pipeline flow per timestep
--------------------------
    load lazy from S3 → process_llc4320 merge →
    compute_fields_fn (lazy dask) → face→latlon → write

Chunking & face-ordering contract (important for xgcm)
------------------------------------------------------
S3 zarr layout:  ``{face: 1, k: 51, j: 720, i: 720}``

Two requirements must be satisfied for xgcm's ``map_overlap`` with LLC
face connections to work correctly:

1. **face must be chunked as (1,1,...,1) — 13 blocks of size 1.**
   xgcm's ``_pad_face_connections`` (in ``xgcm/padding.py``) iterates
   per-face via ``isel(face=i)`` then ``concat(dim='face')``, always
   producing 13 chunks of size 1.  The ``adjust_chunks`` parameter
   that ``dask.array.map_overlap`` passes to ``blockwise`` is computed
   from the *original* (unpadded) input chunksizes.  If the original
   has 1 block of 13 (``face=-1``) but xgcm produces 13 blocks of 1,
   blockwise raises ``ValueError: Dimension 0 has 13 blocks,
   adjust_chunks specified with 1 blocks``.  Solution: use native zarr
   chunks (``face=1``), which is what ``s3_timestep_3D_chunks`` provides.

2. **face must be axis 0 for 3D+ arrays.**
   After ``_pad_face_connections`` does ``isel → concat``, face always
   ends up at position 0 in the padded dask array.  ``adjust_chunks``
   is keyed by axis position (0, 1, 2, ...) and is computed from the
   *original* axis order.  If the original has ``(k, face, j, i)`` then
   axis 0 is k with 1 chunk — but xgcm's padded output has face (13
   chunks) at axis 0.  Mismatch → same ``ValueError``.  Solution:
   ``ds.transpose("face", ...)`` before xgcm sees the data.  This is
   a lazy metadata-only operation (no data movement).

CLI usage
---------
    python -m dbof.cli.generate_global_depth \\
        --config configs/global_depth.yaml \\
        --subset kinematic \\
        [--run_id my_run]
"""

# stdlib
import logging
import argparse
import time
from pathlib import Path

# numerical / compute
import xarray as xr

# distributed / IO
import yaml

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.depth_subsets as depth_subsets
from dbof.utils.faces_to_latlon import (
    interp_staggered_to_tracer, set_vector_pair_attrs, stitch_and_mask,
)

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
from dbof.llc4320_ingestion import grid as llc_grid

import dbof.dataset_creation.zarr_dataset_global as zarr_dataset
import dbof.dataset_creation.config as config

from dbof.utils.logging import generate_logging
from dbof.utils.iterations import mit_date_to_iteration
from dbof.utils.variable_selection import required_model_variables
from dbof.utils.runtime import resolve_config, extract_feature_channels, create_dask_client
from dbof.utils.subset_config import resolve_subset, build_job_config, run_per_date

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_date_to_iteration = mit_date_to_iteration


def set_up_grid_data_and_masks_s3(cfg: config.JobConfig, s3_source: dict, use_halo: bool = False):
    """Load the LLC4320 grid and compute a static land mask from S3 grid store."""
    grid_folder = s3_source.get('grid_folder', s3_source['folder'])
    logging.info(f"Fetching grid file from S3 grid store (folder={grid_folder})")
    co = get_raw_data.get_s3_gridfile(
        s3_source['s3_endpoint'],
        s3_source['bucket'],
        grid_folder,
    )
    ds_grid = preproc_llc_core_data.process_llc4320_3d_grid(co)

    logging.info(f"Calculating land mask (use_halo={use_halo})")
    if use_halo:
        land_mask = native_grid_masks.generate_static_land_mask_for_sampling(
            ds_grid, cfg.output.target_km_res
        )
    else:
        land_mask = ds_grid.hFacC
    return ds_grid, land_mask


def _select_surface(ds: xr.Dataset) -> xr.Dataset:
    """Return a dataset with all k-dependent variables sliced to the surface."""
    out = {}
    for name, da in ds.data_vars.items():
        if "k" in da.dims:
            out[name] = da.isel(k=0)
        elif "k_l" in da.dims:
            out[name] = da.isel(k_l=0)
        else:
            out[name] = da
    return xr.Dataset(out, coords=ds.coords, attrs=ds.attrs)


# ---------------------------------------------------------------------------
# Core snapshot processor
# ---------------------------------------------------------------------------

def process_snapshot(
    cfg: config.JobConfig,
    ds,
    ds_merge,
    grid,
    land_mask,
    model_feature_channels: list,
    computed_feature_channels: list,
    it: int,
    compute_fields_fn,
    surface_only: bool,
) -> "np.ndarray":
    """
    Process one time snapshot and return a ``(C, H, W)`` array.

    ``compute_fields_fn`` returns numpy-backed DataArrays (the dask
    compute functions materialise internally).
    """
    # Computed (mode-specific) derived fields.
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # Extract surface slice from any 3D model vars.
    available = [ch for ch in model_feature_channels if ch in ds_merge]
    ds_model_subset = _select_surface(ds_merge[available])
    surface_model_vars = {ch: ds_model_subset[ch] for ch in available}

    # Materialise staggered-grid variables before grid.interp (dask → numpy).
    for var in ('V', 'U', 'oceTAUY', 'oceTAUX'):
        if var in surface_model_vars:
            surface_model_vars[var] = surface_model_vars[var].compute()

    # Move staggered-grid values to tracer points.
    interp_staggered_to_tracer(surface_model_vars, grid)

    # Assemble all channels into a single Dataset for one conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels

    # Build a surface-level base dataset for face->latlon conversion.
    ds_surface = ds
    for dim_name in ("k", "k_l"):
        if dim_name in ds_surface.dims:
            ds_surface = ds_surface.isel({dim_name: 0})

    update_vars = (
        {ch: surface_model_vars[ch] for ch in model_feature_channels if ch in surface_model_vars}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels if ch in calculated_fields}
    )
    ds_to_convert = ds_surface.assign(update_vars)[channels_to_convert]
    set_vector_pair_attrs(ds_to_convert)

    # Build surface land mask from hFacC.
    hfac = ds_merge.hFacC if "k" not in ds_merge.hFacC.dims else ds_merge.hFacC.isel(k=0)
    mask_dict = {'_land_mask': (hfac == 0)}

    data = stitch_and_mask(ds_to_convert, channels_to_convert, mask_dict,
                           progress_bar=True)

    logging.info("Assembly complete")
    return data


# ---------------------------------------------------------------------------
# Per-subset compute callbacks (all live in depth_subsets.py)
# ---------------------------------------------------------------------------

SUBSET_COMPUTE_FNS = depth_subsets.SUBSET_COMPUTE_FNS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fully-lazy dask LLC4320 pipeline: depth diagnostics via "
            "xgcm on dask arrays (single .compute() at end)."
        ),
    )
    parser.add_argument("--config",  required=True, help="Path to the YAML config file.")
    parser.add_argument("--run_id",  default=None,  help="Override run_id from config.")
    parser.add_argument(
        "--subset", default=None, choices=list(SUBSET_COMPUTE_FNS),
        help=f"Subset to compute. One of: {', '.join(SUBSET_COMPUTE_FNS)}.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_global_pipeline(
    config_file: str = None,
    run_id: str = None,
    compute_fields_fn=None,
    cfg: config.JobConfig = None,
    s3_source: dict = None,
    surface_only: bool = False,
    config_dir: Path = None,
) -> None:
    """
    Main orchestration loop for the fully-lazy dask pipeline.

    Uses the dask distributed scheduler for all computation: the lazy
    task graph built by the compute functions is materialised by the
    scheduler's workers.
    """
    wall_start = time.monotonic()

    cfg = resolve_config(cfg, config_file, run_id, config_module=config)

    generate_logging(cfg, config_dir=config_dir,
                     log_filename="generate_global_depth.log")
    logging.info("Arguments parsed successfully. Logging set up. Running script.")
    logging.info("Pipeline variant: fully-lazy dask (xgcm on dask arrays, single .compute())")

    model_feature_channels, computed_feature_channels = extract_feature_channels(cfg)
    dask_client = create_dask_client(cfg.runtime)

    if surface_only:
        logging.info("Pipeline mode: surface-only variables (dask lazy)")
    else:
        logging.info("Pipeline mode: depth diagnostics (fully-lazy dask -> reduce -> save)")

    logging.info(f"Processing: {len(cfg.data.date_iterations)} date(s) from S3 timestep stores")

    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)

    ds_grid, land_mask = set_up_grid_data_and_masks_s3(cfg, s3_source, use_halo=False)

    # Eagerly load the grid into memory -- it's small (~1.5 GB).
    logging.info("Eagerly loading grid into memory...")
    ds_grid = ds_grid.compute()
    logging.info("Grid loaded into memory.")

    # Drop vertical coords — xgcm only needs the horizontal grid.
    _vert_vars = {'Z', 'Zl', 'Zu', 'Zp1', 'drF'}
    _drop = [v for v in _vert_vars if v in ds_grid]
    grid_for_xgcm = ds_grid.drop_vars(_drop) if _drop else ds_grid
    grid = llc_grid.set_xgcm_grid(grid_for_xgcm, use_connections=True)

    rectangular_shape = (3 * 4320, 4 * 4320)
    logging.info(f"LLC rectangular output shape: {rectangular_shape}")

    channel_names = model_feature_channels + computed_feature_channels

    zarr_ds = zarr_dataset.GlobalZarrDataset(
        cfg.output.bucket, cfg.output.folder, cfg.run.run_id,
        cfg.output.dataset_name, fs=fs,
        channel_names=channel_names, rectangular_shape=rectangular_shape,
    )
    logging.info("Zarr dataset created.")

    vars_needed = required_model_variables(model_feature_channels,
                                           computed_feature_channels)

    logging.info(f"Requesting variables from S3 timestep stores: {vars_needed}")

    for date_str in tqdm.tqdm(cfg.data.date_iterations):
        it = _date_to_iteration(date_str)

        # Load lazily with native face=1 chunks (see module docstring).
        logging.info(f"Loading S3 timestep data for {date_str} (iteration {it})")
        ds = get_raw_data.get_s3_timestep_data(
            s3_source['s3_endpoint'],
            s3_source['bucket'],
            s3_source['folder'],
            date_str,
            vars_requested=vars_needed or None,
            chunks=get_raw_data.s3_timestep_3D_chunks,
            storage_options=get_raw_data._s3_storage_options_3D(s3_source['s3_endpoint']),
        )

        # face must be axis 0 for xgcm — see module docstring.
        if "face" in ds.dims:
            ds = ds.transpose("face", ...)

        if surface_only:
            ds = _select_surface(ds)

        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        if "face" in ds_merge.dims:       # re-apply after merge with ds_grid
            ds_merge = ds_merge.transpose("face", ...)
        logging.info(f"Data loaded for date: {date_str}")

        data = process_snapshot(
            cfg, ds, ds_merge, grid, land_mask,
            model_feature_channels, computed_feature_channels,
            it, compute_fields_fn,
            surface_only=surface_only,
        )

        ds = None
        ds_merge = None

        logging.info("Writing snapshot to zarr dataset")
        zarr_ds.write_snapshot(data, it)

    wall_elapsed = time.monotonic() - wall_start
    wall_hours   = wall_elapsed / 3600.0
    n_workers    = len(dask_client.scheduler_info().get("workers", {}))
    logging.info("=" * 60)
    logging.info("Run complete.")
    logging.info(f"  Wall-clock time : {wall_hours:.2f} h  ({wall_elapsed:.1f} s)")
    logging.info(f"  Dask workers    : {n_workers}")
    logging.info(f"  Snapshots       : {len(cfg.data.date_iterations)}")
    logging.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    config_file: str = None,
    run_id: str = None,
    subset: str = None,
) -> None:
    """Entry point.  Reads CLI args when called with no arguments."""
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        run_id = run_id or cli.run_id
        subset = subset or cli.subset

    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    subset, subset_entry = resolve_subset(raw, subset, SUBSET_COMPUTE_FNS)
    surface_only = bool(subset_entry.get("surface_only", False))

    # S3 source: where the transferred timestep stores live.
    s3_source_cfg = raw.get("s3_source", {})
    if not s3_source_cfg:
        raise ValueError(
            "Missing 's3_source' section in the config YAML.  "
            "This must specify s3_endpoint, bucket, and folder for the "
            "transferred LLC4320 timestep stores."
        )

    date_iterations = raw.get("data", {}).get("date_iterations")
    if not date_iterations:
        raise ValueError(
            "data.date_iterations must be set in the config for the S3-based "
            "pipeline.  Each entry should be a date string matching a "
            "transferred timestep store (e.g., '2012-11-09 12:00:00')."
        )

    pipeline_kwargs = dict(
        s3_source=s3_source_cfg,
        surface_only=surface_only,
        config_dir=Path(config_file).resolve().parent,
    )

    # Per-date looping: when no --run_id, each date gets its own output dir.
    if run_id is None and len(date_iterations) > 1:
        run_per_date(
            raw, subset_entry, date_iterations,
            pipeline_fn=run_global_pipeline,
            compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
            **pipeline_kwargs,
        )
        return

    # Single run (explicit run_id, or only one date).
    cfg = build_job_config(raw, subset_entry)
    run_global_pipeline(
        run_id=run_id,
        compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
        cfg=cfg,
        **pipeline_kwargs,
    )


if __name__ == "__main__":
    main()
