"""
Fully-lazy dask LLC4320 pipeline: depth diagnostics via xgcm + dask.

This variant keeps the entire computation graph lazy and calls
``dask.compute()`` only once at the end of each subset's compute
function.  All horizontal gradient / interpolation work uses xgcm
natively on dask arrays — no per-k materialisation, no per-face loops,
no numpy stencils.

Key differences from the other pipeline variants:

  - ``generate_global_depth_horiz.py`` (per-face numpy):
    Processes one LLC face at a time with explicit halo extraction
    and numpy stencil operators.  No xgcm at all.

  - ``generate_global_depth_opt.py`` (per-k materialise + xgcm):
    Calls ``_materialise_for_xgcm()`` at every k-level, triggering
    xgcm ``map_overlap`` 51× per gradient field.

  - **THIS** (fully lazy dask + xgcm):
    Builds a lazy task graph over the full 3D arrays.  xgcm operates
    on dask arrays with native zarr chunks — the same pattern used by
    ``generate_global.py`` for 2D surface fields.

Chunking strategy
-----------------
S3 zarr layout:  ``{face: 1, k: 51, j: 720, i: 720}``

Data is read lazily from S3 with native zarr chunks — critically,
``face=1`` (13 blocks of size 1).  xgcm's ``_pad_face_connections``
iterates per-face and always produces 13 chunks of size 1 on the
face axis; ``adjust_chunks`` in ``dask.array.map_overlap`` is
computed from the *original* input chunksizes, so these must match
(i.e. 13 blocks, not 1 block of 13).

Each chunk is ~1.5 GB (1 face × 51 k × 720 j × 720 i × 8 bytes),
giving ~468 chunks per variable — small enough for the scheduler to
stay responsive, large enough to amortise overhead.  No chunk
splitting, no local cache.

Pipeline flow per timestep:
    load lazy from S3 → process_llc4320 merge →
    compute_fields_fn (lazy dask) → face→latlon → write

CLI usage
---------
    python dev/generate_global_depth_dask.py \\
        --config configs/global_depth.yaml \\
        --subset kinematic \\
        [--run_id my_run]
"""

# stdlib
import sys
import logging
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

# numerical / compute
import numpy as np
import zarr
import xarray as xr

# distributed / IO
import dask
import yaml
from dask.diagnostics import ProgressBar

# progress
import tqdm

# internal
from dbof.io.filesystems import create_s3_filesystems

import dbof.preprocessing.native_grid_masks as native_grid_masks
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
import dbof.preprocessing.calculated_fields_at_depth as depth_fields

import dbof.llc4320_ingestion.get_raw_data as get_raw_data
from dbof.llc4320_ingestion import grid as llc_grid

import dbof.dataset_creation.zarr_dataset_global as zarr_dataset
import dbof.dataset_creation.config as config

import dbof.utils.faces_to_latlon as faces_to_latlon

# Fully-lazy dask compute functions.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import calculated_fields_at_depth_dask as depth_fields_dask


# ---------------------------------------------------------------------------
# LLC4320 model constants
# ---------------------------------------------------------------------------
TS_PER_HOUR              = 144
LLC_FACES                = range(13)
LLC4320_START_DATE       = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS    = 25
DATE_FMT                 = '%Y-%m-%d %H:%M:%S'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_logging(cfg: config.JobConfig, config_dir: Path = None) -> None:
    """Configure file + stdout logging."""
    log_path = Path(cfg.run.log_dir).expanduser()
    if not log_path.is_absolute() and config_dir is not None:
        log_path = (config_dir / log_path).resolve()
    else:
        log_path = log_path.resolve()

    run_dir = log_path / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "generate_global_depth_dask.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info(f"Log file: {log_file}")


def _date_to_iteration(date_str: str) -> int:
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start ({LLC4320_START_DATE.date()})."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


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
) -> np.ndarray:
    """
    Process one time snapshot and return a ``(C, H, W)`` array.

    ``compute_fields_fn`` returns numpy-backed DataArrays (the dask
    compute functions materialise internally).
    """
    # Computed (mode-specific) derived fields.
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # For surface_only mode, extract surface from any remaining 3D model vars.
    surface_model_vars = {}
    for ch in model_feature_channels:
        if ch not in ds_merge:
            continue
        da = ds_merge[ch]
        if "k" in da.dims:
            surface_model_vars[ch] = da.isel(k=0)
        elif "k_l" in da.dims:
            surface_model_vars[ch] = da.isel(k_l=0)
        else:
            surface_model_vars[ch] = da

    # Materialise staggered-grid variables before grid.interp
    for var in ('V', 'U', 'oceTAUY', 'oceTAUX'):
        if var in surface_model_vars:
            surface_model_vars[var] = surface_model_vars[var].compute()

    # Move staggered-grid values to tracer points.
    if "V" in surface_model_vars:
        surface_model_vars["V"] = grid.interp(surface_model_vars["V"], 'Y', boundary='fill')
    if "U" in surface_model_vars:
        surface_model_vars["U"] = grid.interp(surface_model_vars["U"], 'X', boundary='fill')
    if "oceTAUY" in surface_model_vars:
        surface_model_vars["oceTAUY"] = grid.interp(surface_model_vars["oceTAUY"], 'Y', boundary='fill')
    if "oceTAUX" in surface_model_vars:
        surface_model_vars["oceTAUX"] = grid.interp(surface_model_vars["oceTAUX"], 'X', boundary='fill')

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

    metric_vector_pairs = []
    if 'V' in ds_to_convert.variables: ds_to_convert['V'].attrs.pop('mate', None)
    if 'U' in ds_to_convert.variables: ds_to_convert['U'].attrs['mate'] = 'V'
    if 'oceTAUY' in ds_to_convert.variables: ds_to_convert['oceTAUY'].attrs.pop('mate', None)
    if 'oceTAUX' in ds_to_convert.variables: ds_to_convert['oceTAUX'].attrs['mate'] = 'oceTAUY'

    # Build surface land mask from hFacC.
    hfac = ds_merge.hFacC if "k" not in ds_merge.hFacC.dims else ds_merge.hFacC.isel(k=0)
    land_mask_da = (hfac == 0)
    mask_vars = {'_land_mask': land_mask_da}

    ds_to_convert = ds_to_convert.assign(mask_vars)
    channels_to_convert_with_mask = channels_to_convert + list(mask_vars.keys())

    logging.info("Converting LLC faces -> rectangular lat/lon")
    ds_rect = faces_to_latlon.faces_dataset_to_latlon(
        ds_to_convert[channels_to_convert_with_mask],
        metric_vector_pairs=metric_vector_pairs,
    )

    logging.info("Materializing stitched arrays")
    with ProgressBar():
        land_mask_rect = ds_rect['_land_mask'].values.astype(bool)
        channel_arrays = [ds_rect[ch].values for ch in channels_to_convert]

    logging.info("Stacking into (C, H, W)")
    data = np.stack(channel_arrays, axis=0)
    data = np.where(land_mask_rect[np.newaxis], np.nan, data)

    logging.info("Assembly complete")
    return data


# ---------------------------------------------------------------------------
# Per-subset compute callbacks
# ---------------------------------------------------------------------------

def _compute_native_fields(ds_merge, grid, computed_feature_channels: list) -> dict:
    """No-op callback for subsets that only output raw model variables."""
    return {}


# Fully-lazy dask compute functions for all depth-resolved subsets.
SUBSET_COMPUTE_FNS = {
    # Surface-only subsets (no depth computation).
    "native_fields":     _compute_native_fields,
    "native_surface":    _compute_native_fields,
    "eta":               _compute_native_fields,
    "icearea":           _compute_native_fields,
    "windstress":        _compute_native_fields,
    # Depth-resolved diagnostic subsets — all fully lazy dask.
    "stratification":    depth_fields_dask.compute_stratification,
    "surface_wind":      depth_fields_dask.compute_surface_wind,
    "vertical_shear":    depth_fields_dask.compute_vertical_shear,
    "mixing_parameters": depth_fields_dask.compute_mixing_parameters,
    "ertel_pv":          depth_fields_dask.compute_ertel_pv,
    "buoyancy_fluxes":   depth_fields_dask.compute_buoyancy_fluxes,
    "energetics":        depth_fields_dask.compute_energetics,
    "frontal_structure": depth_fields_dask.compute_frontal_structure,
    "kinematic":         depth_fields_dask.compute_kinematic,
    "frontogenesis":     depth_fields_dask.compute_frontogenesis,
}


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
    cpu_start  = time.process_time()

    if cfg is None:
        if config_file is None:
            cli = config.parse_args()
            config_file = cli.config
            run_id = run_id or cli.run_id
        cfg = config.load_config(config_file)

    if run_id is not None:
        cfg = config.JobConfig(
            run=config.RunConfig(run_id=run_id, log_dir=cfg.run.log_dir),
            data=cfg.data,
            sampling=cfg.sampling,
            output=cfg.output,
            features=cfg.features,
            runtime=cfg.runtime,
        )

    generate_logging(cfg, config_dir=config_dir)
    logging.info("Arguments parsed successfully. Logging set up. Running script.")
    logging.info("Pipeline variant: fully-lazy dask (xgcm on dask arrays, single .compute())")

    model_feature_channels    = [c.strip() for c in cfg.features.model_data_feature_channels if c.strip()]
    computed_feature_channels = [c.strip() for c in cfg.features.compute_features_channels   if c.strip()]

    zarr_concurrency = cfg.runtime.zarr_async_concurrency
    zarr.config.set({'async.concurrency': zarr_concurrency})

    # Dask distributed scheduler.
    from dask.distributed import Client
    client_kwargs = {}
    if cfg.runtime.dask_n_workers is not None:
        client_kwargs["n_workers"] = cfg.runtime.dask_n_workers
    if cfg.runtime.dask_threads_per_worker is not None:
        client_kwargs["threads_per_worker"] = cfg.runtime.dask_threads_per_worker
    if cfg.runtime.dask_memory_limit is not None:
        client_kwargs["memory_limit"] = cfg.runtime.dask_memory_limit
    dask_client = Client(**client_kwargs)
    logging.info(f"Dask distributed client: {dask_client}")
    dask.config.set({"distributed.scheduler.allowed-failures": 10})

    def _set_zarr_concurrency(concurrency):
        import zarr as _zarr
        _zarr.config.set({'async.concurrency': concurrency})
    dask_client.run(_set_zarr_concurrency, zarr_concurrency)
    logging.info(f"Zarr async concurrency set to {zarr_concurrency} on all workers")

    if surface_only:
        logging.info("Pipeline mode: surface-only variables (dask lazy)")
    else:
        logging.info("Pipeline mode: depth diagnostics (fully-lazy dask -> reduce -> save)")

    if s3_source is None:
        raise ValueError(
            "s3_source must be provided with keys: 's3_endpoint', 'bucket', 'folder' "
            "pointing to the S3 location of the transferred LLC4320 timestep stores."
        )

    # Validate required date list.
    if cfg.data.date_iterations is None or len(cfg.data.date_iterations) == 0:
        raise ValueError(
            "data.date_iterations must be set in the config for the S3-based "
            "pipeline.  Each entry should be a date string matching a transferred "
            "timestep store (e.g., '2012-11-09 12:00:00')."
        )

    logging.info(f"Processing: {len(cfg.data.date_iterations)} date(s) from S3 timestep stores")

    fs, fs_synch = create_s3_filesystems(cfg.output.s3_endpoint)

    ds_grid, land_mask = set_up_grid_data_and_masks_s3(cfg, s3_source, use_halo=False)

    # Eagerly load the grid into memory -- it's small (~1.5 GB).
    logging.info("Eagerly loading grid into memory...")
    ds_grid = ds_grid.compute()
    logging.info("Grid loaded into memory.")

    # Build the xgcm Grid from horizontal grid variables only.
    # The vertical coordinates (Z, Zl, Zu, Zp1, drF) are not needed
    # by xgcm for horizontal interp/diff — they remain in ds_grid
    # and are available via ds_merge for the depth compute functions.
    # This matches how generate_global.py creates its Grid (2D only).
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

    # Determine which raw variables are needed from S3.
    vars_needed = list(set(model_feature_channels))

    if computed_feature_channels:
        # Theta/Salt needed for buoyancy/density-based diagnostics.
        tracer_keywords = (
            'N2_', 'mixed_layer', 'ml_heat', 'Ri_', 'Fr_', 'Burger',
            'ertel_pv', 'uB', 'vB', 'wB', 'Ro_', 'KE_',
            'gradb2_', 'gradtheta2_', 'gradsalt2_', 'gradrho2_',
            'turner_angle_', 'frontogenesis_',
        )
        needs_tracers = any(
            any(ch.startswith(kw) for kw in tracer_keywords)
            for ch in computed_feature_channels
        )
        if needs_tracers:
            for v in ('Theta', 'Salt'):
                if v not in vars_needed:
                    vars_needed.append(v)

        # Velocity-dependent diagnostics need U, V.
        velocity_keywords = (
            'vertical_shear', 'Ri_', 'Fr_', 'Ro_', 'Burger',
            'ertel_pv', 'uB', 'vB',
            'relative_vorticity_', 'strain_', 'divergence_',
            'okubo_weiss_', 'frontogenesis_', 'ug_', 'vg_',
        )
        needs_uv = any(
            any(ch.startswith(kw) for kw in velocity_keywords)
            for ch in computed_feature_channels
        )
        if needs_uv:
            for v in ('U', 'V'):
                if v not in vars_needed:
                    vars_needed.append(v)

        # PV and wB need W.
        w_keywords = ('ertel_pv', 'wB')
        needs_w = any(
            any(ch.startswith(kw) for kw in w_keywords)
            for ch in computed_feature_channels
        )
        if needs_w:
            if 'W' not in vars_needed:
                vars_needed.append('W')

        # Wind-stress diagnostics need oceTAUX, oceTAUY.
        wind_keywords = ('wind_stress_curl', 'ekman_pumping', 'u_ekman', 'v_ekman')
        needs_wind = any(ch in wind_keywords for ch in computed_feature_channels)
        if needs_wind:
            for v in ('oceTAUX', 'oceTAUY'):
                if v not in vars_needed:
                    vars_needed.append(v)

        # Eta-dependent diagnostics (gradeta2, frontogenesis ug/vg).
        eta_keywords = ('gradeta2_', 'frontogenesis_', 'ug_', 'vg_')
        needs_eta = any(
            any(ch.startswith(kw) for kw in eta_keywords)
            for ch in computed_feature_channels
        )
        if needs_eta:
            if 'Eta' not in vars_needed:
                vars_needed.append('Eta')

    logging.info(f"Requesting variables from S3 timestep stores: {vars_needed}")

    for date_str in tqdm.tqdm(cfg.data.date_iterations):
        it = _date_to_iteration(date_str)

        # Load all 13 faces lazily from S3 (no local cache).
        #
        # xgcm's _pad_face_connections iterates per-face and always
        # produces 13 chunks of size 1 on the face axis.  The
        # adjust_chunks parameter that dask.array.map_overlap passes
        # to blockwise is computed from the *original* (unpadded) input
        # chunksizes.  For these to match, face must be chunked as
        # (1,1,...,1) — 13 blocks of size 1.  This is exactly what
        # s3_timestep_chunks provides (face=1) and what the working
        # generate_global.py pipeline uses.
        logging.info(f"Loading S3 timestep data for {date_str} (iteration {it})")
        from datetime import datetime as _dt
        _date_tag = _dt.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime("%Y%m%dT%H")
        _store = f"{_date_tag}.zarr"
        _s3_url = get_raw_data._build_s3_store_url(
            s3_source['bucket'], s3_source['folder'], _store
        )
        ds = xr.open_zarr(
            _s3_url,
            consolidated=False,
            chunks=get_raw_data.s3_timestep_chunks,
            storage_options=get_raw_data._s3_storage_options(s3_source['s3_endpoint']),
        )
        if vars_needed:
            available = [v for v in vars_needed if v in ds]
            ds = ds[available]

        # Transpose face to the leading axis for all variables.
        # xgcm's _pad_face_connections does isel(face=i) → concat(dim='face'),
        # which always puts face at axis 0 in the padded dask array.
        # adjust_chunks is computed from the *original* axis order, so face
        # must already be axis 0 for the two to match on 3D+ arrays.
        # (For 2D arrays where face is already axis 0 this is a no-op.)
        if "face" in ds.dims:
            ds = ds.transpose("face", ...)

        logging.info(
            f"S3 timestep data loaded: {_store}, vars={list(ds.data_vars)}"
        )

        if surface_only:
            ds = _select_surface(ds)

        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        # Ensure face-first ordering survives the merge with ds_grid.
        if "face" in ds_merge.dims:
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

    # --- CPU-hour summary ---
    wall_elapsed = time.monotonic() - wall_start
    cpu_elapsed  = time.process_time() - cpu_start
    wall_hours   = wall_elapsed / 3600.0
    cpu_hours    = cpu_elapsed / 3600.0
    logging.info("=" * 60)
    logging.info("Run complete.")
    logging.info(f"  Wall-clock time : {wall_hours:.3f} h  ({wall_elapsed:.1f} s)")
    logging.info(f"  CPU time        : {cpu_hours:.3f} h  ({cpu_elapsed:.1f} s)")
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
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        run_id = run_id or cli.run_id
        subset = subset or cli.subset

    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    if subset is None:
        subset = raw.get("active_subset")
    if subset is None:
        raise ValueError(
            "No subset specified.  Pass --subset on the command line "
            f"(one of: {', '.join(SUBSET_COMPUTE_FNS)}), "
            "or set 'active_subset' in the config YAML."
        )
    if subset not in SUBSET_COMPUTE_FNS:
        raise ValueError(f"Unknown subset '{subset}'. Valid: {list(SUBSET_COMPUTE_FNS)}")

    subsets_cfg  = raw.get("subsets", {})
    subset_entry = subsets_cfg.get(subset, {})
    if not subset_entry:
        raise ValueError(
            f"No entry for subset '{subset}' under 'subsets' in {config_file}."
        )

    surface_only = bool(subset_entry.get("surface_only", False))

    output_dict = {**raw.get("output", {})}
    if "dataset_name" in subset_entry:
        output_dict["dataset_name"] = subset_entry["dataset_name"]

    cfg = config.JobConfig(
        run=config.RunConfig(**raw.get("run", {})),
        data=config.DataConfig(**raw.get("data", {})),
        sampling=config.SamplingConfig(**raw.get("sampling", {})),
        output=config.OutputConfig(**output_dict),
        features=config.FeaturesConfig(
            model_data_feature_channels=subset_entry.get("model_data_feature_channels", []),
            compute_features_channels=subset_entry.get("compute_features_channels", []),
        ),
        runtime=config.RuntimeConfig(**raw.get("runtime", {})),
    )

    # S3 source: where the transferred timestep stores live.
    s3_source_cfg = raw.get("s3_source", {})
    if not s3_source_cfg:
        raise ValueError(
            "Missing 's3_source' section in the config YAML.  "
            "This must specify s3_endpoint, bucket, and folder for the "
            "transferred LLC4320 timestep stores."
        )

    run_global_pipeline(
        run_id=run_id,
        compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
        cfg=cfg,
        s3_source=s3_source_cfg,
        surface_only=surface_only,
        config_dir=Path(config_file).resolve().parent,
    )


if __name__ == "__main__":
    main()
