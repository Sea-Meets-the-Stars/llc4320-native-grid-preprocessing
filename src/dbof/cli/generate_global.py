"""
Unified global LLC4320 dataset generation pipeline.

Single entry point that dispatches on the ``pipeline`` key in the YAML
config. 

Pipeline variants
-----------------
SURF
    Core ocean variables (Theta, Salt, Eta, U, V, W) from OSN kerchunk;
    forcing variables (oceTAUX, oceTAUY, SIarea) from S3 timestep stores
    written by ``transfer_llc4320.py`` into the ``LLC4320`` folder.

OSN
    All variables from OSN kerchunk endpoints (surface + wind).  No S3
    timestep stores are used.

DEPTH
    All variables from S3 timestep stores in ``LLC4320_v1`` (full depth).
    Grid is read from the ``LLC4320`` folder (original, non-corrupt
    transfer location).

Chunking & face-ordering contract (important for xgcm / DEPTH)
---------------------------------------------------------------
S3 zarr layout:  ``{face: 1, k: 51, j: 720, i: 720}``

Two requirements must be satisfied for xgcm's ``map_overlap`` with LLC
face connections to work correctly:

1. **face must be chunked as (1,1,...,1)** — 13 blocks of size 1.
2. **face must be axis 0 for 3D+ arrays.**

The DEPTH loading path handles both via ``transpose("face", ...)`` after
opening the zarr store with native ``face=1`` chunks.

CLI usage
---------
    generate-global --config configs/global/run/run.yaml
    generate-global --config configs/global/run/run.yaml --subset kinematic
    generate-global --config configs/global/run/run.yaml --pipeline OSN

Each pipeline supports a specific set of subsets.  See the valid
pipeline × subset combinations in ``configs/global/run/run.yaml``.

Output layout
-------------
    s3://{bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}

Each date gets its own subdirectory (``date_prefix``).
"""

# stdlib
import argparse
import logging
import time
from pathlib import Path

# distributed / IO
import yaml
import tqdm

# internal — data loading
import dbof.llc4320_ingestion.get_raw_data as get_raw_data
import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data

# internal — IO
from dbof.io.filesystems import create_s3_filesystems

# internal — global pipeline modules
import dbof.global_dataset_creation.config as config
import dbof.global_dataset_creation.check_existence as check_existence
import dbof.global_dataset_creation.zarr_dataset_global as zarr_dataset
from dbof.global_dataset_creation.data_sources import (
    OSN_ENDPOINT,
    OSN_SURFACE_VARS,
    OSN_WIND_VARS,
    get_data_source,
)
from dbof.global_dataset_creation.grid_setup import set_up_grid
from dbof.global_dataset_creation.iterations import (
    LLC_FACES,
    date_to_run_id,
    mit_date_to_iteration,
    osn_date_to_iteration,
)
from dbof.global_dataset_creation.dask import create_dask_client
from dbof.global_dataset_creation.logging import setup_logging, save_run_metadata
from dbof.global_dataset_creation.subset_definitions import (
    expand_channels_with_suffixes,
    get_compute_fn,
    get_subset_definition,
    valid_subsets,
)
from dbof.global_dataset_creation.variable_selection import required_model_variables
import dbof.global_dataset_creation.process_surface as process_surface
import dbof.global_dataset_creation.process_depth as process_depth


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECTANGULAR_SHAPE = (3 * 4320, 4 * 4320)   # (12960, 17280)


# ---------------------------------------------------------------------------
# Snapshot loading — dispatches on pipeline
# ---------------------------------------------------------------------------

def load_snapshot(
    pipeline: str,
    date_str: str,
    ds_grid,
    vars_needed: list[str],
    surface_only: bool,
    data_source: dict | None,
):
    """
    Load one time snapshot from the appropriate store.

    Parameters
    ----------
    pipeline : str
        ``"SURF"``, ``"OSN"``, or ``"DEPTH"``.
    date_str : str
        ISO date string (e.g. ``'2012-11-09 12:00:00'``).
    ds_grid : xr.Dataset
        Pre-loaded grid dataset.
    vars_needed : list[str]
        Raw model variable names required for this subset.
    surface_only : bool
        If ``True`` (DEPTH pipeline), slice 3D data to surface after load.
    data_source : dict or None
        S3 data-source dict.  Required for SURF and DEPTH; ``None`` for OSN.

    Returns
    -------
    ds : xr.Dataset
        Raw dataset (face topology preserved for face→latlon stitching).
    ds_merge : xr.Dataset
        Merged dataset (raw + grid variables).
    it : int
        LLC4320 iteration number for this snapshot.
    """

    if pipeline == "DEPTH":
        it = mit_date_to_iteration(date_str)
        logging.info(f"Loading LLC_DEPTH timestep data for {date_str} (MIT iteration {it})")
        ds = get_raw_data.get_llc_timestep_data(
            data_source["s3_endpoint"],
            data_source["bucket"],
            data_source["folder"],
            date_str,
            vars_requested=vars_needed or None,
            chunks=get_raw_data.llc_depth_timestep_chunks,
            storage_options=get_raw_data._llc_depth_storage_options(
                data_source["s3_endpoint"]
            ),
        )
        # face must be axis 0 for xgcm (see module docstring).
        if "face" in ds.dims:
            ds = ds.transpose("face", ...)
        if surface_only:
            ds = process_depth._select_surface(ds)

        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)
        if "face" in ds_merge.dims:
            ds_merge = ds_merge.transpose("face", ...)

    elif pipeline == "OSN":
        it = osn_date_to_iteration(date_str)
        logging.info(f"Loading OSN kerchunk data for {date_str} (OSN iteration {it})")
        ds = get_raw_data.get_remote_llc_data(OSN_ENDPOINT, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)

        # Merge wind/sea-ice kerchunk variables if needed.
        if set(vars_needed) & OSN_WIND_VARS:
            logging.info("Merging llc_wind kerchunk variables")
            ds_wind = get_raw_data.get_remote_llc_wind_data(
                OSN_ENDPOINT, it, LLC_FACES
            )
            ds_merge = ds_merge.merge(ds_wind)

    elif pipeline == "SURF":
        it = osn_date_to_iteration(date_str)
        logging.info(
            f"Loading SURF data for {date_str} (OSN iteration {it})"
        )
        # Core ocean variables from OSN kerchunk.
        ds = get_raw_data.get_remote_llc_data(OSN_ENDPOINT, it, LLC_FACES)
        ds_merge = preproc_llc_core_data.process_llc4320(ds, ds_grid)

        # Supplement with LLC_SURF forcing variables not in kerchunk.
        llc_surf_vars = [v for v in vars_needed if v not in OSN_SURFACE_VARS]
        if llc_surf_vars and data_source:
            logging.info(f"Loading LLC_SURF forcing variables: {llc_surf_vars}")
            ds_llc = get_raw_data.get_llc_timestep_data(
                data_source["s3_endpoint"],
                data_source["bucket"],
                data_source["folder"],
                date_str,
                face_range=LLC_FACES,
                vars_requested=llc_surf_vars,
            )
            # LLC_SURF stores carry full depth; select surface before merging.
            for dim_name in ("k", "k_l"):
                if dim_name in ds_llc.dims:
                    ds_llc = ds_llc.isel({dim_name: 0})
            for v in llc_surf_vars:
                if v in ds_llc:
                    ds_merge[v] = ds_llc[v]
            logging.info(
                f"LLC_SURF variables merged: {[v for v in llc_surf_vars if v in ds_llc]}"
            )
            # Cross-check OSN vs. LLC_SURF timestamps for data integrity.
            get_raw_data.verify_osn_llc_surf_timestamp(
                ds, data_source, date_str, LLC_FACES
            )
        elif llc_surf_vars:
            logging.warning(
                f"LLC_SURF-only variables {llc_surf_vars} requested but no "
                "data_source configured; these will be missing from the output."
            )

    else:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'.  Expected SURF, OSN, or DEPTH."
        )

    logging.info(f"Data loaded for date: {date_str}")
    return ds, ds_merge, it


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Unified global LLC4320 dataset generation pipeline.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--run_id", default=None, help="Override run_id from config."
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Override active_subsets with a single subset.  "
            "Valid values depend on the pipeline."
        ),
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        choices=["SURF", "OSN", "DEPTH"],
        help="Override the pipeline key in the config YAML.",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        default=False,
        help=("Regenerate subset/date zarr stores that already exist on S3.  "
              "Default is to skip existing stores (per date)."),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    config_file: str = None,
    run_id: str = None,
    subset: str = None,
    pipeline: str = None,
    clobber: bool = False,
) -> None:
    """
    Entry point for the unified global pipeline.

    Can be called from the CLI (no arguments) or programmatically.

    Parameters
    ----------
    config_file : str, optional
        Path to the YAML config.  If ``None``, read from ``--config``.
    run_id : str, optional
        Override for ``run.run_id``.
    subset : str, optional
        Override ``active_subsets`` with a single subset name.
    pipeline : str, optional
        Override the ``pipeline`` key in the YAML.
    clobber : bool, optional
        If ``False`` (default), skip any subset/date whose zarr store already
        exists on S3 *and* is complete (all expected channels present, at
        least one timestep written).  If ``True``, regenerate it (snapshots
        are overwritten in place; the store never grows duplicates).

    """
    wall_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Parse args, load YAML
    # ------------------------------------------------------------------
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        run_id = run_id or cli.run_id
        subset = subset or cli.subset
        pipeline = pipeline or cli.pipeline
        clobber = clobber or cli.clobber

    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # 2. Resolve pipeline and active_subsets
    # ------------------------------------------------------------------
    pipeline_str = pipeline or raw.get("pipeline")
    if pipeline_str is None:
        raise ValueError(
            "No pipeline specified.  Set 'pipeline' in the YAML config "
            "or pass --pipeline on the command line."
        )
    pipeline_str = pipeline_str.upper()

    if subset is not None:
        active_subsets = [subset]
    else:
        active_subsets = raw.get("active_subsets")
        if not active_subsets:
            raise ValueError(
                "No subsets specified.  Set 'active_subsets' (list) in the "
                "YAML config, or pass --subset on the command line."
            )

    # Validate subsets early.
    valid = valid_subsets(pipeline_str)
    for s in active_subsets:
        if s not in valid:
            raise ValueError(
                f"Subset '{s}' is not valid for pipeline '{pipeline_str}'.  "
                f"Valid subsets: {valid}"
            )

    # ------------------------------------------------------------------
    # 3. Date iterations (from YAML)
    # ------------------------------------------------------------------
    date_iterations = raw.get("data", {}).get("date_iterations")
    if not date_iterations:
        raise ValueError(
            "data.date_iterations must be set in the config YAML.  "
            "Each entry should be a date string in ISO format "
            "(e.g. '2012-11-09 12:00:00')."
        )
    # ------------------------------------------------------------------
    # 4. Build GlobalJobConfig
    # ------------------------------------------------------------------
    run_cfg = config.RunConfig(**(raw.get("run") or {}))
    if run_id is not None:
        run_cfg = config.RunConfig(run_id=run_id, log_dir=run_cfg.log_dir)

    output_cfg = config.GlobalOutputConfig(**(raw.get("output") or {}))
    # Resolve folder from pipeline if not explicitly set in YAML.
    if output_cfg.folder is None:
        output_cfg = config.GlobalOutputConfig(
            s3_endpoint=output_cfg.s3_endpoint,
            bucket=output_cfg.bucket,
            folder=config.default_output_folder(pipeline_str),
            dataset_name=output_cfg.dataset_name,
        )

    cfg = config.GlobalJobConfig(
        run=run_cfg,
        data=config.GlobalDataConfig(**(raw.get("data") or {})),
        output=output_cfg,
        runtime=config.RuntimeConfig(**(raw.get("runtime") or {})),
        pipeline=pipeline_str,
        active_subsets=active_subsets,
        depth_suffixes=raw.get("depth_suffixes"),
    )

    # ------------------------------------------------------------------
    # 5. Logging and run metadata
    # ------------------------------------------------------------------
    log_file = setup_logging(cfg)  # appends if the log already exists
    logging.info("Unified global pipeline starting.")
    logging.info(f"Pipeline: {cfg.pipeline}")
    logging.info(f"Active subsets: {cfg.active_subsets}")
    logging.info(f"Depth suffixes (YAML override): {cfg.depth_suffixes}")
    logging.info(f"Dates: {date_iterations}")

    # ------------------------------------------------------------------
    # 6. Resolve per-subset specs (cheap: no S3, no dask)
    # ------------------------------------------------------------------
    subset_specs = []
    for subset_name in cfg.active_subsets:
        defn = get_subset_definition(cfg.pipeline, subset_name)
        model_channels = list(defn["model_data_feature_channels"])

        # Expand depth suffixes: YAML override > subset definition default.
        # Only apply the YAML override when the subset definition itself
        # declares depth_suffixes — surface subsets must never get suffixes.
        defn_suffixes = defn.get("depth_suffixes")
        depth_suffixes = (cfg.depth_suffixes or defn_suffixes) if defn_suffixes is not None else None
        compute_channels = expand_channels_with_suffixes(
            defn["compute_features_channels"],
            depth_suffixes=depth_suffixes,
            extra_channels=defn.get("extra_channels"),
        )
        subset_specs.append({
            "subset_name": subset_name,
            "compute_fn": get_compute_fn(cfg.pipeline, subset_name),
            "surface_only": defn.get("surface_only", False),
            "dataset_name": defn["dataset_name"],
            "model_channels": model_channels,
            "compute_channels": compute_channels,
            "channel_names": model_channels + compute_channels,
            "vars_needed": required_model_variables(model_channels, compute_channels),
        })

    # ------------------------------------------------------------------
    # 7. Pre-flight: decide each subset/date BEFORE expensive setup.
    # ------------------------------------------------------------------
    fs, fs_sync = create_s3_filesystems(cfg.output.s3_endpoint)

    logging.info("Pre-flight plan (zarr existence per subset/date):")
    to_generate = []  # list of (spec, date_str, date_prefix)
    for spec in subset_specs:
        for date_str in date_iterations:
            date_prefix = date_to_run_id(date_str)
            tag = f"{spec['subset_name']:<22} {date_str}"
            store_path = zarr_dataset.make_run_prefix(
                cfg.output.bucket, cfg.output.folder, cfg.run.run_id,
                spec["dataset_name"], date_prefix=date_prefix,
            )

            if clobber:
                logging.info(f"  {tag}  ->  GENERATE (clobber: regenerate in place)")
                to_generate.append((spec, date_str, date_prefix))
                continue

            zarr_state = check_existence.plan_zarr(
                fs_sync, store_path, spec["channel_names"])
            if zarr_state == check_existence.ZARR_FULL:
                logging.info(f"  {tag}  ->  SKIP (zarr store complete)")
            elif zarr_state == check_existence.ZARR_MISSING:
                logging.info(f"  {tag}  ->  GENERATE (zarr store missing)")
                to_generate.append((spec, date_str, date_prefix))
            else:  # ZARR_INCOMPLETE
                logging.error(f"  {tag}  ->  ERROR (zarr store incomplete)")
                raise ValueError(
                    f"Existing zarr store is incomplete: {store_path}\n"
                    "  (missing expected channels and/or has 0 timesteps -- "
                    "e.g. a crashed run or a different depth-suffix set).\n"
                    "  Delete the store and rerun, or pass --clobber to "
                    "regenerate it in place."
                )

    # ------------------------------------------------------------------
    # 8. Nothing to generate?  Skip all expensive setup and return.
    # ------------------------------------------------------------------
    if not to_generate:
        logging.info("All subset/date zarr stores are already complete -- "
                     "nothing to generate (skipping dask client + grid load).")
        try:
            from s3fs import S3FileSystem
            S3FileSystem.clear_instance_cache()
        except Exception:
            pass
        logging.info("Run complete (no work).  Wall-clock time: %.1f s",
                     time.monotonic() - wall_start)
        return

    # ------------------------------------------------------------------
    # 9. Expensive one-time setup — only when there is work to do.
    # ------------------------------------------------------------------
    data_source = get_data_source(cfg.pipeline)
    dask_client = create_dask_client(cfg.runtime)
    # Save run metadata (local + S3).  Read-merge-write: per-subset entries
    # accumulate across invocations instead of overwriting the file.
    # Use the sync filesystem — the async one conflicts with the Dask
    # distributed client's event loop.
    save_run_metadata(cfg, log_file, fs=fs_sync, clobber=clobber)
    ds_grid, land_mask, grid = set_up_grid(cfg.pipeline, data_source)
    logging.info(f"LLC rectangular output shape: {RECTANGULAR_SHAPE}")

    # ------------------------------------------------------------------
    # 10. Generate each planned subset/date.
    # ------------------------------------------------------------------
    for spec, date_str, date_prefix in tqdm.tqdm(to_generate, desc="generate"):
        logging.info(f"\n{'='*60}")
        logging.info(f"Generating subset: {spec['subset_name']}  date: {date_str}")
        logging.info(f"{'='*60}")
        logging.info(f"  dataset_name: {spec['dataset_name']}")
        logging.info(f"  surface_only: {spec['surface_only']}")
        logging.info(f"  model_channels: {spec['model_channels']}")
        logging.info(f"  compute_channels: {spec['compute_channels']}")
        logging.info(f"  vars_needed from storage: {spec['vars_needed']}")

        # Create zarr output store for this subset + date.
        zarr_ds = zarr_dataset.GlobalZarrDataset(
            cfg.output.bucket,
            cfg.output.folder,
            cfg.run.run_id,
            spec["dataset_name"],
            fs=fs,
            channel_names=spec["channel_names"],
            rectangular_shape=RECTANGULAR_SHAPE,
            date_prefix=date_prefix,
        )

        # Load snapshot.
        ds, ds_merge, it = load_snapshot(
            cfg.pipeline, date_str, ds_grid, spec["vars_needed"],
            spec["surface_only"], data_source,
        )

        # Process snapshot — dispatch to surface or depth processor.
        if cfg.pipeline == "DEPTH":
            data = process_depth.process_snapshot(
                ds, ds_merge, grid,
                spec["model_channels"], spec["compute_channels"],
                spec["compute_fn"], spec["surface_only"],
            )
        else:
            data = process_surface.process_snapshot(
                ds, ds_merge, grid,
                spec["model_channels"], spec["compute_channels"],
                spec["compute_fn"],
            )

        # Write to zarr.
        logging.info("Writing snapshot to zarr dataset")
        zarr_ds.write_snapshot(data, it)

        # Release references for GC.
        ds = None
        ds_merge = None
        data = None

    # ------------------------------------------------------------------
    # 11. Cleanup
    # ------------------------------------------------------------------
    wall_elapsed = time.monotonic() - wall_start
    wall_hours = wall_elapsed / 3600.0
    n_workers = len(dask_client.scheduler_info().get("workers", {}))

    logging.info("=" * 60)
    logging.info("Run complete.")
    logging.info(f"  Pipeline          : {cfg.pipeline}")
    logging.info(f"  Subsets           : {cfg.active_subsets}")
    logging.info(f"  Wall-clock time   : {wall_hours:.2f} h  ({wall_elapsed:.1f} s)")
    logging.info(f"  Dask workers      : {n_workers}")
    # 🔵 CHANGED: report stores actually generated, not just the date count.
    logging.info(f"  Stores generated  : {len(to_generate)}")
    logging.info("=" * 60)

    dask_client.close()
    logging.info("Dask client closed.")
    try:
        from s3fs import S3FileSystem
        S3FileSystem.clear_instance_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
