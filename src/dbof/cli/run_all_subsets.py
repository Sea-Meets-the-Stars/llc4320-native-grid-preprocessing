"""
run_all_subsets.py
------------------
Batch driver that loops over all active subsets in a ``run.yaml`` config:

1. **Phase 1 — Generate:** calls ``generate_global.main()`` for each
   subset, producing Zarr stores on S3.
2. **Phase 2 — Export:** calls ``zarr_to_netcdf.main()`` once per
   channel, writing individual NetCDF files.

This script reads the *same* ``run.yaml`` consumed by
``generate_global.py`` — it does not have its own config format.
Subset definitions and channel lists come from code
(``subset_definitions.py``), not from the YAML.

NetCDF output layout
~~~~~~~~~~~~~~~~~~~~
::

    {netcdf_base}/{run_id}/{date_prefix}/LLC4320_{date}_{channel}_{run_id}.nc

CLI usage
---------
The ``--pipeline`` flag selects which pipeline variant to run (SURF, OSN,
or DEPTH).  It can also be set via the ``pipeline`` key in the YAML
config; the CLI flag takes precedence.
::

    # DEPTH pipeline — generate + export all active subsets:
    run-all-subsets --pipeline DEPTH \
        --config configs/global/run.yaml --netcdf-base /mnt/tank/Oceanography/data/OGCM/LLC/Fronts --subsets stratification --run-id vtest --ice-mask

    # SURF pipeline:
    run-all-subsets --pipeline SURF \\
        --config configs/global/run.yaml --netcdf-base /path/to/output

    # OSN pipeline:
    run-all-subsets --pipeline OSN \\
        --config configs/global/run.yaml --netcdf-base /path/to/output

    # Only specific subsets (overrides active_subsets in YAML):
    run-all-subsets --pipeline DEPTH \\
        --config configs/global/run.yaml --netcdf-base /path/to/output \\
        --subsets stratification native_fields

    # Export only (assumes Zarr stores already exist):
    run-all-subsets --pipeline DEPTH \\
        --config configs/global/run.yaml --netcdf-base /path/to/output \\
        --export-only

    # Generate only (skip NetCDF export):
    run-all-subsets --pipeline DEPTH \\
        --config configs/global/run.yaml --netcdf-base /path/to/output \\
        --generate-only

    # Override run_id:
    run-all-subsets --pipeline DEPTH \\
        --config configs/global/run.yaml --netcdf-base /path/to/output \\
        --run-id my_run_01

    # Dry run:
    run-all-subsets --pipeline DEPTH \\
        --config configs/global/run.yaml --netcdf-base /path/to/output \\
        --dry-run

    # With ice masking (mask SIarea > 0 → NaN during NetCDF export):
    run-all-subsets --pipeline DEPTH \\
        --config configs/global/run.yaml --netcdf-base /path/to/output \\
        --ice-mask
"""

import argparse
import logging
import os
import sys
import time

import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ice-mask dependency: ensure icearea.zarr exists
# ---------------------------------------------------------------------------

ICE_MASK_DATASET = "icearea.zarr"


def _icearea_store_exists(
    s3_endpoint: str, bucket: str, folder: str, run_id: str, date_prefix: str,
) -> bool:
    """Check whether icearea.zarr exists at the expected S3 path.

    Parameters
    ----------
    s3_endpoint : str
        S3 endpoint URL.
    bucket : str
        S3 bucket name.
    folder : str
        S3 folder path.
    run_id : str
        Run identifier.
    date_prefix : str
        Date prefix string (``YYYYMMDD_HHMMSS``).

    Returns
    -------
    bool
        ``True`` if the store exists.
    """
    from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix
    from dbof.io.filesystems import create_s3_filesystems

    path = make_run_prefix(bucket, folder, run_id, ICE_MASK_DATASET,
                           date_prefix=date_prefix)
    s3_key = path.removeprefix("s3://")
    _, fs_sync = create_s3_filesystems(s3_endpoint)
    return fs_sync.exists(s3_key)


def _ensure_icearea(
    config_file: str,
    pipeline: str,
    run_id: str,
    s3_endpoint: str,
    bucket: str,
    folder: str,
    date_prefixes: list[str],
    dry_run: bool = False,
) -> None:
    """Generate the icearea subset if its Zarr store is missing for any date.

    Called automatically when ``--ice-mask`` is set so that the export
    phase can always find ``icearea.zarr``.

    Parameters
    ----------
    config_file : str
        Path to the YAML config (passed through to ``generate_global``).
    pipeline : str
        Pipeline name (``"SURF"``, ``"OSN"``, or ``"DEPTH"``).
    run_id : str
        Run identifier.
    s3_endpoint : str
        S3 endpoint URL.
    bucket : str
        S3 bucket name.
    folder : str
        S3 folder path.
    date_prefixes : list[str]
        Date prefix strings to check.
    dry_run : bool
        If ``True``, log what would be done without generating.
    """
    missing = [
        dp for dp in date_prefixes
        if not _icearea_store_exists(s3_endpoint, bucket, folder, run_id, dp)
    ]

    if not missing:
        log.info("icearea.zarr already exists for all %d date(s) — skipping.",
                 len(date_prefixes))
        return

    log.info("icearea.zarr missing for %d/%d date(s) — generating now.",
             len(missing), len(date_prefixes))

    _run_generate(
        config_file=config_file,
        subset="icearea",
        run_id=run_id,
        pipeline=pipeline,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Phase 1: Generate zarr stores
# ---------------------------------------------------------------------------

def _run_generate(
    config_file: str,
    subset: str,
    run_id: str,
    pipeline: str,
    dry_run: bool = False,
) -> None:
    """Call ``generate_global.main()`` for one subset.

    Parameters
    ----------
    config_file : str
        Path to the YAML config.
    subset : str
        Subset name to generate.
    run_id : str
        Run identifier (passed as override to ``generate_global``).
    pipeline : str
        Pipeline name override.
    dry_run : bool
        If ``True``, log the call without executing.
    """
    log.info("=" * 60)
    log.info("GENERATE  config=%s  subset=%s  pipeline=%s",
             config_file, subset, pipeline)
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY RUN] Would call generate_global.main("
                 "config_file=%r, subset=%r, run_id=%r, pipeline=%r)",
                 config_file, subset, run_id, pipeline)
        return

    from dbof.cli.generate_global import main as generate_main
    generate_main(
        config_file=config_file,
        run_id=run_id,
        subset=subset,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# Phase 2: Export zarr → per-variable NetCDF
# ---------------------------------------------------------------------------

from dbof.global_dataset_creation.iterations import (
    date_to_run_id as _date_to_prefix,
    prefix_to_filename_date as _prefix_to_filename_date,
)


def _run_export_channel(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    run_id: str,
    dataset_name: str,
    date_prefix: str,
    channel: str,
    output_dir: str,
    dry_run: bool = False,
    ice_mask: bool = False,
) -> None:
    """Call ``zarr_to_netcdf.main()`` for a single channel.

    Produces one ``.nc`` file with naming convention::

        LLC4320_{date}_{channel}_{run_id}.nc

    Parameters
    ----------
    s3_endpoint : str
        S3 endpoint URL.
    bucket : str
        S3 bucket name.
    folder : str
        S3 folder path.
    run_id : str
        Run identifier.
    dataset_name : str
        Zarr store name (e.g. ``"stratification.zarr"``).
    date_prefix : str
        Date prefix string (``YYYYMMDD_HHMMSS``).
    channel : str
        Channel name to export.
    output_dir : str
        Local directory for the NetCDF file.
    dry_run : bool
        If ``True``, log the call without executing.
    ice_mask : bool
        If ``True``, mask ice-covered points with NaN.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename_date = _prefix_to_filename_date(date_prefix)
    output_filename = f"LLC4320_{filename_date}_{channel}_{run_id}.nc"

    log.info("  EXPORT  %s -> %s/%s", channel, output_dir, output_filename)

    if dry_run:
        return

    from dbof.cli.zarr_to_netcdf import main as netcdf_main

    netcdf_main(
        output_dir=output_dir,
        output_filename=output_filename,
        mode="snapshots",
        s3_endpoint=s3_endpoint,
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name=dataset_name,
        date_prefix=date_prefix,
        channels=[channel],
        ice_mask=ice_mask,
    )


def _run_export_subset(
    s3_endpoint: str,
    bucket: str,
    folder: str,
    run_id: str,
    subset_name: str,
    dataset_name: str,
    channels: list[str],
    date_prefixes: list[str],
    netcdf_base: str,
    dry_run: bool = False,
    ice_mask: bool = False,
) -> None:
    """Export all channels in one subset to individual NetCDF files.

    Parameters
    ----------
    s3_endpoint : str
        S3 endpoint URL.
    bucket : str
        S3 bucket name.
    folder : str
        S3 folder path.
    run_id : str
        Run identifier.
    subset_name : str
        Name of the subset being exported.
    dataset_name : str
        Zarr store name (e.g. ``"stratification.zarr"``).
    channels : list[str]
        Expanded channel names to export.
    date_prefixes : list[str]
        Date prefix strings (``YYYYMMDD_HHMMSS``).
    netcdf_base : str
        Root directory for NetCDF output.
    dry_run : bool
        If ``True``, log what would be done without executing.
    ice_mask : bool
        If ``True``, mask ice-covered points with NaN.
    """
    if not channels:
        log.warning("  No channels for subset '%s' — skipping export.", subset_name)
        return

    # Don't self-mask the icearea subset.
    apply_ice_mask = ice_mask and subset_name != "icearea"
    mask_label = " [ice-masked]" if apply_ice_mask else ""

    log.info("-" * 60)
    log.info("EXPORT subset=%s  dataset=%s  channels=%d  dates=%d%s",
             subset_name, dataset_name, len(channels), len(date_prefixes),
             mask_label)
    log.info("-" * 60)

    for dp in date_prefixes:
        output_dir = os.path.join(netcdf_base, run_id, dp)
        log.info("  date_prefix=%s  -> %s", dp, output_dir)

        for channel in channels:
            try:
                _run_export_channel(
                    s3_endpoint=s3_endpoint,
                    bucket=bucket,
                    folder=folder,
                    run_id=run_id,
                    dataset_name=dataset_name,
                    date_prefix=dp,
                    channel=channel,
                    output_dir=output_dir,
                    dry_run=dry_run,
                    ice_mask=apply_ice_mask,
                )
            except Exception:
                log.exception("  FAILED to export channel '%s' from subset '%s' "
                              "(date_prefix=%s)", channel, subset_name, dp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "Batch driver: generate all Zarr subsets then export each "
            "channel to a separate NetCDF file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pipeline", default=None, choices=["SURF", "OSN", "DEPTH"],
        help=("Pipeline variant: SURF, OSN, or DEPTH.  "
              "Overrides the 'pipeline' key in the YAML config."),
    )
    p.add_argument(
        "--config", required=True, metavar="YAML",
        help="Path to the run YAML config (same format as generate-global).",
    )
    p.add_argument(
        "--subsets", nargs="+", metavar="NAME",
        help=("Only process these subset(s).  Default: all subsets listed "
              "in active_subsets in the config."),
    )
    p.add_argument(
        "--run-id", default=None,
        help="Override run_id from config.",
    )
    p.add_argument(
        "--netcdf-base", required=True,
        help="Base directory for NetCDF output (required).",
    )
    p.add_argument(
        "--generate-only", action="store_true",
        help="Only run the generate step (skip NetCDF export).",
    )
    p.add_argument(
        "--export-only", action="store_true",
        help="Only run the NetCDF export step (skip generate).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running anything.",
    )
    p.add_argument(
        "--ice-mask", action="store_true", default=False,
        help=("Mask ice-covered points (SIarea > 0) with NaN during "
              "NetCDF export.  The icearea subset itself is never masked."),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Batch-generate and export LLC4320 global subsets.

    Reads the YAML config, resolves pipeline / subsets / output folder,
    then runs Phase 1 (generate Zarr) and Phase 2 (export NetCDF) for
    each active subset.
    """
    args = _parse_args()

    # ---- Load config YAML (same format as generate_global) ----
    with open(args.config) as fh:
        raw = yaml.safe_load(fh) or {}

    # ---- Resolve pipeline ----
    pipeline = (args.pipeline or raw.get("pipeline", "")).upper()
    if not pipeline:
        log.error("No pipeline specified.  Set 'pipeline' in the YAML "
                  "or pass --pipeline on the CLI.")
        sys.exit(1)

    # ---- Resolve active subsets ----
    from dbof.global_dataset_creation.subset_definitions import (
        get_subset_definition,
        expand_channels_with_suffixes,
        valid_subsets,
    )

    if args.subsets:
        active_subsets = args.subsets
    else:
        active_subsets = raw.get("active_subsets")
        if not active_subsets:
            log.error("No subsets specified.  Set 'active_subsets' in the "
                      "YAML or pass --subsets on the CLI.")
            sys.exit(1)

    # Validate subsets.
    valid = valid_subsets(pipeline)
    for s in active_subsets:
        if s not in valid:
            log.error("Subset '%s' is not valid for pipeline '%s'.  "
                      "Valid subsets: %s", s, pipeline, valid)
            sys.exit(1)

    # ---- Resolve run_id ----
    run_id = args.run_id or raw.get("run", {}).get("run_id", "unknown_run")

    # ---- Resolve dates ----
    date_iterations = raw.get("data", {}).get("date_iterations", [])
    if not date_iterations:
        log.error("data.date_iterations must be set in the config YAML.")
        sys.exit(1)
    date_prefixes = [_date_to_prefix(d) for d in date_iterations]

    # ---- Resolve output S3 location ----
    from dbof.global_dataset_creation.config import default_output_folder

    output_raw = raw.get("output") or {}
    s3_endpoint = output_raw.get("s3_endpoint", "https://s3-west.nrp-nautilus.io")
    bucket = output_raw.get("bucket", "dbof/")
    folder = output_raw.get("folder") or default_output_folder(pipeline)

    # ---- Resolve depth suffixes (DEPTH pipeline only) ----
    depth_suffixes_override = raw.get("depth_suffixes")

    # ---- Build work plan ----
    work = []  # list of (subset_name, dataset_name, channels)
    for subset_name in active_subsets:
        defn = get_subset_definition(pipeline, subset_name)

        # Apply depth_suffixes override from YAML if present.
        if depth_suffixes_override and "depth_suffixes" in defn:
            defn["depth_suffixes"] = depth_suffixes_override

        model_channels = defn.get("model_data_feature_channels", []) or []
        computed_channels = expand_channels_with_suffixes(
            channels=defn.get("compute_features_channels", []) or [],
            depth_suffixes=defn.get("depth_suffixes"),
            extra_channels=defn.get("extra_channels"),
        )
        channels = model_channels + computed_channels
        dataset_name = defn["dataset_name"]
        work.append((subset_name, dataset_name, channels))

    # ---- Log the plan ----
    log.info("Pipeline: %s  |  run_id: %s  |  dates: %d  |  subsets: %d",
             pipeline, run_id, len(date_prefixes), len(work))
    for subset_name, dataset_name, channels in work:
        log.info("  %s  (%s, %d channels)", subset_name, dataset_name, len(channels))

    wall_start = time.monotonic()

    # ---- Pre-flight: ensure icearea.zarr exists when ice masking ----
    if args.ice_mask and not args.export_only:
        try:
            _ensure_icearea(
                config_file=args.config,
                pipeline=pipeline,
                run_id=run_id,
                s3_endpoint=s3_endpoint,
                bucket=bucket,
                folder=folder,
                date_prefixes=date_prefixes,
                dry_run=args.dry_run,
            )
        except Exception:
            log.exception(
                "FAILED to ensure icearea.zarr.  "
                "Ice masking may fail during export."
            )

    # ---- Phase 1: Generate ----
    if not args.export_only:
        log.info("")
        log.info("=" * 60)
        log.info("PHASE 1: GENERATE ZARR STORES")
        log.info("=" * 60)

        for subset_name, dataset_name, channels in work:
            try:
                _run_generate(
                    config_file=args.config,
                    subset=subset_name,
                    run_id=run_id,
                    pipeline=pipeline,
                    dry_run=args.dry_run,
                )
            except Exception:
                log.exception("FAILED to generate subset '%s'", subset_name)

    # ---- Clear stale S3 filesystem cache between phases ----
    if not args.export_only and not args.generate_only:
        try:
            from s3fs import S3FileSystem
            S3FileSystem.clear_instance_cache()
            log.info("Cleared s3fs instance cache between phases.")
        except Exception:
            pass  # Best-effort.

    # ---- Phase 2: Export ----
    if not args.generate_only:
        if args.ice_mask and args.export_only:
            log.warning(
                "Ice masking requested with --export-only.  If icearea.zarr "
                "has not been generated yet, channel exports will fail.  "
                "Run without --export-only to auto-generate it."
            )

        log.info("")
        log.info("=" * 60)
        log.info("PHASE 2: EXPORT ZARR -> PER-VARIABLE NETCDF")
        log.info("=" * 60)

        for subset_name, dataset_name, channels in work:
            try:
                _run_export_subset(
                    s3_endpoint=s3_endpoint,
                    bucket=bucket,
                    folder=folder,
                    run_id=run_id,
                    subset_name=subset_name,
                    dataset_name=dataset_name,
                    channels=channels,
                    date_prefixes=date_prefixes,
                    netcdf_base=args.netcdf_base,
                    dry_run=args.dry_run,
                    ice_mask=args.ice_mask,
                )
            except Exception:
                log.exception("FAILED to export subset '%s'", subset_name)

    wall_elapsed = time.monotonic() - wall_start
    log.info("")
    log.info("=" * 60)
    log.info("All done.  Total wall-clock time: %.1f s (%.2f h)",
             wall_elapsed, wall_elapsed / 3600)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
