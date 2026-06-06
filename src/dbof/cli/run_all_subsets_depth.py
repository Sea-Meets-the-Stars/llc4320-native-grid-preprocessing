"""
run_all_subsets.py
------------------
Batch driver that:

1. **Generates** all subsets listed in every config YAML found under
   ``configs/`` by calling ``generate_global.main()`` serially for each
   subset.

2. **Exports** each channel to its own NetCDF file by calling
   ``zarr_to_netcdf.main()`` once per channel per subset.

Output layout
~~~~~~~~~~~~~
::

    {netcdf_base}/{run_id}/{date_prefix}/LLC4320_{date}_{channel}_{run_id}.nc

where *netcdf_base* defaults to
``/mnt/tank/Oceanography/data/OGCM/LLC/Fronts`` and *date* is formatted
as ``2012-11-09T12_00_00``.

CLI usage
---------
::

    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml [OPTIONS]

    # Generate + export all subsets:
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml

    # Only run specific subsets:
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml \
        --subsets stratification native_fields

    # Skip the generate step (export only — assumes zarr stores already exist):
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml --export-only

    # Skip the export step (generate only):
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml --generate-only

    # Override run_id:
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml --run-id my_run_01

    # Dry run — print what would be done without running anything:
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml --dry-run

    # Export with ice mask — mask points where SIarea > 0 with NaN during
    # the NetCDF export step (via zarr_to_netcdf).  Reads SIarea from
    # icearea.zarr (same bucket/folder/run_id/date_prefix).
    # The icearea subset itself is never self-masked.
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml --export-only --ice-mask

    # Generate + export with ice masking:
    python -m dbof.cli.run_all_subsets --config configs/global/run.yaml --ice-mask
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

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
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NETCDF_BASE = "/mnt/tank/Oceanography/data/OGCM/LLC/Fronts"
DEFAULT_CONFIGS_DIR = "configs"


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def discover_configs(configs_dir: str) -> list[Path]:
    """Return sorted list of YAML config files in *configs_dir* (non-recursive).

    Skips the ``data_access/`` subdirectory and any dotfiles.
    """
    configs_path = Path(configs_dir)
    if not configs_path.is_dir():
        log.warning("Configs directory does not exist: %s", configs_dir)
        return []

    yamls = sorted(
        p for p in configs_path.glob("*.yaml")
        if not p.name.startswith(".")
    )
    return yamls


def parse_subsets_from_config(config_path: Path) -> dict:
    """Parse a config YAML and return its metadata.

    Returns
    -------
    dict with keys:
        config_path : Path
        run_id      : str
        s3_source   : dict
        output      : dict   (s3_endpoint, bucket, folder)
        date_iterations : list[str]
        subsets     : dict[str, dict]  (subset_name -> subset entry)
    """
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    run_id = raw.get("run", {}).get("run_id", "unknown_run")
    s3_source = raw.get("s3_source", {})
    output = raw.get("output", {})
    date_iterations = raw.get("data", {}).get("date_iterations", [])
    subsets = raw.get("subsets", {})

    return {
        "config_path": config_path,
        "run_id": run_id,
        "s3_source": s3_source,
        "output": output,
        "date_iterations": date_iterations,
        "subsets": subsets,
    }


def get_channels_for_subset(subset_entry: dict) -> list[str]:
    """Extract the ordered channel list from a subset config entry.

    Handles ``depth_suffixes`` expansion: if the entry contains a
    ``depth_suffixes`` key, each base name in ``compute_features_channels``
    is expanded to ``{base}_{suffix}`` for every suffix.  Entries in
    ``extra_channels`` are appended unchanged.
    """
    from dbof.global_dataset_creation.subset_definitions import expand_channels_with_suffixes

    model_channels = subset_entry.get("model_data_feature_channels", []) or []
    computed_channels = expand_channels_with_suffixes(
        channels=subset_entry.get("compute_features_channels", []) or [],
        depth_suffixes=subset_entry.get("depth_suffixes"),
        extra_channels=subset_entry.get("extra_channels"),
    )
    return model_channels + computed_channels


# ---------------------------------------------------------------------------
# Ice-mask dependency: ensure icearea.zarr exists
# ---------------------------------------------------------------------------

ICE_MASK_DATASET = "icearea.zarr"


def _icearea_store_exists(
    s3_endpoint: str, bucket: str, folder: str, run_id: str, date_prefix: str,
) -> bool:
    """Check whether icearea.zarr exists at the expected S3 path."""
    from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix
    from dbof.io.filesystems import create_s3_filesystems

    path = make_run_prefix(bucket, folder, run_id, ICE_MASK_DATASET,
                           date_prefix=date_prefix)
    # Strip s3:// for fs.exists()
    s3_key = path.removeprefix("s3://")
    _, fs_sync = create_s3_filesystems(s3_endpoint)
    return fs_sync.exists(s3_key)


def ensure_icearea(
    config_info: dict,
    run_id: str,
    dry_run: bool = False,
) -> None:
    """Generate the icearea subset if its Zarr store is missing for any date.

    Called automatically when ``--ice-mask`` is set so that the export
    phase can always find ``icearea.zarr``.
    """
    output = config_info["output"]
    s3_endpoint = output.get("s3_endpoint", "https://s3-west.nrp-nautilus.io")
    bucket = output.get("bucket", "dbof/")
    folder = output.get("folder", "depth_fields/")
    date_iterations = config_info["date_iterations"]

    date_prefixes = [_date_to_prefix(d) for d in date_iterations]

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

    run_generate(
        config_path=str(config_info["config_path"]),
        subset="icearea",
        run_id=run_id,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Phase 1: Generate zarr stores
# ---------------------------------------------------------------------------

def run_generate(config_path: str, subset: str, run_id: str = None,
                 dry_run: bool = False):
    """Call generate_global.main() for one subset."""
    log.info("=" * 60)
    log.info("GENERATE  config=%s  subset=%s", config_path, subset)
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY RUN] Would run generate_global.main("
                 "config_file=%r, subset=%r, run_id=%r)", config_path, subset, run_id)
        return

    from dbof.cli.generate_global import main as generate_main
    generate_main(config_file=str(config_path), run_id=run_id, subset=subset)


# ---------------------------------------------------------------------------
# Phase 2: Export zarr → per-variable NetCDF
# ---------------------------------------------------------------------------

from dbof.global_dataset_creation.iterations import (
    date_to_run_id as _date_to_prefix,
    prefix_to_filename_date as _prefix_to_filename_date,
)


def run_export_channel(
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
):
    """Call zarr_to_netcdf.main() for a single channel, producing one .nc file.

    Output filename follows the convention::

        LLC4320_{date}_{channel}_{run_id}.nc

    where *date* is formatted as ``2012-11-09T12_00_00`` from the
    *date_prefix*.
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


def run_export_subset(
    config_info: dict,
    subset_name: str,
    subset_entry: dict,
    netcdf_base: str,
    run_id_override: str = None,
    dry_run: bool = False,
    ice_mask: bool = False,
):
    """Export all channels in one subset to individual NetCDF files.

    Iterates over each date_prefix (derived from date_iterations in the
    config) and exports each channel as a separate .nc file.
    """
    run_id = run_id_override or config_info["run_id"]
    output = config_info["output"]
    s3_endpoint = output.get("s3_endpoint", "https://s3-west.nrp-nautilus.io")
    bucket = output.get("bucket", "dbof/")
    folder = output.get("folder", "depth_fields/")
    dataset_name = subset_entry.get("dataset_name", f"{subset_name}.zarr")
    date_iterations = config_info["date_iterations"]

    # Convert date strings to date_prefix strings.
    date_prefixes = [_date_to_prefix(d) for d in date_iterations]

    channels = get_channels_for_subset(subset_entry)
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
                run_export_channel(
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
    p = argparse.ArgumentParser(
        description=(
            "Batch driver: generate all zarr subsets then export each "
            "channel to a separate NetCDF file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", nargs="+", metavar="YAML",
        help=("One or more config YAML paths. Default: auto-discover "
              f"all .yaml files in {DEFAULT_CONFIGS_DIR}/."),
    )
    p.add_argument(
        "--configs-dir", default=DEFAULT_CONFIGS_DIR,
        help=f"Directory to scan for config YAMLs (default: {DEFAULT_CONFIGS_DIR}).",
    )
    p.add_argument(
        "--subsets", nargs="+", metavar="NAME",
        help="Only process these subset(s). Default: all subsets in each config.",
    )
    p.add_argument(
        "--run-id", default=None,
        help="Override run_id from config (applies to all configs).",
    )
    p.add_argument(
        "--netcdf-base", default=DEFAULT_NETCDF_BASE,
        help=f"Base directory for NetCDF output (default: {DEFAULT_NETCDF_BASE}).",
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
        help=("Mask ice-covered points (SIarea > 0) with NaN before "
              "writing NetCDF.  The icearea subset itself is never masked."),
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # ---- Discover configs ----
    if args.config:
        config_paths = [Path(c) for c in args.config]
    else:
        config_paths = discover_configs(args.configs_dir)

    if not config_paths:
        log.error("No config YAML files found.")
        sys.exit(1)

    log.info("Found %d config file(s): %s",
             len(config_paths), [str(p) for p in config_paths])

    # ---- Parse all configs ----
    configs = []
    for cp in config_paths:
        try:
            info = parse_subsets_from_config(cp)
            configs.append(info)
        except Exception:
            log.exception("Failed to parse config: %s", cp)

    if not configs:
        log.error("No valid configs parsed.")
        sys.exit(1)

    # ---- Build the work plan ----
    # Import the dispatch table to know which subsets are valid for
    # generate_global.  Subsets not in the table are export-only
    # (they were generated by a different pipeline).
    from dbof.preprocessing.depth_subsets import SUBSET_COMPUTE_FNS

    work = []  # list of (config_info, subset_name, subset_entry, can_generate)
    for info in configs:
        for subset_name, subset_entry in info["subsets"].items():
            if args.subsets and subset_name not in args.subsets:
                continue
            can_generate = subset_name in SUBSET_COMPUTE_FNS
            work.append((info, subset_name, subset_entry, can_generate))

    log.info("Work plan: %d subset(s) across %d config(s)",
             len(work), len(configs))
    for info, sn, se, cg in work:
        channels = get_channels_for_subset(se)
        tag = "generate+export" if cg else "export-only"
        log.info("  [%s] %s :: %s  (%d channels)",
                 tag, info["config_path"].name, sn, len(channels))

    wall_start = time.monotonic()

    # ---- Pre-flight: ensure icearea.zarr exists when ice masking ----
    if args.ice_mask and not args.export_only:
        # Check every config in the work plan for missing icearea stores.
        # Use a set to avoid duplicate checks when multiple subsets share
        # the same config.
        checked_configs = set()
        for info, subset_name, subset_entry, can_generate in work:
            config_key = str(info["config_path"])
            if config_key in checked_configs:
                continue
            checked_configs.add(config_key)

            run_id = args.run_id or info["run_id"]
            try:
                ensure_icearea(
                    config_info=info,
                    run_id=run_id,
                    dry_run=args.dry_run,
                )
            except Exception:
                log.exception(
                    "FAILED to ensure icearea.zarr for config %s. "
                    "Ice masking may fail during export.", config_key
                )

    # ---- Phase 1: Generate ----
    if not args.export_only:
        log.info("")
        log.info("=" * 60)
        log.info("PHASE 1: GENERATE ZARR STORES")
        log.info("=" * 60)

        for info, subset_name, subset_entry, can_generate in work:
            if not can_generate:
                log.info("Skipping generate for '%s' (not in SUBSET_COMPUTE_FNS)",
                         subset_name)
                continue
            try:
                run_generate(
                    config_path=str(info["config_path"]),
                    subset=subset_name,
                    run_id=args.run_id,
                    dry_run=args.dry_run,
                )
            except Exception:
                log.exception("FAILED to generate subset '%s' from %s",
                              subset_name, info["config_path"])

    # ---- Clear stale S3 filesystem cache between phases ----
    # Phase 1 uses Dask distributed, which creates its own asyncio event
    # loop.  After the Dask cluster shuts down, any s3fs instances cached
    # by fsspec are still bound to that (now-dead) loop.  Clearing the
    # cache forces Phase 2 to create fresh connections.
    if not args.export_only and not args.generate_only:
        try:
            from s3fs import S3FileSystem
            S3FileSystem.clear_instance_cache()
            log.info("Cleared s3fs instance cache between phases.")
        except Exception:
            pass  # Best-effort; not all installations expose this.

    # ---- Phase 2: Export ----
    if not args.generate_only:
        # Warn early if ice masking was requested but icearea may be missing.
        if args.ice_mask and args.export_only:
            log.warning(
                "Ice masking requested with --export-only.  If icearea.zarr "
                "has not been generated yet, every channel export will fail.  "
                "Run without --export-only to auto-generate it, or generate "
                "it separately with --subsets icearea --generate-only."
            )

        log.info("")
        log.info("=" * 60)
        log.info("PHASE 2: EXPORT ZARR -> PER-VARIABLE NETCDF")
        log.info("=" * 60)

        for info, subset_name, subset_entry, can_generate in work:
            try:
                run_export_subset(
                    config_info=info,
                    subset_name=subset_name,
                    subset_entry=subset_entry,
                    netcdf_base=args.netcdf_base,
                    run_id_override=args.run_id,
                    dry_run=args.dry_run,
                    ice_mask=args.ice_mask,
                )
            except Exception:
                log.exception("FAILED to export subset '%s' from %s",
                              subset_name, info["config_path"])

    wall_elapsed = time.monotonic() - wall_start
    log.info("")
    log.info("=" * 60)
    log.info("All done.  Total wall-clock time: %.1f s (%.2f h)",
             wall_elapsed, wall_elapsed / 3600)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
