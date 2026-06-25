#!/usr/bin/env python3
"""
CLI entry point for transferring LLC4320 variables from a local zarr store to
S3.  This is a thin dispatcher; the real work lives in the
:mod:`dbof.transfer` package.

#NOTE: this script is designed to be run from the MIT machines.

One unified pipeline (:mod:`dbof.transfer.pipeline`) handles both spatial
extents, selected by ``transfer.mode`` in the YAML config:

* ``all``    -- transfer the whole native dataset.  Static grid variables go to
  ``{folder}/grid.zarr`` (once); time-varying fields go to
  ``{folder}/{YYYYMMDDTHH}.zarr`` (per-date).
* ``chunks`` -- transfer a single native 720x720 chunk (all depths) surrounding
  ``transfer.location`` (lat/lon).  Static grid goes to
  ``{chunks_prefix}/{chunk_name}/grid.zarr`` (once); time-varying fields go to
  ``{chunks_prefix}/{chunk_name}/{YYYYMMDD_HHMMSS}/`` (per-date).

Both extents transfer the dates in ``data.date_iterations`` (looped in one run),
or a single date via ``--date``.

CLI usage
---------
    # Full dataset (mode: all), all configured dates in one run:
    transfer-timestep --config configs/transfer/run.yaml --init-store
    transfer-timestep --config configs/transfer/run.yaml --subset static --init-store
    transfer-timestep --config configs/transfer/run.yaml --subset time --variables Theta
    transfer-timestep --config configs/transfer/run.yaml --date "2012-11-09 12:00:00"

    # Single chunk (mode: chunks), all configured dates in one run:
    transfer-timestep --config configs/transfer/run_chunks_monterey_bay.yaml --init-store

With ``--subset all`` the static grid is written once; subsequent dates write
only the time-varying fields.  ``--date`` / ``--subset`` / ``--variables`` apply
to both modes.
"""

import argparse
import logging

from dbof.transfer import config as config
from dbof.transfer import pipeline
from dbof.utils.logging import generate_logging


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer LLC4320 variables from a local zarr store to S3 Zarr. "
            "Mode (all-data vs single-chunk) is selected by transfer.mode in "
            "the config."
        ),
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config file (e.g. configs/transfer/run.yaml).",
    )
    parser.add_argument(
        "--date", default=None,
        help=(
            "[all mode] Override date for time-varying variables (ISO, e.g. "
            "'2012-11-09 12:00:00').  Overrides the first data.date_iterations entry."
        ),
    )
    parser.add_argument(
        "--init-store", action="store_true",
        help="Initialize/reset output stores before writing (wipes existing data).",
    )
    parser.add_argument(
        "--subset", choices=["static", "time", "all"], default="all",
        help="[all mode] Which group to transfer: 'static', 'time', or 'all'.",
    )
    parser.add_argument(
        "--variables", default=None,
        help="[all mode] Comma-separated variable names to transfer, overriding the config.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip variables that already exist in the target zarr store.",
    )
    return parser.parse_args()


def main(config_file: str = None, date: str = None, init_store: bool = False,
         subset: str = "all", skip_existing: bool = False,
         variables_override: list = None) -> None:
    """Load the config and dispatch to the requested transfer mode.

    Can be called from the CLI (no args -> reads ``sys.argv``) or directly from
    Python by passing arguments explicitly.
    """
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        date = date or cli.date
        init_store = init_store or cli.init_store
        subset = cli.subset
        skip_existing = cli.skip_existing
        if cli.variables is not None:
            variables_override = [v.strip() for v in cli.variables.split(",")]

    cfg = config.load_config(config_file)

    generate_logging(cfg.run, log_filename="transfer_llc4320.log")
    logging.info(f"Config loaded from: {config_file} (mode={cfg.transfer.mode})")

    # One pipeline handles both spatial extents (full dataset vs single chunk),
    # selected by transfer.mode inside the config; the date loop lives in run().
    pipeline.run(
        cfg=cfg,
        init_store=init_store,
        subset=subset,
        skip_existing=skip_existing,
        date_override=date,
        variables_override=variables_override,
    )


if __name__ == "__main__":
    main()
