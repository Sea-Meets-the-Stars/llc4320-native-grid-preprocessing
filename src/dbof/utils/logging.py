"""
Shared logging configuration for all pipeline scripts.
"""
import logging
import sys
from pathlib import Path

import dbof.dataset_creation.config as config


def generate_logging(
    cfg: config.JobConfig,
    config_dir: Path = None,
    log_filename: str = "pipeline.log",
) -> None:
    """
    Configure file + stdout logging for a pipeline run.

    Parameters
    ----------
    cfg : config.JobConfig
        Pipeline configuration (must have ``cfg.run.log_dir`` and
        ``cfg.run.run_id``).
    config_dir : Path, optional
        If *log_dir* is relative, resolve it against this directory.
        Used by the depth pipeline where the YAML sits in a sub-folder.
    log_filename : str, default ``"pipeline.log"``
        Name of the log file inside the run directory.
    """
    log_path = Path(cfg.run.log_dir).expanduser()

    if not log_path.is_absolute() and config_dir is not None:
        log_path = (config_dir / log_path).resolve()
    else:
        log_path = log_path.resolve()

    run_dir = log_path / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / log_filename

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
