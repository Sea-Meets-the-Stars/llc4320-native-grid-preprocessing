"""
Shared logging configuration for all pipeline scripts.
"""
import logging
import sys
from pathlib import Path

from dbof.config import BaseRunConfig


def generate_logging(
    run: BaseRunConfig,
    config_dir: Path = None,
    log_filename: str = "pipeline.log",
) -> Path:
    """
    Configure file + stdout logging for a pipeline run.

    Parameters
    ----------
    run : BaseRunConfig
        Any run config providing ``run_id`` and ``log_dir``.
    config_dir : Path, optional
        If *log_dir* is relative, resolve it against this directory.
        Used by the depth pipeline where the YAML sits in a sub-folder.
    log_filename : str, default ``"pipeline.log"``
        Name of the log file inside the run directory.

    Returns
    -------
    Path
        Absolute path to the created log file.
    """
    log_path = Path(run.log_dir).expanduser()

    if not log_path.is_absolute() and config_dir is not None:
        log_path = (config_dir / log_path).resolve()
    else:
        log_path = log_path.resolve()

    run_dir = log_path / run.run_id
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
    return log_file
