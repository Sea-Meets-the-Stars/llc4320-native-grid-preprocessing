"""
Logging and run-metadata persistence for the global pipeline.
"""

import logging
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _find_repo_root() -> Path:
    """Locate the repository root by looking for a ``.git`` directory."""
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return Path(root)
    except Exception:
        # Fallback: walk upward from this file looking for .git
        p = Path(__file__).resolve().parent
        for parent in [p] + list(p.parents):
            if (parent / ".git").exists():
                return parent
        # Last resort: current working directory
        return Path.cwd()


def setup_logging(cfg) -> Path:
    """
    Configure file + stdout logging for a global pipeline run.

    Log directory layout::

        {log_dir}/{run_id}/
            native_fields_20260608_142210.log   ← one log per invocation
            kinematic_20260608_143055.log
            run_meta.yaml

    If ``cfg.run.log_dir`` is relative it is resolved against the
    **repository root** (detected via ``git rev-parse``).  This keeps log
    output in a predictable location regardless of which directory the
    pipeline is launched from (CLI at the repo root, Jupyter notebook, etc.).

    The run directory (``{log_dir}/{run_id}/``) may already exist — multiple
    subset runs share the same ``run_id`` directory.  Each invocation writes a
    **new, timestamped** log file (``{subset(s)}_{YYYYMMDD_HHMMSS}.log``, run
    time in UTC), so re-runs never clobber or interleave with previous logs.
    Whether work is actually redone is decided by the S3 zarr-store existence
    check in ``generate_global`` (and the ``--clobber`` flag), not by this log.

    Parameters
    ----------
    cfg : GlobalJobConfig
        Full pipeline configuration.  ``cfg.active_subsets`` determines the
        log filename.

    Returns
    -------
    log_file : Path
        Absolute path to the log file created for this invocation.
    """
    log_path = Path(cfg.run.log_dir).expanduser()

    if not log_path.is_absolute():
        log_path = (_find_repo_root() / log_path).resolve()
    else:
        log_path = log_path.resolve()

    run_dir = log_path / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build log filename from subset(s) + a run-time UTC stamp, so every
    # invocation gets its own file ({subset(s)}_{YYYYMMDD_HHMMSS}.log) instead
    # of clobbering or interleaving with previous runs.
    if len(cfg.active_subsets) == 1:
        base = cfg.active_subsets[0]
    else:
        base = "_".join(cfg.active_subsets)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base}_{run_stamp}.log"

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


def _git_commit_hash() -> str:
    """Return the short git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_run_metadata(cfg, log_file: Path, fs=None) -> Path:
    """
    Write a YAML metadata file capturing the full run specification.

    The file is written next to the log file (same directory) and,
    optionally, pushed to S3 alongside the output data.

    Parameters
    ----------
    cfg : GlobalJobConfig
        Full pipeline configuration.
    log_file : Path
        Path returned by :func:`setup_logging`.
    fs : s3fs.S3FileSystem, optional
        If provided, also write the metadata to
        ``s3://{bucket}/{folder}/{run_id}/run_meta.yaml``.

    Returns
    -------
    meta_path : Path
        Local path to the metadata file.
    """
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "pipeline": cfg.pipeline,
        "run_id": cfg.run.run_id,
        "active_subsets": list(cfg.active_subsets),
        "depth_suffixes": cfg.depth_suffixes,
        "date_iterations": cfg.data.date_iterations,
        "run": asdict(cfg.run),
        "data": asdict(cfg.data),
        "output": asdict(cfg.output),
        "runtime": asdict(cfg.runtime),
    }

    # -- Local copy (next to log file) --
    meta_path = log_file.parent / "run_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Run metadata saved to {meta_path}")

    # -- S3 copy (alongside output data) --
    if fs is not None:
        s3_key = (
            f"{cfg.output.bucket}{cfg.output.folder}"
            f"{cfg.run.run_id}/run_meta.yaml"
        )
        try:
            with fs.open(s3_key, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Run metadata pushed to s3://{s3_key}")
        except Exception:
            logging.warning(f"Failed to push run metadata to s3://{s3_key}",
                            exc_info=True)

    return meta_path
