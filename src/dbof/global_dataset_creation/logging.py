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


def _subset_meta_entry(cfg, subset_name: str, clobber: bool) -> dict:
    """Build the per-subset metadata entry (channels, suffixes, run info).

    Mirrors the channel expansion in ``generate_global`` (YAML
    ``depth_suffixes`` override > subset-definition default; surface
    subsets never get suffixes) so the metadata records what was
    *actually* produced for this subset.
    """
    from dbof.global_dataset_creation.subset_definitions import (
        expand_channels_with_suffixes,
        get_subset_definition,
    )

    defn = get_subset_definition(cfg.pipeline, subset_name)
    defn_suffixes = defn.get("depth_suffixes")
    depth_suffixes = ((cfg.depth_suffixes or defn_suffixes)
                      if defn_suffixes is not None else None)
    model_channels = list(defn.get("model_data_feature_channels", []) or [])
    compute_channels = expand_channels_with_suffixes(
        defn.get("compute_features_channels", []) or [],
        depth_suffixes=depth_suffixes,
        extra_channels=defn.get("extra_channels"),
    )

    return {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "depth_suffixes": depth_suffixes,
        "channels": model_channels + compute_channels,
        "clobber": bool(clobber),
    }


def save_run_metadata(cfg, log_file: Path, fs=None, clobber: bool = False) -> Path:
    """
    Read-merge-write a YAML metadata file capturing the run specification.

    The file holds one shared block (run_id, pipeline, dates, configs) plus
    a ``subsets:`` mapping with one entry per subset ever run under this
    run_id.  Each invocation **merges** its subsets into the existing file
    rather than overwriting it, so running subsets one at a time (as
    ``run_all_subsets`` does) accumulates a complete record.  Re-running a
    subset replaces only that subset's entry (fresh timestamp/commit).

    The file is written next to the log file (same directory) and,
    optionally, pushed to S3 alongside the output data.  If no local copy
    exists but an S3 copy does (e.g. running from a new machine), the S3
    copy is used as the merge base.

    Parameters
    ----------
    cfg : GlobalJobConfig
        Full pipeline configuration.
    log_file : Path
        Path returned by :func:`setup_logging`.
    fs : s3fs.S3FileSystem, optional
        If provided, also write the metadata to
        ``s3://{bucket}/{folder}/{run_id}/run_meta.yaml``.
    clobber : bool
        Recorded in each subset entry for provenance.

    Returns
    -------
    meta_path : Path
        Local path to the metadata file.
    """
    meta_path = log_file.parent / "run_meta.yaml"
    s3_key = (
        f"{cfg.output.bucket}{cfg.output.folder}"
        f"{cfg.run.run_id}/run_meta.yaml"
    )

    # -- Load existing metadata as the merge base (local first, then S3) --
    meta = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception:
            logging.warning(f"Could not parse existing {meta_path}; "
                            "starting fresh.", exc_info=True)
    if meta is None and fs is not None:
        try:
            if fs.exists(s3_key):
                with fs.open(s3_key, "r") as f:
                    meta = yaml.safe_load(f)
        except Exception:
            logging.warning(f"Could not read existing s3://{s3_key}; "
                            "starting fresh.", exc_info=True)
    if not isinstance(meta, dict):
        meta = {}

    # -- Shared block (latest invocation wins) --
    meta.update({
        "run_id": cfg.run.run_id,
        "pipeline": cfg.pipeline,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "date_iterations": cfg.data.date_iterations,
        "run": asdict(cfg.run),
        "data": asdict(cfg.data),
        "output": asdict(cfg.output),
        "runtime": asdict(cfg.runtime),
    })

    # -- Per-subset entries (upsert; entries for other subsets untouched) --
    subsets = meta.setdefault("subsets", {})
    for subset_name in cfg.active_subsets:
        try:
            subsets[subset_name] = _subset_meta_entry(cfg, subset_name, clobber)
        except Exception:
            logging.warning(f"Could not build metadata entry for subset "
                            f"'{subset_name}'", exc_info=True)
            subsets[subset_name] = {
                "last_run": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit_hash(),
                "clobber": bool(clobber),
            }

    # -- Local copy (next to log file) --
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Run metadata saved to {meta_path}")

    # -- S3 copy (alongside output data; merged content) --
    if fs is not None:
        try:
            with fs.open(s3_key, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Run metadata pushed to s3://{s3_key}")
        except Exception:
            logging.warning(f"Failed to push run metadata to s3://{s3_key}",
                            exc_info=True)

    return meta_path
