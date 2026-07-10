"""
Run-metadata generation and persistence for the global pipeline.

This module builds and persists the ``run_meta.yaml`` record 
describing *what* a run produced (run_id, pipeline,
configs, per-subset channels, git commit, timestamps).
"""

import logging
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


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
        Path returned by :func:`dbof.global_dataset_creation.logging.setup_logging`.
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
