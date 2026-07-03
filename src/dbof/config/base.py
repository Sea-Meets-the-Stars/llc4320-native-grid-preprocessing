"""Base config classes shared across pipelines.

Every pipeline's run configuration provides a run id and a log directory.
Defining them once here lets shared tooling (e.g. logging setup) accept any
pipeline's run config, and lets each pipeline extend the base with its own
fields.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseRunConfig:
    """Run identity + log location common to all pipelines."""
    run_id: str
    log_dir: str = "./logs"
