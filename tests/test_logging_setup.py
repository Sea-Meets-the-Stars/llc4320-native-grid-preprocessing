"""Unit tests for the shared run-logging setup (dbof.utils.logging)."""
import logging
from pathlib import Path

import pytest

from dbof.config import BaseRunConfig
from dbof.utils.logging import generate_logging


@pytest.fixture
def reset_root_logging():
    """Close/remove handlers added by basicConfig(force=True) so the temp log
    file is released (Windows) and root logging is clean for later tests."""
    yield
    root = logging.getLogger()
    for h in root.handlers[:]:
        try:
            h.close()
        finally:
            root.removeHandler(h)


def test_creates_log_file_under_run_dir(tmp_path, reset_root_logging):
    run = BaseRunConfig(run_id="run42", log_dir=str(tmp_path))
    log_file = generate_logging(run, log_filename="x.log")
    assert log_file == (tmp_path / "run42" / "x.log").resolve()
    assert log_file.exists()


def test_returns_path_and_default_filename(tmp_path, reset_root_logging):
    run = BaseRunConfig(run_id="r", log_dir=str(tmp_path))
    log_file = generate_logging(run)
    assert isinstance(log_file, Path)
    assert log_file.name == "pipeline.log"


def test_relative_log_dir_resolved_against_config_dir(tmp_path, reset_root_logging):
    run = BaseRunConfig(run_id="r", log_dir="logs")  # relative
    log_file = generate_logging(run, config_dir=tmp_path)
    assert log_file == (tmp_path / "logs" / "r" / "pipeline.log").resolve()
    assert log_file.exists()


def test_rerun_same_run_id_does_not_raise(tmp_path, reset_root_logging):
    run = BaseRunConfig(run_id="dup", log_dir=str(tmp_path))
    generate_logging(run, log_filename="a.log")
    log_file = generate_logging(run, log_filename="b.log")  # exist_ok=True
    assert log_file.exists()


def test_accepts_subclass_of_base_run_config(tmp_path, reset_root_logging):
    from dbof.cutout_dataset_creation.config import RunConfig
    run = RunConfig(run_id="sub", log_dir=str(tmp_path))
    log_file = generate_logging(run, log_filename="c.log")
    assert log_file.exists()
