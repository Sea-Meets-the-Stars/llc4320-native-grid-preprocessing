"""Unit tests for the shared base config (dbof.config)."""
import dataclasses

import pytest

import dbof.config
from dbof.config import BaseRunConfig
from dbof.config.base import BaseRunConfig as BaseFromModule


def test_package_reexports_base():
    assert BaseRunConfig is BaseFromModule
    assert "BaseRunConfig" in dbof.config.__all__


def test_defaults_and_required_fields():
    rc = BaseRunConfig(run_id="abc")
    assert rc.run_id == "abc"
    assert rc.log_dir == "./logs"
    with pytest.raises(TypeError):
        BaseRunConfig()  # run_id is required


def test_is_frozen_dataclass():
    assert dataclasses.is_dataclass(BaseRunConfig)
    rc = BaseRunConfig(run_id="abc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rc.run_id = "x"


def test_cutout_runconfig_inherits_base():
    from dbof.cutout_dataset_creation.config import RunConfig
    assert issubclass(RunConfig, BaseRunConfig)
    rc = RunConfig(run_id="r")
    assert rc.log_dir == "./logs"
