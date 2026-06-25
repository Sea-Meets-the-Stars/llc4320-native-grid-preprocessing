"""Unit tests for the cutout pipeline config loader (run-from-globals)."""
from pathlib import Path

import pytest

import dbof.cutout_dataset_creation.config as config
from dbof.cutout_dataset_creation.config import (
    load_config,
    InputConfig,
    GridAccessConfig,
    JobConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "cutouts" / "run" / "run_from_globals_example.yaml"


def test_example_config_loads():
    """The shipped example config parses into a valid JobConfig."""
    cfg = load_config(str(EXAMPLE_CONFIG))

    assert isinstance(cfg, JobConfig)
    assert isinstance(cfg.input, InputConfig)

    # Source path + explicit timestamps as written in the example.
    assert cfg.input.folder == "surface_fields/global_SURF_test01"
    assert cfg.input.bucket == "dbof"
    assert cfg.input.s3_endpoint == "https://s3-west.nrp-nautilus.io"
    assert cfg.input.date_prefixes == ["20121109_120000"]

    # grid_access parsed into the nested dataclass (example matches defaults).
    assert isinstance(cfg.input.grid_access, GridAccessConfig)
    assert cfg.input.grid_access == GridAccessConfig()

    # OSN DataConfig is gone; JobConfig no longer carries `data`.
    assert not hasattr(config, "DataConfig")
    assert not hasattr(cfg, "data")


def test_input_defaults():
    """date_prefixes is optional (None = all dates); grid_access defaults."""
    ic = InputConfig(folder="surface_fields/run")
    assert ic.date_prefixes is None
    assert ic.grid_access == GridAccessConfig()


def test_missing_folder_raises(tmp_path):
    """An input section without `folder` fails with a clear message."""
    cfg_file = tmp_path / "no_folder.yaml"
    cfg_file.write_text("input:\n  bucket: dbof\n")
    with pytest.raises(ValueError, match="input.folder"):
        load_config(str(cfg_file))


def test_missing_input_section_raises(tmp_path):
    """A config with no input section at all fails on the folder check."""
    cfg_file = tmp_path / "no_input.yaml"
    cfg_file.write_text("run:\n  run_id: x\n")
    with pytest.raises(ValueError, match="input.folder"):
        load_config(str(cfg_file))


def test_feature_channels_loaded_from_top_level():
    """Top-level `feature_channels` is read into FeaturesConfig."""
    cfg = load_config(str(EXAMPLE_CONFIG))
    assert cfg.features.feature_channels == [
        "Eta", "Salt", "Theta", "U", "V", "W", "gradb2", "SIarea",
    ]


def test_feature_channels_default_when_omitted(tmp_path):
    """Omitting `feature_channels` falls back to the default channel list."""
    cfg_file = tmp_path / "no_features.yaml"
    cfg_file.write_text("run:\n  run_id: x\ninput:\n  folder: surface_fields/run\n")
    cfg = load_config(str(cfg_file))
    assert cfg.features.feature_channels == ["Eta", "Salt", "Theta", "U", "V", "W", "gradb2", "SIarea"]


def test_missing_required_channel_raises(tmp_path):
    """A config whose feature_channels omits a required channel fails."""
    cfg_file = tmp_path / "missing_required.yaml"
    cfg_file.write_text(
        "run:\n  run_id: x\n"
        "input:\n  folder: surface_fields/run\n"
        "feature_channels:\n  - Theta\n  - gradb2\n"  # missing SIarea
    )
    with pytest.raises(ValueError, match="SIarea"):
        load_config(str(cfg_file))
