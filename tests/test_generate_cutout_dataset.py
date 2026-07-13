"""Unit tests for generate_cutout_dataset helpers."""
import pytest

from dbof.cutout_dataset_creation.config import InputConfig, GridAccessConfig
from dbof.cutout_dataset_creation.global_input import resolve_input_locations, resolve_date_prefixes


def test_resolve_input_locations_basic():
    ic = InputConfig(
        folder="surface_fields/run01",
        bucket="dbof",
        grid_access=GridAccessConfig(bucket="dbof", folder="grids", dataset_name="g.zarr"),
    )
    base, grid_uri = resolve_input_locations(ic)
    assert base == "s3://dbof/surface_fields/run01"
    assert grid_uri == "s3://dbof/grids/g.zarr"


def test_resolve_input_locations_strips_slashes():
    ic = InputConfig(
        folder="/surface_fields/run01/",
        bucket="dbof/",
        grid_access=GridAccessConfig(bucket="/dbof/", folder="/grids/", dataset_name="g.zarr"),
    )
    base, grid_uri = resolve_input_locations(ic)
    assert base == "s3://dbof/surface_fields/run01"
    assert grid_uri == "s3://dbof/grids/g.zarr"


class _FakeFS:
    def __init__(self, entries):
        self._entries = entries

    def ls(self, path, detail=True):
        return self._entries


def test_resolve_date_prefixes_returns_configured_list():
    ic = InputConfig(folder="x", date_prefixes=["20121109_120000"])
    assert resolve_date_prefixes(ic, fs=None) == ["20121109_120000"]


def test_resolve_date_prefixes_discovers_folders_when_none():
    entries = [
        {"name": "dbof/surface_fields/run/20121110_120000", "type": "directory"},
        {"name": "dbof/surface_fields/run/20121109_120000", "type": "directory"},
        {"name": "dbof/surface_fields/run/notes.txt", "type": "file"},
    ]
    ic = InputConfig(folder="surface_fields/run", bucket="dbof")
    assert resolve_date_prefixes(ic, _FakeFS(entries)) == ["20121109_120000", "20121110_120000"]


def test_verify_feature_channels_raises_on_missing(monkeypatch):
    import dbof.cutout_dataset_creation.global_input as gi
    monkeypatch.setattr(gi, "available_channels", lambda *a, **k: {"Theta", "Salt"})
    with pytest.raises(ValueError, match="gradb2"):
        gi.verify_feature_channels(InputConfig(folder="x"), "d", ["Theta", "gradb2"], None, None)


def test_verify_feature_channels_ok(monkeypatch):
    import dbof.cutout_dataset_creation.global_input as gi
    monkeypatch.setattr(gi, "available_channels", lambda *a, **k: {"Theta", "Salt"})
    gi.verify_feature_channels(InputConfig(folder="x"), "d", ["Theta"], None, None)


def test_open_feature_readers_maps_channels(monkeypatch):
    import dbof.cutout_dataset_creation.global_input as gi

    class _FakeReader:
        def __init__(self, chans):
            self.channel_names = chans

    native = _FakeReader(["Theta", "Salt"])
    frontal = _FakeReader(["gradb2"])
    monkeypatch.setattr(gi, "_open_subset_readers",
                        lambda *a, **k: {"native_fields.zarr": native, "frontal_structure.zarr": frontal})

    out = gi.open_feature_readers(InputConfig(folder="x"), "d", ["Theta", "gradb2"], None, None)
    assert out["Theta"] is native
    assert out["gradb2"] is frontal
    assert "Salt" not in out  # present in store but not requested


def test_load_snapshot_features_reads_requested(monkeypatch):
    import numpy as np
    import dbof.cutout_dataset_creation.global_input as gi

    class _FakeReader:
        def get_channel_snapshot(self, ch):
            return np.full((2, 3), float(len(ch)))

    r = _FakeReader()
    monkeypatch.setattr(gi, "open_feature_readers", lambda *a, **k: {"Theta": r, "gradb2": r})
    ds = gi.load_snapshot_features(InputConfig(folder="x"), "d", ["Theta", "gradb2"], None, None)
    assert set(ds.data_vars) == {"Theta", "gradb2"}
    assert ds["Theta"].dims == ("j", "i")
    assert ds["Theta"].shape == (2, 3)


def test_load_snapshot_features_raises_on_missing(monkeypatch):
    import dbof.cutout_dataset_creation.global_input as gi
    monkeypatch.setattr(gi, "open_feature_readers", lambda *a, **k: {"Theta": object()})
    with pytest.raises(ValueError, match="gradb2"):
        gi.load_snapshot_features(InputConfig(folder="x"), "d", ["Theta", "gradb2"], None, None)


def test_verify_required_channels_raises_when_absent(monkeypatch):
    import dbof.cutout_dataset_creation.global_input as gi
    monkeypatch.setattr(gi, "available_channels", lambda *a, **k: {"gradb2", "Theta"})  # no SIarea
    with pytest.raises(ValueError, match="SIarea"):
        gi.verify_required_channels(InputConfig(folder="x"), "d", None, None)


def test_verify_required_channels_ok(monkeypatch):
    import dbof.cutout_dataset_creation.global_input as gi
    monkeypatch.setattr(gi, "available_channels", lambda *a, **k: {"gradb2", "SIarea", "Theta"})
    gi.verify_required_channels(InputConfig(folder="x"), "d", None, None)
