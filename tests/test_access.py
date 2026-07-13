"""Unit tests for the cutout dataset access reader + data-access config loader."""
import dask.array as da
import numpy as np
import pandas as pd
import pytest

import dbof.cutout_dataset_access.access as access
from dbof.cutout_dataset_creation.config import load_data_access_config


def _write_access_yaml(tmp_path, run_id="cfg_run"):
    p = tmp_path / "da.yaml"
    p.write_text(
        "data_access:\n"
        f"  run_id: {run_id}\n"
        "  folder: f\n"
        "  bucket: dbof\n"
        "  s3_endpoint: x\n"
        "  dataset_name: d.zarr\n"
    )
    return str(p)


# --- access config loader --------------------------------------------------

def test_load_data_access_config_ignores_feature_channels(tmp_path):
    p = tmp_path / "da.yaml"
    p.write_text(
        "data_access:\n"
        "  bucket: dbof\n"
        "  folder: test_data_for_cutouts/run\n"
        "  run_id: r\n"
        "  s3_endpoint: https://s3-west.nrp-nautilus.io\n"
        "  dataset_name: cutout_dataset_creation.zarr\n"
        "  feature_channels: [Theta]\n"
    )
    cfg = load_data_access_config(str(p))
    assert cfg.run_id == "r"
    assert cfg.folder == "test_data_for_cutouts/run"
    assert cfg.bucket == "dbof"
    assert not hasattr(cfg, "feature_channels")


def test_load_data_access_config_requires_all_keys(tmp_path):
    p = tmp_path / "da.yaml"
    p.write_text("data_access:\n  bucket: dbof\n  run_id: r\n")  # missing folder/endpoint/dataset_name
    with pytest.raises(ValueError, match="missing required keys"):
        load_data_access_config(str(p))


# --- combined reader: linking + hole handling ------------------------------

class _FakeReader:
    channel_names = ["A", "B"]

    def __init__(self, *a, **k):
        pass

    def full_dataset_as_dask(self):
        images = da.from_array(np.arange(4 * 2).reshape(4, 2, 1, 1).astype("float32"),
                               chunks=(2, 2, 1, 1))
        ids = da.from_array(np.array([b"id0", b"", b"id2", b"id3"], dtype="S32"))
        return images, ids, ids != b""


class _FakeMetaReader:
    def read(self):
        # out-of-order; id9 has no image, id3 has no metadata row
        return pd.DataFrame({
            "image_id": [b"id2", b"id0", b"id9"],
            "center_lat": [2.0, 0.0, 9.0],
        })


def test_load_cutout_dataset_links_and_drops_holes(monkeypatch, tmp_path):
    monkeypatch.setattr(access.zarr_dataset, "ZarrDatasetReader", _FakeReader)
    monkeypatch.setattr(access.metadata, "create_metadata_reader", lambda *a, **k: _FakeMetaReader())
    monkeypatch.setattr(access, "create_s3_filesystems", lambda endpoint: (None, None))

    ds = access.load_cutout_dataset(_write_access_yaml(tmp_path))

    # valid ids: id0(pos0), id2(pos2), id3(pos3). Metadata has id0, id2 (not id3; id9 extra).
    # Keep intersection in zarr order: id0, id2.
    assert ds.channel_names == ["A", "B"]
    assert list(ds.metadata["image_id"]) == ["id0", "id2"]
    assert list(ds.metadata["center_lat"]) == [0.0, 2.0]

    imgs = np.asarray(ds.images)
    assert imgs.shape == (2, 2, 1, 1)
    base = np.arange(4 * 2).reshape(4, 2, 1, 1).astype("float32")
    np.testing.assert_array_equal(imgs[0], base[0])  # id0 -> zarr pos 0
    np.testing.assert_array_equal(imgs[1], base[2])  # id2 -> zarr pos 2


def test_load_cutout_dataset_run_id_override(monkeypatch, tmp_path):
    captured = {}

    class _CaptureReader(_FakeReader):
        def __init__(self, *a, **k):
            captured["zarr_run_id"] = k.get("run_id")

    def _fake_meta(bucket, folder, run_id, fs_sync):
        captured["meta_run_id"] = run_id
        return _FakeMetaReader()

    monkeypatch.setattr(access.zarr_dataset, "ZarrDatasetReader", _CaptureReader)
    monkeypatch.setattr(access.metadata, "create_metadata_reader", _fake_meta)
    monkeypatch.setattr(access, "create_s3_filesystems", lambda endpoint: (None, None))

    access.load_cutout_dataset(_write_access_yaml(tmp_path, run_id="cfg_run"),
                               run_id="override_run")

    assert captured["zarr_run_id"] == "override_run"
    assert captured["meta_run_id"] == "override_run"
