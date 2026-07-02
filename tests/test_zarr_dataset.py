"""Unit tests for zarr_dataset path helpers."""
from dbof.cutout_dataset_creation.zarr_dataset import make_run_prefix


def test_make_run_prefix_strips_and_joins():
    assert make_run_prefix("dbof/", "/folder/sub/", "run", "d.zarr") == "s3://dbof/folder/sub/run/d.zarr"
    assert make_run_prefix("dbof", "folder", "run", "d.zarr") == "s3://dbof/folder/run/d.zarr"
