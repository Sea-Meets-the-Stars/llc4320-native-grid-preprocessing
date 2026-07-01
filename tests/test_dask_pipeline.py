"""Unit tests for dask_pipeline cutout-image extraction."""
import numpy as np
import xarray as xr
import torch

from dbof.cutout_dataset_creation.dask_pipeline import (
    create_image_cutout_lazy,
    create_image_cutouts_batch_as_tensors_dask,
    extract_cutout_extents_and_metadata_in_series,
    downsample_and_write_cutout_lazy,
)
from dbof.cutout_dataset_creation.processing import metadata_cols


def test_create_image_cutout_lazy_stacks_requested_features_only():
    ds = xr.Dataset({
        "A":  (("j", "i"), np.arange(20).reshape(4, 5).astype("float32")),
        "B":  (("j", "i"), (np.arange(20).reshape(4, 5) + 100).astype("float32")),
        "XC": (("j", "i"), np.zeros((4, 5), "float32")),  # grid var, must be excluded
    })
    cutout = dict(i_start=1, i_end=3, j_start=0, j_end=2)

    out = create_image_cutout_lazy(ds, ["A", "B"], cutout).compute()

    assert out.shape == (2, 3, 3)  # C=2, H=j[0:3], W=i[1:4]
    np.testing.assert_array_equal(out[0], ds["A"].isel(j=slice(0, 3), i=slice(1, 4)).values)
    np.testing.assert_array_equal(out[1], ds["B"].isel(j=slice(0, 3), i=slice(1, 4)).values)


def test_create_image_cutouts_batch_stacks_requested_features():
    ds = xr.Dataset({
        "A": (("j", "i"), np.arange(20).reshape(4, 5).astype("float32")),
        "B": (("j", "i"), (np.arange(20).reshape(4, 5) + 100).astype("float32")),
    })
    cutout = dict(i_start=1, i_end=3, j_start=0, j_end=2)

    tensors = create_image_cutouts_batch_as_tensors_dask(ds, ["A", "B"], [cutout])

    assert len(tensors) == 1
    t = tensors[0].numpy()
    assert t.shape == (2, 3, 3)  # exactly the requested channels, nothing appended
    np.testing.assert_array_equal(t[0], ds["A"].isel(j=slice(0, 3), i=slice(1, 4)).values)
    np.testing.assert_array_equal(t[1], ds["B"].isel(j=slice(0, 3), i=slice(1, 4)).values)


# --- extract_cutout_extents_and_metadata_in_series ------------------------

def _grid(n=21, spacing_m=1000.0):
    dx = np.full((n, n), spacing_m)
    dy = np.full((n, n), spacing_m)
    XC = np.fromfunction(lambda j, i: 200.0 + i, (n, n))   # lon varies along i
    YC = np.fromfunction(lambda j, i: 100.0 + j, (n, n))   # lat varies along j
    return XC, YC, dx, dy


def test_extract_metadata_maps_index_fields():
    XC, YC, dx, dy = _grid()
    log_gradb_np = np.arange(21 * 21, dtype=float).reshape(21, 21)
    t = np.datetime64("2012-02-09T12:00:00", "ns")
    index = (10, 5)   # non-symmetric to catch j/i swaps

    cutout, meta = extract_cutout_extents_and_metadata_in_series(
        index, XC, YC, log_gradb_np, dx, dy, target_km_res=6, time_snapshot=t)

    assert cutout is not None
    assert meta["index"] == index
    assert meta["center_lat"] == YC[index]              # from YC, not XC
    assert meta["center_lon"] == XC[index]
    assert meta["log_grad_b_2_center"] == log_gradb_np[index]
    assert meta["time_snapshot"] == t


def test_extract_returns_none_off_grid_edge():
    XC, YC, dx, dy = _grid()
    t = np.datetime64("2012-02-09T12:00:00", "ns")
    # i = 0 forces the cutout off the grid edge -> get_lat_lon_extents returns None
    cutout, meta = extract_cutout_extents_and_metadata_in_series(
        (10, 0), XC, YC, np.zeros((21, 21)), dx, dy, target_km_res=6, time_snapshot=t)
    assert cutout is None and meta is None


# --- downsample_and_write_cutout_lazy -------------------------------------

class _FakeZarr:
    def __init__(self):
        self.appended = []

    def append_image(self, img):
        self.appended.append(img)
        return b"imgid-abc"


class _FakeMeta:
    def __init__(self):
        self.records = []

    def add(self, meta):
        self.records.append(meta)


def _write_inputs():
    img = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8)
    cutout = {"real_km_w": 12.5, "real_km_h": 34.5}
    cutout_data = {"index": (3, 7), "center_lat": 1.5, "center_lon": 2.5,
                   "log_grad_b_2_center": 9.0, "time_snapshot": np.datetime64("2012-01-01", "ns")}
    return img, cutout, cutout_data


def test_downsample_write_metadata_mapping():
    zarr_ds, mw = _FakeZarr(), _FakeMeta()
    img, cutout, cutout_data = _write_inputs()

    _, meta = downsample_and_write_cutout_lazy(
        zarr_ds, mw, cutout_data, img, cutout, 4, 150, metadata_cols, ["Theta", "XC"]).compute()

    assert meta["native_grid"] == "LLC4320"
    assert meta["center_grid_j"] == 3 and meta["center_grid_i"] == 7
    assert meta["target_km_res"] == 150
    assert meta["center_lat"] == 1.5 and meta["center_lon"] == 2.5
    assert meta["log_grad_b_2_center"] == 9.0
    assert meta["time_snapshot"] == np.datetime64("2012-01-01", "ns")
    assert meta["real_km_w"] == 12.5 and meta["real_km_h"] == 34.5
    assert tuple(meta["pre_interp_res"]) == (8, 8)
    assert meta["image_id"] == b"imgid-abc"


def test_downsample_write_downsamples_and_appends():
    zarr_ds, mw = _FakeZarr(), _FakeMeta()
    img, cutout, cutout_data = _write_inputs()

    downsample_and_write_cutout_lazy(
        zarr_ds, mw, cutout_data, img, cutout, 4, 150, metadata_cols, ["Theta", "XC"]).compute()

    assert len(zarr_ds.appended) == 1
    assert tuple(zarr_ds.appended[0].shape) == (2, 4, 4)   # (C, down_sample_res, down_sample_res)
    assert len(mw.records) == 1
    assert mw.records[0]["image_id"] == b"imgid-abc"
