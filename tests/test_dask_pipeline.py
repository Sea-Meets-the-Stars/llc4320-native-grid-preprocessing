"""Unit tests for dask_pipeline cutout-image extraction."""
import numpy as np
import xarray as xr

from dbof.cutout_dataset_creation.dask_pipeline import (
    create_image_cutout_lazy,
    create_image_cutouts_batch_as_tensors_dask,
)


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
