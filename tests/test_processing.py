"""Unit tests for processing helpers."""
import numpy as np
import xarray as xr

from dbof.cutout_dataset_creation.processing import sample_cutout_centers_with_loggradb


def test_sample_cutout_centers_uses_log10_gradb2_and_mask():
    arr = np.arange(1, 4 * 5 + 1, dtype="float32").reshape(4, 5)
    ds = xr.Dataset({"gradb2": (("j", "i"), arr)})
    mask = np.ones((4, 5), dtype=bool)
    mask[0, :] = False

    np.random.seed(0)
    indices, log_gradb_np = sample_cutout_centers_with_loggradb(ds, mask, 5, 1.3)

    assert np.allclose(log_gradb_np, np.log10(arr))     # samples on log10(gradb2)
    assert len(indices) == 5 and len(set(indices)) == 5
    for j, i in indices:
        assert mask[j, i]
