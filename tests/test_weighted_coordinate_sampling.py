"""Unit tests for weighted_sample_on_grid on the stitched (j, i) grid."""
import numpy as np
import xarray as xr

from dbof.preprocessing.weighted_coordinate_sampling import weighted_sample_on_grid


def _field(j=5, i=6):
    """Increasing (j, i) field with no coords — mirrors ds_merge feature layout."""
    arr = np.arange(1, j * i + 1, dtype="float32").reshape(j, i)
    return xr.DataArray(arr, dims=("j", "i")), arr


def test_returns_positional_ji_indices_respecting_mask():
    np.random.seed(0)
    da, arr = _field()
    mask = np.ones(arr.shape, dtype=bool)
    mask[0, :] = False  # exclude first row

    idx = weighted_sample_on_grid(5, 1.3, da, mask)

    assert len(idx) == 5
    assert len(set(idx)) == 5            # sampled without replacement
    for j, i in idx:
        assert 0 <= j < arr.shape[0] and 0 <= i < arr.shape[1]
        assert mask[j, i]                # never a masked cell


def test_recovers_exact_cells_when_all_but_k_masked():
    """Deterministic check of (j, i) index recovery on the stitched grid."""
    da, arr = _field()
    mask = np.zeros(arr.shape, dtype=bool)
    keep = {(1, 2), (3, 4), (4, 0)}
    for j, i in keep:
        mask[j, i] = True
    idx = weighted_sample_on_grid(3, 1.3, da, mask)
    assert set(idx) == keep
