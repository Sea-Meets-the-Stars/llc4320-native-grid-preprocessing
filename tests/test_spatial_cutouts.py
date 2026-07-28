"""Unit tests for spatial_cutouts extent helpers on the stitched (j, i) grid."""
import numpy as np
import torch

from dbof.cutout_dataset_creation.spatial_cutouts import (
    extent_in_i, extent_in_j, get_lat_lon_extents_of_cutout, downsample_image,
)


def test_downsample_image_nearest_for_coord_channels():
    ch = torch.tensor([[10., -10., 10., -10.]] * 4)   # seam that averages to 0
    img = torch.stack([ch, ch], dim=0)                # (2, 4, 4)

    out = downsample_image(img, channels=["Theta", "XC"], target_dim=2)
    assert out.shape == (2, 2, 2)
    assert torch.allclose(out[0], torch.zeros(2, 2), atol=1e-5)   # Theta: area-averaged
    assert set(out[1].flatten().tolist()) <= {10.0, -10.0}        # XC: nearest, real values


def _uniform(n_j, n_i, km=1.0):
    return np.full((n_j, n_i), km * 1000.0, dtype="float32")  # meters


def test_extent_in_i_symmetric_on_uniform_grid():
    L, R, real = extent_in_i(_uniform(5, 31), j0=2, i0=15, km_x=3.0)
    assert L == R and real > 0


def test_get_extents_symmetric_centered():
    dxC = dyC = _uniform(31, 31)
    patch = get_lat_lon_extents_of_cutout((15, 15), dxC, dyC, (31, 31), km_size=6.0)
    assert patch is not None
    assert (15 - patch["i_start"]) == (patch["i_end"] - 15)
    assert (15 - patch["j_start"]) == (patch["j_end"] - 15)
    assert np.isclose(patch["real_km_w"], patch["real_km_h"])
    assert "face" not in patch


def test_get_extents_rejects_off_edge():
    dxC = dyC = _uniform(31, 31)
    assert get_lat_lon_extents_of_cutout((15, 0), dxC, dyC, (31, 31), km_size=6.0) is None


# ---------------------------------------------------------------------------
# Non-uniform spacing (every cell differs), matching real-grid behavior.
# dxC/dyC are distinct everywhere; the relevant row/col is set to a known
# ramp so the expected extents can be hand-computed.
#
# For row [1,2,3,4,5,6] km*1e3, the cell-center spacing after
# 0.5*(dx[:-1]+dx[1:]) is [1.5, 2.5, 3.5, 4.5, 5.5] km.  From i0=2 with
# km=7.0: right cumsum [3.5, 8.0, 13.5] -> R=1; left cumsum [3.5, 6.0, 7.5]
# -> L=2; real = (1.5+2.5) + 3.5 = 7.5 km.  Asymmetric (L != R).
# ---------------------------------------------------------------------------

def _distinct(n_j, n_i):
    return (np.arange(1, n_j * n_i + 1, dtype="float32").reshape(n_j, n_i)) * 1000.0


_RAMP = np.array([1000, 2000, 3000, 4000, 5000, 6000], dtype="float32")


def test_extent_in_i_nonuniform_asymmetric():
    dxC = _distinct(6, 6)
    dxC[1] = _RAMP                       # known ramp on the sampled row j0=1
    L, R, real = extent_in_i(dxC, j0=1, i0=2, km_x=7.0)
    assert (L, R) == (2, 1)             # asymmetric extents from non-uniform spacing
    assert np.isclose(real, 7.5)


def test_extent_in_j_nonuniform_asymmetric():
    dyC = _distinct(6, 6)
    dyC[:, 2] = _RAMP                    # known ramp on the sampled column i0=2
    D, U, real = extent_in_j(dyC, j0=2, i0=2, km_y=7.0)
    assert (D, U) == (2, 1)
    assert np.isclose(real, 7.5)


def test_get_extents_nonuniform_exact():
    dxC = _distinct(6, 6)
    dyC = _distinct(6, 6) + 500.0        # distinct from dxC, still all-different
    dxC[2] = _RAMP                       # row used for i-extent at j=2
    dyC[:, 2] = _RAMP                    # column used for j-extent at i=2

    patch = get_lat_lon_extents_of_cutout((2, 2), dxC, dyC, (6, 6), km_size=14.0)
    assert (patch["i_start"], patch["i_end"]) == (0, 3)
    assert (patch["j_start"], patch["j_end"]) == (0, 3)
    assert np.isclose(patch["real_km_w"], 7.5)
    assert np.isclose(patch["real_km_h"], 7.5)
