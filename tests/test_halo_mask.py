"""Unit tests for stitched_halo_mask."""
import numpy as np
import pytest

from dbof.preprocessing.halo_mask import stitched_halo_mask


def _spacing(shape, km=1.0):
    return np.full(shape, km * 1000.0)  # meters


def test_halo_zero_removes_only_masked_cell():
    mask = np.zeros((5, 5), bool); mask[2, 2] = True
    out = stitched_halo_mask(mask, _spacing((5, 5)), _spacing((5, 5)), halo_km=0.0)
    assert out[2, 2] == False
    assert out.sum() == 24


def test_large_halo_excludes_everything():
    mask = np.zeros((5, 5), bool); mask[2, 2] = True
    out = stitched_halo_mask(mask, _spacing((5, 5)), _spacing((5, 5)), halo_km=100.0)
    assert out.sum() == 0


def test_no_mask_keeps_all():
    mask = np.zeros((4, 4), bool)
    out = stitched_halo_mask(mask, _spacing((4, 4)), _spacing((4, 4)), halo_km=5.0)
    assert out.all()


def test_all_masked_raises():
    mask = np.ones((4, 4), bool)
    with pytest.raises(ValueError):
        stitched_halo_mask(mask, _spacing((4, 4)), _spacing((4, 4)), halo_km=5.0)
