"""Unit tests for ice_mask helpers."""
import numpy as np

from dbof.preprocessing.ice_mask import generate_siarea_mask


def test_generate_siarea_mask_threshold():
    """True only where SIarea > 0; 0 and NaN are treated as open water."""
    siarea = np.array([0.0, 0.3, np.nan, 1.0], dtype=np.float32)
    mask = generate_siarea_mask(siarea)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [False, True, False, True])
