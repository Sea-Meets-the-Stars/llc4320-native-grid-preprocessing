"""
Depth-selection strategies for reducing 3D fields to 2D.

Four strategies map a 3D field to a 2D output:

- ``surface``      : z = 0  (k = 0)
- ``fixed_depth``  : nearest model level to 25 m
- ``at_mld``       : nearest model level to the mixed-layer depth
- ``mld_mean``     : thickness-weighted mean over 0 ≤ z ≤ MLD

All strategies operate lazily on dask-backed arrays — no ``.compute()``
calls.  The caller's final ``dask.compute()`` materialises everything
at once.
"""

import logging

import numpy as np

from dbof.preprocessing.vertical_helpers import (
    _get_vertical_dim,
    _get_depth_coord,
    _nearest_k_to_depth,
    _select_at_depth,
    _extract_at_mld,
    _masked_ml_mean,
)


# ===========================================================================
#  Constants
# ===========================================================================

FIXED_DEPTH_M = 25.0   # default fixed depth (≈ k=3 for LLC4320)


# ===========================================================================
#  Individual strategy functions
# ===========================================================================

def select_surface(field3d, ds_merge, mld=None, fixed_depth_m=None):
    """Select z = 0 (surface).  Uses k=0 / k_l=0."""
    zdim = _get_vertical_dim(field3d)
    return field3d.isel({zdim: 0})


def select_fixed_depth(field3d, ds_merge, mld=None,
                       fixed_depth_m=FIXED_DEPTH_M):
    """Select the nearest model level to a fixed depth (default 25 m)."""
    return _select_at_depth(field3d, fixed_depth_m, ds_merge)


def select_at_mld(field3d, ds_merge, mld=None, fixed_depth_m=None):
    """Select the nearest model level to the mixed-layer depth."""
    if mld is None:
        raise ValueError("select_at_mld requires mld to be precomputed.")
    return _extract_at_mld(field3d, mld, ds_merge)


def select_mld_mean(field3d, ds_merge, mld=None, fixed_depth_m=None):
    """Thickness-weighted mean over 0 ≤ z ≤ MLD."""
    if mld is None:
        raise ValueError("select_mld_mean requires mld to be precomputed.")
    return _masked_ml_mean(field3d, mld, ds_merge)


# ===========================================================================
#  Strategy dispatch table
# ===========================================================================

# Ordered mapping: suffix → strategy function.
DEPTH_STRATEGIES = {
    "sfc":      select_surface,
    "z25m":     select_fixed_depth,
    "mld":      select_at_mld,
    "mld_mean": select_mld_mean,
}


# ===========================================================================
#  Apply strategies (lazy — no .compute())
# ===========================================================================

def apply_depth_strategies(field3d, field_base_name, ds_merge, mld=None,
                           requested=None):
    """Apply depth strategies to a lazy 3D field → dict of lazy 2D results.

    Operates on the full lazy 3D array directly — no per-k loops or
    ``field_at_k`` closures.  All returned DataArrays stay dask-backed
    until the caller's final ``dask.compute()``.

    Parameters
    ----------
    field3d : xr.DataArray
        Lazy 3D field to reduce.
    field_base_name : str
        Base name for the output keys (e.g. ``"N2"``).
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates.
    mld : xr.DataArray or None
        Pre-computed mixed-layer depth (needed by mld/mld_mean strategies).
    requested : set[str] or None
        If given, only channels in this set are computed.  If *None*, all
        four channels are computed.

    Returns
    -------
    dict[str, xr.DataArray]
        Keys are ``{field_base_name}_{suffix}``.
    """
    if requested is None:
        requested_suffixes = set(DEPTH_STRATEGIES)
    else:
        requested_suffixes = {
            s for s in DEPTH_STRATEGIES
            if f"{field_base_name}_{s}" in requested
        }

    zdim = _get_vertical_dim(field3d)
    results = {}

    for suffix in requested_suffixes:
        key = f"{field_base_name}_{suffix}"
        if suffix == "sfc":
            results[key] = field3d.isel({zdim: 0})
        elif suffix == "z25m":
            z = _get_depth_coord(ds_merge, zdim=zdim)
            k25 = _nearest_k_to_depth(z.values.astype(np.float64),
                                      FIXED_DEPTH_M)
            results[key] = field3d.isel({zdim: k25})
        elif suffix == "mld":
            results[key] = _extract_at_mld(field3d, mld, ds_merge)
        elif suffix == "mld_mean":
            results[key] = _masked_ml_mean(field3d, mld, ds_merge)

    return results
