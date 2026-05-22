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
import re

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
#  Strategy dispatch table (reference)
# ===========================================================================

# Canonical strategy suffixes and their functions.  ``apply_depth_strategies``
# builds its own suffix→function mapping at call time so the fixed-depth
# suffix can vary (e.g. ``z25m``, ``z50m``).  This dict is kept as
# documentation of the default strategy names.
DEPTH_STRATEGIES = {
    "sfc":      select_surface,
    "z25m":     select_fixed_depth,
    "mld":      select_at_mld,
    "mld_mean": select_mld_mean,
}


# ===========================================================================
#  Apply strategies (lazy — no .compute())
# ===========================================================================

def _fixed_depth_suffix(depth_m):
    """Generate the suffix for a fixed-depth channel, e.g. 'z25m', 'z50m'."""
    if depth_m == int(depth_m):
        return f"z{int(depth_m)}m"
    return f"z{depth_m}m"


# Regex matching a fixed-depth suffix like ``z25m``, ``z50m``, ``z100m``.
_FIXED_DEPTH_RE = re.compile(r"^z(\d+(?:\.\d+)?)m$")


def _detect_fixed_depth(field_base_name, requested):
    """Scan *requested* channels for a ``z{N}m`` suffix and return the depth.

    If a channel like ``N2_z50m`` is present in *requested*, this returns
    ``50.0``.  If no fixed-depth channel is found, returns ``FIXED_DEPTH_M``
    (the 25 m default).

    Parameters
    ----------
    field_base_name : str
        Base name (e.g. ``"N2"``).
    requested : set[str] or None
        Channel names from the config.

    Returns
    -------
    float
        The fixed depth in metres.
    """
    if requested is None:
        return FIXED_DEPTH_M
    prefix = field_base_name + "_"
    for ch in requested:
        if ch.startswith(prefix):
            suffix = ch[len(prefix):]
            m = _FIXED_DEPTH_RE.match(suffix)
            if m:
                return float(m.group(1))
    return FIXED_DEPTH_M


def apply_depth_strategies(field3d, field_base_name, ds_merge, mld=None,
                           requested=None, fixed_depth_m=None):
    """Apply depth strategies to a lazy 3D field → dict of lazy 2D results.

    Operates on the full lazy 3D array directly. All returned DataArrays
    stay dask-backed until the caller's final ``dask.compute()``.

    Fixed-depth auto-detection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``fixed_depth_m`` is *None* (the default), the fixed depth is
    **auto-detected from the requested channel names**.  If *requested*
    contains a channel like ``N2_z50m``, the fixed depth is set to 50 m
    and the output key is ``N2_z50m``.  If no ``z{N}m`` channel is found,
    the default of 25 m (``FIXED_DEPTH_M``) is used.

    This means the YAML config is the single source of truth for which
    depth to use — no separate config key is needed.

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
    fixed_depth_m : float or None
        Explicit override for the fixed depth.  When *None*, the depth is
        auto-detected from *requested* channel names (see above).

    Returns
    -------
    dict[str, xr.DataArray]
        Keys are ``{field_base_name}_{suffix}``.
    """
    if fixed_depth_m is None:
        fixed_depth_m = _detect_fixed_depth(field_base_name, requested)

    fd_suffix = _fixed_depth_suffix(fixed_depth_m)

    # Build the mapping from suffix → strategy function for this call.
    # The fixed-depth suffix is dynamic, so rebuild it each time.
    suffix_to_fn = {
        "sfc":      select_surface,
        fd_suffix:  select_fixed_depth,
        "mld":      select_at_mld,
        "mld_mean": select_mld_mean,
    }

    if requested is None:
        requested_suffixes = set(suffix_to_fn)
    else:
        requested_suffixes = {
            s for s in suffix_to_fn
            if f"{field_base_name}_{s}" in requested
        }

    results = {}
    for suffix in requested_suffixes:
        key = f"{field_base_name}_{suffix}"
        strategy_fn = suffix_to_fn[suffix]
        results[key] = strategy_fn(
            field3d, ds_merge, mld=mld, fixed_depth_m=fixed_depth_m,
        )

    return results
