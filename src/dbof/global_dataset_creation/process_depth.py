"""
Depth snapshot processor for the DEPTH pipeline.

Processes one time snapshot of 3D fields (reduced to 2D by the
subset compute function via depth strategies) and returns a
``(C, H, W)`` numpy array ready for zarr writing.

Pipeline flow
-------------
    compute_fields_fn(ds_merge, grid, channels)          [3D → 2D reduction]
    → extract surface slice of raw model vars
    → face → latlon stitch + land mask
    → return (C, H, W) array

Vector fields (U/V, oceTAUX/oceTAUY) are COMPUTED channels — rotated to
geographic components upstream and stitched as scalars.  See the
vector-handling policy in ``dbof.utils.faces_to_latlon``.

Key difference from ``process_surface``
---------------------------------------
Raw model variables live on a 3D grid (face, k, j, i).  Before the
face→latlon stitch they must be sliced to the surface level.
"""

import logging

import numpy as np
import xarray as xr

from dbof.global_dataset_creation.process_surface import (
    _assert_no_staggered_model_channels,
)
from dbof.utils.faces_to_latlon import stitch_and_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_surface(ds: xr.Dataset) -> xr.Dataset:
    """Return a dataset with all k-dependent variables sliced to the surface."""
    out = {}
    for name, da in ds.data_vars.items():
        if "k" in da.dims:
            out[name] = da.isel(k=0)
        elif "k_l" in da.dims:
            out[name] = da.isel(k_l=0)
        else:
            out[name] = da
    return xr.Dataset(out, coords=ds.coords, attrs=ds.attrs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_snapshot(
    ds,
    ds_merge,
    grid,
    model_feature_channels: list[str],
    computed_feature_channels: list[str],
    compute_fields_fn,
    surface_only: bool,
) -> np.ndarray:
    """
    Process one depth-pipeline time snapshot.

    Parameters
    ----------
    ds : xr.Dataset
        Raw LLC4320 dataset from S3 (face topology preserved).
    ds_merge : xr.Dataset
        Merged dataset (raw + grid) on the native face grid.
    grid : xgcm.Grid
        xgcm Grid with LLC face connections.
    model_feature_channels : list[str]
        Raw model fields to include (tracer-point scalars only,
        e.g. ``['oceQnet']``).
    computed_feature_channels : list[str]
        Derived fields (already depth-expanded, e.g. ``['N2_sfc', 'N2_mld']``).
    compute_fields_fn : callable
        ``(ds_merge, grid, computed_feature_channels) -> dict``
        Subset-specific computation callback (from ``depth_subsets.py``).
    surface_only : bool
        When ``True``, the raw data was already sliced to the surface before
        ``process_llc4320``.  The surface-extraction step here is still
        applied to handle any remaining k-dims (e.g. from grid variables).

    Returns
    -------
    np.ndarray
        Shape ``(C, H, W)`` where C = len(model + computed channels),
        H = 12960, W = 17280.  Land pixels are NaN.
    """
    # 1. Compute derived fields (3D → 2D reduction happens inside).
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # 2. Extract surface slice from any 3D raw model variables.
    available = [ch for ch in model_feature_channels if ch in ds_merge]
    ds_model_subset = _select_surface(ds_merge[available])
    surface_model_vars = {ch: ds_model_subset[ch] for ch in available}
    _assert_no_staggered_model_channels(surface_model_vars)

    # 3. Assemble all channels into a single Dataset for one conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels

    # Build a surface-level base dataset for face→latlon conversion.
    ds_surface = ds
    for dim_name in ("k", "k_l"):
        if dim_name in ds_surface.dims:
            ds_surface = ds_surface.isel({dim_name: 0})

    update_vars = (
        {ch: surface_model_vars[ch] for ch in model_feature_channels if ch in surface_model_vars}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels if ch in calculated_fields}
    )
    ds_to_convert = ds_surface.assign(update_vars)[channels_to_convert]

    # 4. Build surface land mask from hFacC.
    hfac = ds_merge.hFacC if "k" not in ds_merge.hFacC.dims else ds_merge.hFacC.isel(k=0)
    mask_dict = {"_land_mask": (hfac == 0)}

    # 5. Face → latlon stitch + mask → (C, H, W).
    data = stitch_and_mask(ds_to_convert, channels_to_convert, mask_dict,
                           progress_bar=True)

    logging.info("Depth snapshot assembly complete")
    return data
