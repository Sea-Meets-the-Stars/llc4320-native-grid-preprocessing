"""
Surface snapshot processor for the SURF and OSN pipelines.

Processes one time snapshot of 2D surface fields and returns a
``(C, H, W)`` numpy array ready for zarr writing.

Pipeline flow
-------------
    compute_fields_fn(ds_merge, grid, channels)
    → interp staggered → tracer
    → face → latlon stitch + land mask
    → return (C, H, W) array
"""

import logging

import numpy as np

from dbof.utils.faces_to_latlon import (
    interp_staggered_to_tracer,
    set_vector_pair_attrs,
    stitch_and_mask,
)


def process_snapshot(
    ds,
    ds_merge,
    grid,
    model_feature_channels: list[str],
    computed_feature_channels: list[str],
    compute_fields_fn,
) -> np.ndarray:
    """
    Process one surface-level time snapshot.

    Parameters
    ----------
    ds : xr.Dataset
        Raw LLC4320 dataset (face topology preserved) — used as the base
        for face-to-latlon stitching.
    ds_merge : xr.Dataset
        Merged dataset (raw + grid) on the native face grid.
    grid : xgcm.Grid
        xgcm Grid with LLC face connections.
    model_feature_channels : list[str]
        Raw model fields to include (e.g. ``['Theta', 'Salt']``).
    computed_feature_channels : list[str]
        Derived fields to compute (e.g. ``['relative_vorticity']``).
    compute_fields_fn : callable
        ``(ds_merge, grid, computed_feature_channels) -> dict``
        Subset-specific computation callback.

    Returns
    -------
    np.ndarray
        Shape ``(C, H, W)`` where C = len(model + computed channels),
        H = 12960, W = 17280.  Land pixels are NaN.
    """
    # 1. Compute derived fields.
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # 2. Interpolate staggered-grid variables (U, V, oceTAUX, oceTAUY)
    #    to tracer points so all channels share the same (face, j, i) grid.
    interp_staggered_to_tracer(ds_merge, grid)

    # 3. Assemble all channels into a single Dataset for one conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels
    update_vars = (
        {ch: ds_merge[ch] for ch in model_feature_channels}
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]
    set_vector_pair_attrs(ds_to_convert)

    # 4. Land mask only (ice mask removed — applied post-hoc if needed).
    mask_dict = {"_land_mask": (ds_merge.hFacC == 0)}

    # 5. Face → latlon stitch + mask → (C, H, W).
    data = stitch_and_mask(ds_to_convert, channels_to_convert, mask_dict)

    logging.info("Surface snapshot assembly complete")
    return data
