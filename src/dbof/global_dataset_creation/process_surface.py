"""
Surface snapshot processor for the SURF and OSN pipelines.

Processes one time snapshot of 2D surface fields and returns a
``(C, H, W)`` numpy array ready for zarr writing.

Pipeline flow
-------------
    compute_fields_fn(ds_merge, grid, channels)
    → face → latlon stitch + land mask
    → return (C, H, W) array

Vector fields (U/V, oceTAUX/oceTAUY) are COMPUTED channels — rotated to
geographic components upstream and stitched as scalars.  See the
vector-handling policy in ``dbof.utils.faces_to_latlon``.
"""

import logging

import numpy as np

from dbof.utils.faces_to_latlon import stitch_and_mask


#: Staggered horizontal dims — model channels must never carry these.
_STAGGERED_DIMS = frozenset({"i_g", "j_g"})


def _assert_no_staggered_model_channels(model_vars: dict) -> None:
    """Refuse staggered-grid variables in the model-channel path.

    Vector fields must go through the compute-function path (interp +
    CS/SN rotation) — see the vector-handling policy in
    ``dbof.utils.faces_to_latlon``.
    """
    for name, da in model_vars.items():
        if _STAGGERED_DIMS & set(da.dims):
            raise ValueError(
                f"Model channel '{name}' is on a staggered grid "
                f"(dims {da.dims}).  Staggered vectors must be handled as "
                "computed channels (interp + CS/SN rotation) — see the "
                "vector-handling policy in dbof.utils.faces_to_latlon."
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
        Raw model fields to include (tracer-point scalars only,
        e.g. ``['Theta', 'Salt']``).
    computed_feature_channels : list[str]
        Derived fields to compute (e.g. ``['relative_vorticity', 'U', 'V']``).
    compute_fields_fn : callable
        ``(ds_merge, grid, computed_feature_channels) -> dict``
        Subset-specific computation callback.

    Returns
    -------
    np.ndarray
        Shape ``(C, H, W)`` where C = len(model + computed channels),
        H = 12960, W = 17280.  Land pixels are NaN.
    """
    # 1. Compute derived fields (includes tracer-point interpolation and
    #    geographic rotation for vector channels).
    calculated_fields = compute_fields_fn(ds_merge, grid, computed_feature_channels)

    # 2. Assemble all channels into a single Dataset for one conversion pass.
    channels_to_convert = model_feature_channels + computed_feature_channels
    model_vars = {ch: ds_merge[ch] for ch in model_feature_channels}
    _assert_no_staggered_model_channels(model_vars)
    update_vars = (
        model_vars
        | {ch: calculated_fields[ch] for ch in computed_feature_channels}
    )
    ds_to_convert = ds.assign(update_vars)[channels_to_convert]

    # 3. Land mask only (ice mask removed — applied post-hoc if needed).
    mask_dict = {"_land_mask": (ds_merge.hFacC == 0)}

    # 4. Face → latlon stitch + mask → (C, H, W).
    data = stitch_and_mask(ds_to_convert, channels_to_convert, mask_dict)

    logging.info("Surface snapshot assembly complete")
    return data
