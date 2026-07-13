"""
ice_mask.py
-----------
Helpers for masking sea-ice-covered points in LLC4320 global outputs.

Two approaches are provided:

1. ``mask_by_theta`` — legacy mask derived from potential temperature
   (Theta <= threshold → ice).  Used by the surface pipeline.

2. ``load_siarea_mask`` / ``apply_ice_mask`` — mask derived from the
   SIarea (sea-ice area fraction) field stored in ``icearea.zarr``.
   Any point with SIarea > 0 is considered ice-covered and set to NaN.
"""

import logging

import numpy as np

from dbof.global_dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader
import dbof.preprocessing.halo_mask as halo_mask

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy theta-based mask (used by generate_global.py surface pipeline)
# ---------------------------------------------------------------------------

def mask_by_theta(ds, theta_threshold: float = 0.0):
    """Return a boolean mask where True = open water (Theta > threshold).

    Args:
        ds: xarray.Dataset containing a ``Theta`` variable.
        theta_threshold: float

    Returns:
        xarray.DataArray — boolean, True where Theta > threshold.
    """
    ice_mask = ~(ds.Theta <= theta_threshold)
    return ice_mask


# ---------------------------------------------------------------------------
# SIarea-based mask (used by depth pipeline exports)
# ---------------------------------------------------------------------------

def load_siarea(
    bucket: str,
    folder: str,
    run_id: str,
    date_prefix: str,
    fs,
    dataset_name: str = "icearea.zarr",
    channel_name: str = "SIarea",
) -> np.ndarray:
    """Load the raw SIarea (sea-ice area fraction) field from the icearea store.

    Parameters
    ----------
    bucket, folder, run_id, date_prefix :
        S3 coordinates — identical to those used for the variable being
        exported, so the SIarea snapshot matches in time.
    fs :
        An s3fs filesystem instance (synchronous).
    dataset_name :
        Zarr store name for the sea-ice data (default: ``icearea.zarr``).
    channel_name :
        Channel within the store (default: ``SIarea``).

    Returns
    -------
    np.ndarray, dtype float32, shape (H, W)
    """
    reader = GlobalZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name=dataset_name,
        fs=fs,
        date_prefix=date_prefix,
    )
    return reader.get_channel_snapshot(channel_name).astype(np.float32)


def generate_siarea_mask(siarea: np.ndarray) -> np.ndarray:
    """Boolean ice mask from a SIarea field.

    Returns
    -------
    np.ndarray, dtype bool, shape (H, W)
        ``True`` where SIarea > 0 (ice-covered → should be masked/NaN).
        ``False`` where SIarea == 0 or SIarea is NaN (open water → keep).
    """
    return siarea > 0


def generate_halo_ice_mask(ds_grid, ice_mask, target_km_res, DXC=None, DYC=None, stitched=True):
    """Buffer a sea-ice mask with a halo, mirroring
    ``static_masks.generate_halo_land_mask``.

    Unlike land (derived from the static grid's ``hFacC``), the ice mask is
    per-snapshot, so the boolean ice mask (``True`` = ice-covered) is passed in.
    A halo of *target_km_res* is applied; returns ``True`` = retained
    (i.e. >= halo_km from any ice cell).

    Parameters
    ----------
    ds_grid : xarray.Dataset
        Source of ``dxC``/``dyC`` grid spacing.
    ice_mask : np.ndarray of bool
        ``True`` = ice-covered.
    target_km_res : int
    DXC, DYC : xarray.DataArray, optional
        Grid spacing overrides (meters); default reads them from *ds_grid*.
    stitched : bool
        True if the grid is the stitched (j, i) array rather than native faces.
    """
    halo_km = target_km_res  # buffer to account for mean usage

    DXC = ds_grid["dxC"].persist() if DXC is None else DXC
    DYC = ds_grid["dyC"].persist() if DYC is None else DYC

    if stitched:
        return halo_mask.stitched_halo_mask(mask=ice_mask, dxC=DXC, dyC=DYC, halo_km=halo_km)
    return halo_mask.llc_native_grid_halo_mask(mask=ice_mask, dxC=DXC, dyC=DYC, halo_km=halo_km)


def load_siarea_mask(
    bucket: str,
    folder: str,
    run_id: str,
    date_prefix: str,
    fs,
    dataset_name: str = "icearea.zarr",
    channel_name: str = "SIarea",
) -> np.ndarray:
    """Load SIarea from the icearea zarr store and return an ice mask.

    Thin wrapper composing :func:`load_siarea` and
    :func:`generate_siarea_mask`.

    Returns
    -------
    np.ndarray, dtype bool, shape (H, W)
        ``True`` where SIarea > 0 (ice-covered → should be masked/NaN).
        ``False`` where SIarea == 0 or SIarea is NaN (open water → keep).
    """
    siarea = load_siarea(
        bucket, folder, run_id, date_prefix, fs,
        dataset_name=dataset_name, channel_name=channel_name,
    )
    mask = generate_siarea_mask(siarea)
    n_masked = int(np.sum(mask))
    n_total = mask.size
    log.info(
        "Ice mask from %s: %d / %d points masked (%.1f%%)",
        dataset_name, n_masked, n_total, 100.0 * n_masked / n_total,
    )
    return mask


def apply_ice_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set ice-covered points to NaN.

    Parameters
    ----------
    arr : np.ndarray, float32, shape (H, W)
        The variable data to mask.
    mask : np.ndarray, bool, shape (H, W)
        ``True`` at points to mask (set to NaN).

    Returns
    -------
    np.ndarray, float32, shape (H, W)
        Copy of *arr* with ice-covered points set to NaN.
    """
    return np.where(mask, np.nan, arr)