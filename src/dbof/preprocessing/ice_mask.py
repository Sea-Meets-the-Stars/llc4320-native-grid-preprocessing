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

def load_siarea_mask(
    bucket: str,
    folder: str,
    run_id: str,
    date_prefix: str,
    fs,
    dataset_name: str = "icearea.zarr",
    channel_name: str = "SIarea",
    timestep: int = 0,
) -> np.ndarray:
    """Load SIarea from the icearea zarr store and return an ice mask.

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
    timestep :
        Timestep index within the store (default: 0).

    Returns
    -------
    np.ndarray, dtype bool, shape (H, W)
        ``True`` where SIarea > 0 (ice-covered → should be masked/NaN).
        ``False`` where SIarea == 0 or SIarea is NaN (open water → keep).
    """
    reader = GlobalZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name=dataset_name,
        fs=fs,
        date_prefix=date_prefix,
    )
    siarea = reader.get_channel_snapshot(timestep, channel_name).astype(np.float32)
    mask = siarea > 0
    n_masked = int(np.sum(mask))
    n_total = mask.size
    log.info(
        "Ice mask from %s (t=%d): %d / %d points masked (%.1f%%)",
        dataset_name, timestep, n_masked, n_total, 100.0 * n_masked / n_total,
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