"""
check_existence.py
------------------
Existence / completeness checks for global pipeline outputs, shared by the
CLI drivers (``generate_global``, ``run_all_subsets``).

Two kinds of products are checked:

1. **Zarr stores** (one per subset x date on S3), written by
   ``GlobalZarrDataset``.  A store is *complete* when its root metadata
   exists, its ``channel_names`` attribute covers every expected channel,
   and at least one timestep has been written.  All checks read only the
   small ``zarr.json`` metadata objects -- no chunk data is downloaded.

2. **NetCDF exports** (one file per channel x date on local disk), written
   by ``zarr_to_netcdf``::

       {output_dir}/LLC4320_{date}_{channel}_{run_id}.nc

The planning helper :func:`plan_subset_date` implements the ".nc-first"
ordering used by ``run_all_subsets``:

    check .nc files
        -> all exist:            SKIP (zarr store is not even consulted)
        -> some missing:         check zarr store completeness
               -> complete:      EXPORT only the missing channels
               -> incomplete:    GENERATE the store, then export
"""

import json
import logging
import os

from dbof.global_dataset_creation.iterations import prefix_to_filename_date

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan actions
# ---------------------------------------------------------------------------

SKIP = "skip"          #: nothing to do for this subset/date
EXPORT = "export"      #: zarr store is complete; export missing channels
GENERATE = "generate"  #: zarr store missing/incomplete; regenerate it


# ---------------------------------------------------------------------------
# Zarr store checks (S3, metadata-only)
# ---------------------------------------------------------------------------

def _store_key(store_path: str) -> str:
    """Normalise an ``s3://...`` store path to a bare S3 key prefix."""
    return store_path.removeprefix("s3://").rstrip("/")


def store_exists(fs, store_path: str) -> bool:
    """Check that the store's root metadata object exists.

    Tests the concrete ``{store}/zarr.json`` object rather than the bare
    prefix: ``fs.exists()`` on a pure S3 prefix is a listing operation that
    is unreliable (directory-cache dependent) and returns ``True`` for any
    stray key under the prefix (e.g. debris from a crashed run).

    Parameters
    ----------
    fs : fsspec filesystem (sync)
        S3 filesystem.
    store_path : str
        Full store path (``s3://bucket/.../dataset.zarr``).

    Returns
    -------
    bool
    """
    key = _store_key(store_path)
    fs.invalidate_cache(key)
    return fs.exists(f"{key}/zarr.json")


def _read_json(fs, key: str) -> dict | None:
    """Read a JSON object from S3, returning ``None`` if it is missing."""
    try:
        with fs.open(key, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def store_channels(fs, store_path: str) -> list[str] | None:
    """Return the store's ``channel_names`` attribute.

    Reads only the root ``zarr.json`` (zarr v3 keeps group attributes
    there), so this is a single small GET.

    Returns
    -------
    list[str] or None
        Channel names, or ``None`` if the store / attribute is missing.
    """
    key = _store_key(store_path)
    fs.invalidate_cache(key)
    meta = _read_json(fs, f"{key}/zarr.json")
    if meta is None:
        return None
    channels = meta.get("attributes", {}).get("channel_names")
    return list(channels) if channels is not None else None


def store_n_timesteps(fs, store_path: str) -> int:
    """Return the number of timesteps written to the store's ``data`` array.

    Reads ``{store}/data/zarr.json`` and returns ``shape[0]``.  A freshly
    created but never-written store (e.g. from a crashed run) has shape
    ``(0, C, H, W)`` and therefore returns 0.

    Returns
    -------
    int
        Number of timesteps; 0 if the array metadata is missing.
    """
    key = _store_key(store_path)
    meta = _read_json(fs, f"{key}/data/zarr.json")
    if meta is None:
        return 0
    shape = meta.get("shape") or [0]
    return int(shape[0])


def store_is_complete(
    fs,
    store_path: str,
    expected_channels: list[str],
    min_timesteps: int = 1,
) -> bool:
    """Check that a zarr store exists AND contains the expected channels/data.

    A store is complete when:

    1. the root metadata object exists,
    2. its ``channel_names`` attribute is a superset of *expected_channels*
       (catches stores generated with a different depth-suffix set), and
    3. its ``data`` array holds at least *min_timesteps* timesteps
       (catches stores created but never written by a crashed run).

    All three checks are metadata-only (small JSON GETs).

    Parameters
    ----------
    fs : fsspec filesystem (sync)
        S3 filesystem.
    store_path : str
        Full store path (``s3://...``).
    expected_channels : list[str]
        Fully suffix-expanded channel names the store must contain.
    min_timesteps : int
        Minimum number of written timesteps.  Per-date stores hold exactly
        one snapshot, so the default of 1 is correct for the global layout.

    Returns
    -------
    bool
    """
    channels = store_channels(fs, store_path)
    if channels is None:
        log.debug("Store missing or has no channel_names: %s", store_path)
        return False

    missing = set(expected_channels) - set(channels)
    if missing:
        log.info("Store %s is missing channel(s): %s", store_path,
                 sorted(missing))
        return False

    n_t = store_n_timesteps(fs, store_path)
    if n_t < min_timesteps:
        log.info("Store %s has %d timestep(s) (< %d) -- treated as "
                 "incomplete.", store_path, n_t, min_timesteps)
        return False

    return True


# ---------------------------------------------------------------------------
# NetCDF export checks (local disk)
# ---------------------------------------------------------------------------

def netcdf_filename(date_prefix: str, channel: str, run_id: str) -> str:
    """Return the export filename for one channel x date.

    ``LLC4320_{date}_{channel}_{run_id}.nc`` with
    ``date = prefix_to_filename_date(date_prefix)``.
    """
    return f"LLC4320_{prefix_to_filename_date(date_prefix)}_{channel}_{run_id}.nc"


def missing_netcdfs(
    output_dir: str,
    date_prefix: str,
    run_id: str,
    channels: list[str],
) -> list[str]:
    """Return the channels whose exported ``.nc`` file does not exist.

    Parameters
    ----------
    output_dir : str
        Directory holding the exports for this date
        (``{netcdf_base}/{run_id}/{date_prefix}``).
    date_prefix : str
        Date prefix string (``YYYYMMDD_HHMMSS``).
    run_id : str
        Run identifier.
    channels : list[str]
        Fully suffix-expanded channel names to check.

    Returns
    -------
    list[str]
        Subset of *channels* with no existing ``.nc`` file (order preserved).
    """
    return [
        ch for ch in channels
        if not os.path.exists(
            os.path.join(output_dir, netcdf_filename(date_prefix, ch, run_id))
        )
    ]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_subset_date(
    fs,
    store_path: str,
    output_dir: str,
    date_prefix: str,
    run_id: str,
    channels: list[str],
) -> tuple[str, list[str]]:
    """Decide what to do for one subset x date (".nc-first" ordering).

    Logic::

        all .nc files exist                -> (SKIP, [])
        some missing, zarr complete        -> (EXPORT, missing_channels)
        some missing, zarr incomplete      -> (GENERATE, missing_channels)

    Note the zarr store is only consulted when at least one ``.nc`` file is
    missing, so deleting a store whose exports are all on disk never
    triggers an (expensive) regeneration.

    Parameters
    ----------
    fs : fsspec filesystem (sync)
        S3 filesystem.
    store_path : str
        Full zarr store path for this subset x date.
    output_dir : str
        Local NetCDF directory for this date.
    date_prefix : str
        Date prefix string (``YYYYMMDD_HHMMSS``).
    run_id : str
        Run identifier.
    channels : list[str]
        Fully suffix-expanded channel names for this subset.

    Returns
    -------
    (action, channels_to_export) : tuple[str, list[str]]
        *action* is one of :data:`SKIP`, :data:`EXPORT`, :data:`GENERATE`.
        *channels_to_export* lists the channels whose ``.nc`` is missing
        (empty for SKIP).
    """
    missing = missing_netcdfs(output_dir, date_prefix, run_id, channels)
    if not missing:
        return SKIP, []

    if store_is_complete(fs, store_path, channels):
        return EXPORT, missing

    return GENERATE, missing
