"""
LLC4320 date <-> iteration-number conversions and related constants.

Every pipeline needs to map human-readable dates to LLC4320 iteration numbers.
This module centralises the conversion logic and the calendar constants that
drive it.  It lives in ``llc4320_ingestion`` (the ingestion layer) rather than
inside any single pipeline because the LLC4320 calendar is a property of the
*dataset*, not of any one pipeline: the global, transfer, and any future
pipelines all depend on it.

(Historically this lived at ``dbof.global_dataset_creation.iterations``; that
import path is preserved as a thin re-export shim for backwards compatibility.)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# LLC4320 calendar / model constants
# ---------------------------------------------------------------------------
TS_PER_HOUR              = 144          # 25 s timestep -> 144 steps/hr
MAX_ITER                 = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368       # OSN iteration-numbering shift

LLC_FACES                = range(13)
LLC4320_START_DATE       = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS    = 25           # seconds per model step
DATE_FMT                 = '%Y-%m-%d %H:%M:%S'


# ---------------------------------------------------------------------------
# Date -> iteration converters
# ---------------------------------------------------------------------------

def mit_date_to_iteration(date_str: str) -> int:
    """
    Convert a date string to a *raw* LLC4320 (MIT) iteration number.

    The LLC4320 model starts at 2011-09-13 00:00:00 UTC (iteration 0) with a
    25-second timestep.  The returned iteration is rounded to the nearest step.

    This variant does **not** apply the OSN offset and should be used by any
    pipeline that reads directly from S3 zarr stores keyed by MIT iterations
    (e.g. the depth pipeline).

    Examples
    --------
    >>> mit_date_to_iteration('2011-09-13 00:00:00')
    0
    >>> mit_date_to_iteration('2012-01-01 00:00:00')  # ~1,011,456
    1011456
    """
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    if delta.total_seconds() < 0:
        raise ValueError(
            f"Date '{date_str}' is before LLC4320 start ({LLC4320_START_DATE.date()}). "
            f"Expected format: YYYY-MM-DD HH:MM:SS  (e.g. '2011-09-13 00:00:00')."
        )
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def osn_date_to_iteration(date_str: str) -> int:
    """
    Convert a date string to an **OSN** iteration number.

    Same as :func:`mit_date_to_iteration` but adds
    ``FIRST_WIND_RECORD_OFFSET`` (10 368) to align with the OSN data store's
    iteration numbering, which is shifted relative to the MIT model epoch
    (i.e. the effective OSN start date is 2011-09-10 00:00:00 UTC).
    """
    return mit_date_to_iteration(date_str) + FIRST_WIND_RECORD_OFFSET


def mit_date_to_time_idx(date_str: str, ntime: int) -> int:
    """
    Resolve a date string to the hourly time index into an MIT LLC4320 store.

    Stores keyed by raw MIT iterations are written at hourly cadence, so the
    time index is the MIT iteration divided by ``TS_PER_HOUR``.  *ntime* is the
    length of the store's ``time`` dimension; the result is validated to fall
    within ``[0, ntime)``.  (Named for the MIT iteration convention, mirroring
    :func:`mit_date_to_iteration`, since OSN-numbered stores would index
    differently.)
    """
    iteration = mit_date_to_iteration(date_str)
    time_idx = iteration // TS_PER_HOUR
    if not (0 <= time_idx < ntime):
        raise ValueError(f"time_idx={time_idx} out of range [0, {ntime})")
    return time_idx


# ---------------------------------------------------------------------------
# Prefix <-> display / filename converters
# ---------------------------------------------------------------------------

def date_to_run_id(date_str: str) -> str:
    """
    Convert a date string to a directory-safe run-id.

    ``'2011-12-09 12:00:00'`` -> ``'20111209_120000'``
    """
    dt = datetime.strptime(date_str.strip(), DATE_FMT)
    return dt.strftime("%Y%m%d_%H%M%S")


def prefix_to_display(date_prefix: str) -> str:
    """
    Convert a date prefix back to a human-readable date string.

    ``'20121109_120000'`` -> ``'2012-11-09 12:00:00'``
    """
    try:
        dt = datetime.strptime(date_prefix, "%Y%m%d_%H%M%S")
        return dt.strftime(DATE_FMT)
    except ValueError:
        return date_prefix


def prefix_to_filename_date(date_prefix: str) -> str:
    """
    Convert a date prefix to the filename-safe date format used by NetCDF exports.

    ``'20121109_120000'`` -> ``'2012-11-09T12_00_00'``
    """
    dt = datetime.strptime(date_prefix, "%Y%m%d_%H%M%S")
    return dt.strftime("%Y-%m-%dT%H_%M_%S")
