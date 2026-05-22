"""
LLC4320 date ↔ iteration-number conversions and related constants.

Every pipeline needs to map human-readable dates to LLC4320 iteration numbers.
This module centralises the conversion logic and the calendar constants that
drive it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# LLC4320 calendar / model constants
# ---------------------------------------------------------------------------
TS_PER_HOUR              = 144          # 25 s timestep → 144 steps/hr
MAX_ITER                 = 1_495_008
FIRST_WIND_RECORD_OFFSET = 10_368       # OSN iteration-numbering shift

LLC_FACES                = range(13)
LLC4320_START_DATE       = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS    = 25           # seconds per model step
DATE_FMT                 = '%Y-%m-%d %H:%M:%S'


# ---------------------------------------------------------------------------
# Date → iteration converters
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


def date_to_run_id(date_str: str) -> str:
    """
    Convert a date string to a directory-safe run-id.

    ``'2011-12-09 12:00:00'`` → ``'20111209_120000'``
    """
    dt = datetime.strptime(date_str.strip(), DATE_FMT)
    return dt.strftime("%Y%m%d_%H%M%S")


def osn_date_to_iteration(date_str: str) -> int:
    """
    Convert a date string to an **OSN** iteration number.

    Same as :func:`mit_date_to_iteration` but adds
    ``FIRST_WIND_RECORD_OFFSET`` (10 368) to align with the OSN data store's
    iteration numbering, which is shifted relative to the MIT model epoch
    (i.e. the effective OSN start date is 2011-09-10 00:00:00 UTC).
    """
    return mit_date_to_iteration(date_str) + FIRST_WIND_RECORD_OFFSET


# ---------------------------------------------------------------------------
# Iteration-list builder (used by surface / OSN pipelines)
# ---------------------------------------------------------------------------

def calculate_iterations_for_llc(
    cfg,
    *,
    use_osn_offset: bool = True,
) -> np.ndarray:
    """
    Return the array of LLC4320 iteration numbers to process.

    ``cfg.data.date_iterations`` must be a list of date strings.
    Each is converted via the appropriate date-to-iteration converter.

    Parameters
    ----------
    cfg : GlobalJobConfig or JobConfig
        Pipeline configuration.  Must have ``cfg.data.date_iterations``.
    use_osn_offset : bool, default True
        If True, use :func:`osn_date_to_iteration` (adds OSN offset);
        otherwise use :func:`mit_date_to_iteration`.
    """
    date_to_iter = osn_date_to_iteration if use_osn_offset else mit_date_to_iteration

    if cfg.data.date_iterations is None or len(cfg.data.date_iterations) == 0:
        raise ValueError(
            "cfg.data.date_iterations must be set.  The range-mode fallback "
            "(sampling_step / start_record / timestep_hours) has been removed."
        )

    iterations = [date_to_iter(d) for d in cfg.data.date_iterations]
    label = "OSN offset applied" if use_osn_offset else "MIT iterations"
    logging.info(
        f"Using date-derived iteration list ({label}): "
        + ", ".join(
            f"'{d}' → {it}"
            for d, it in zip(cfg.data.date_iterations, iterations)
        )
    )
    return np.array(iterations, dtype=int)
