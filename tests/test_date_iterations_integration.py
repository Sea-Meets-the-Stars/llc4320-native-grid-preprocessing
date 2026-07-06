"""
Unit + integration tests for LLC4320 date <-> iteration conversions
(``dbof.llc4320_ingestion.date_iterations``).

Addresses PR review on ``mit_date_to_iteration`` / ``osn_date_to_iteration``:
prove that a config date resolves to the *same physical timestamp* in every
store.  The proof chain, for each date in
``configs/transfer/run_surface.yaml``:

1. **MIT**  ``ds_mit.time[mit_date_to_time_idx(date)] == date``
   (runs only where /orcd is mounted; auto-skipped elsewhere)
2. **S3**   the transferred store ``dbof/LLC4320_RAW/SURFACE/{stamp}.zarr``
   carries ``time == date`` (needs NRP S3 credentials)
3. **OSN**  the kerchunk store at ``osn_date_to_iteration(date)`` decodes to
   ``time == date`` (anonymous, network only)

If (1) and (3) both pass for the same date, a valid MIT index and a valid OSN
index are guaranteed to refer to the same timestamp — the reviewer's concern.
Test (4) additionally compares OSN vs. our S3 store *directly*, which is the
strict version of the inline check in ``generate_global``
(``verify_osn_llc_surf_timestamp`` logs-and-continues on infrastructure
errors; the test version hard-asserts).

Running
-------
Plain ``pytest`` runs only the offline unit tests — integration tests are
deselected by ``addopts`` in ``pyproject.toml``.  Select them explicitly:

    pytest tests/test_date_iterations_integration.py -m mit      # on MIT machine
    pytest tests/test_date_iterations_integration.py -m s3_dbof  # needs NRP creds
    pytest tests/test_date_iterations_integration.py -m osn      # network only
    pytest tests/test_date_iterations_integration.py -m integration  # all

If the dbof S3 bucket is down, run only the MIT + OSN checks:

    pytest tests/test_date_iterations_integration.py -m "(mit or osn) and not s3_dbof"
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from dbof.llc4320_ingestion.date_iterations import (
    DATE_FMT,
    FIRST_WIND_RECORD_OFFSET,
    TS_PER_HOUR,
    mit_date_to_iteration,
    mit_date_to_time_idx,
    osn_date_to_iteration,
)

# ---------------------------------------------------------------------------
# Config-driven constants (single source of truth: the transfer yaml)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
SURFACE_CONFIG = REPO / "configs" / "transfer" / "run_surface.yaml"

with open(SURFACE_CONFIG) as _f:
    _CFG = yaml.safe_load(_f)

MIT_PATH = _CFG["data"]["MIT_data_path"]
DATES = list(dict.fromkeys(_CFG["data"]["date_iterations"]))
S3_ENDPOINT = _CFG["output"]["s3_endpoint"]
# e.g. "dbof/" + "LLC4320_RAW" + "/SURFACE"  ->  dbof/LLC4320_RAW/SURFACE
S3_PREFIX = (
    f"{_CFG['output']['bucket'].rstrip('/')}/"
    f"{_CFG['output']['raw_prefix'].strip('/')}/"
    f"{_CFG['output']['folder'].strip('/')}"
)

S3_STORAGE_OPTIONS = {
    "anon": False,
    "client_kwargs": {"endpoint_url": S3_ENDPOINT},
    "config_kwargs": {"signature_version": "s3v4",
                      "s3": {"addressing_style": "path"}},
}

needs_mit = pytest.mark.skipif(
    not Path(MIT_PATH).exists(),
    reason=f"MIT source store not accessible at {MIT_PATH} "
           "(run this on the MIT machine)",
)


def _stamp(date_str: str) -> str:
    """'2012-01-01 12:00:00' -> '20120101T12' (per-date store name)."""
    return datetime.strptime(date_str, DATE_FMT).strftime("%Y%m%dT%H")


def _expected_time(date_str: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(date_str, DATE_FMT), "ns")


def _first_time_value(ds) -> np.datetime64:
    """Scalar time from a store's 'time' variable, whatever its shape."""
    return np.datetime64(np.ravel(np.asarray(ds["time"].values))[0], "ns")


# ---------------------------------------------------------------------------
# Unit tests — offline, always run
# ---------------------------------------------------------------------------

def test_mit_epoch_is_iteration_zero():
    assert mit_date_to_iteration("2011-09-13 00:00:00") == 0


def test_mit_known_iteration():
    # 2011-09-13 00:00 -> 2012-01-01 00:00 is exactly 110 days of 25-s steps.
    # NOTE: the docstring example in mit_date_to_iteration ("~1,011,456") is
    # wrong — 1,011,456 steps corresponds to 2012-07-01 16:00, not 2012-01-01.
    assert mit_date_to_iteration("2012-01-01 00:00:00") == 110 * 24 * TS_PER_HOUR  # 380_160


def test_osn_offset_is_constant_shift():
    for date_str in DATES:
        assert (osn_date_to_iteration(date_str)
                - mit_date_to_iteration(date_str)) == FIRST_WIND_RECORD_OFFSET


def test_osn_offset_is_three_days():
    # OSN numbering starts 2011-09-10, three days before the MIT epoch.
    assert FIRST_WIND_RECORD_OFFSET == 3 * 24 * TS_PER_HOUR
    assert osn_date_to_iteration("2011-09-13 00:00:00") == FIRST_WIND_RECORD_OFFSET


def test_config_dates_align_to_model_hours():
    """All yaml dates must land exactly on an hourly record."""
    for date_str in DATES:
        assert mit_date_to_iteration(date_str) % TS_PER_HOUR == 0, date_str


def test_time_idx_matches_iteration():
    for date_str in DATES:
        it = mit_date_to_iteration(date_str)
        assert mit_date_to_time_idx(date_str, ntime=10**9) == it // TS_PER_HOUR


def test_time_idx_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        mit_date_to_time_idx(DATES[0], ntime=1)


def test_date_before_model_start_raises():
    with pytest.raises(ValueError, match="before LLC4320 start"):
        mit_date_to_iteration("2011-09-12 23:59:59")


# ---------------------------------------------------------------------------
# (1) MIT source store — run on the MIT machine
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_mit():
    xr = pytest.importorskip("xarray")
    ds = xr.open_zarr(MIT_PATH, consolidated=False)
    yield ds
    ds.close()


@pytest.mark.integration
@pytest.mark.mit
@needs_mit
@pytest.mark.parametrize("date_str", DATES)
def test_mit_store_time_matches_config_date(ds_mit, date_str):
    """time[mit_date_to_time_idx(date)] in the MIT store IS the config date."""
    tidx = mit_date_to_time_idx(date_str, ds_mit.sizes["time"])
    actual = np.datetime64(ds_mit["time"].isel(time=tidx).values, "ns")
    logging.info(f"MIT   asked '{date_str}' -> time_idx {tidx} "
                 f"-> store says {actual}")
    assert actual == _expected_time(date_str), (
        f"MIT store time at idx {tidx} is {actual}, expected {date_str}"
    )


# ---------------------------------------------------------------------------
# (2) Transferred S3 store (dbof/LLC4320_RAW/SURFACE) — needs NRP creds
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.s3_dbof
@pytest.mark.parametrize("date_str", DATES)
def test_s3_surface_store_time_matches_config_date(date_str):
    """The per-date store we transferred carries the date it is named for."""
    xr = pytest.importorskip("xarray")
    store = f"s3://{S3_PREFIX}/{_stamp(date_str)}.zarr"
    ds = xr.open_zarr(store, consolidated=False,
                      storage_options=S3_STORAGE_OPTIONS)
    try:
        assert "time" in ds, f"{store} has no 'time' variable"
        actual = _first_time_value(ds)
        logging.info(f"S3    asked '{date_str}' -> {store} "
                     f"-> store says {actual}")
        assert actual == _expected_time(date_str), (
            f"{store} carries time={actual}, expected {date_str}"
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# (3) OSN kerchunk store — anonymous, network only
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.osn
@pytest.mark.parametrize("date_str", DATES)
def test_osn_kerchunk_time_matches_config_date(date_str):
    """The OSN record at osn_date_to_iteration(date) decodes to the date."""
    from dbof.global_dataset_creation.data_sources import OSN_ENDPOINT
    from dbof.llc4320_ingestion import get_raw_data

    it = osn_date_to_iteration(date_str)
    ds = get_raw_data.get_remote_llc_data(OSN_ENDPOINT, it, [0])  # 1 face is enough
    try:
        actual = _first_time_value(ds)
        logging.info(f"OSN   asked '{date_str}' -> iteration {it} "
                     f"-> store says {actual}")
        assert actual == _expected_time(date_str), (
            f"OSN iteration {it} decodes to time={actual}, expected {date_str}"
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# (4) OSN vs. our S3 store, compared directly
# ---------------------------------------------------------------------------
# Strict test version of the inline check in generate_global
# (get_raw_data.verify_osn_llc_surf_timestamp).  The inline guard swallows
# infrastructure errors (logs a warning and continues); here every failure
# mode is a hard assertion, which is what a reviewer can point to as proof.

@pytest.mark.integration
@pytest.mark.osn
@pytest.mark.s3_dbof
@pytest.mark.parametrize("date_str", DATES)
def test_osn_and_s3_surface_resolve_to_same_timestamp(date_str):
    xr = pytest.importorskip("xarray")
    from dbof.global_dataset_creation.data_sources import OSN_ENDPOINT
    from dbof.llc4320_ingestion import get_raw_data

    ds_osn = get_raw_data.get_remote_llc_data(
        OSN_ENDPOINT, osn_date_to_iteration(date_str), [0]
    )
    ds_s3 = xr.open_zarr(
        f"s3://{S3_PREFIX}/{_stamp(date_str)}.zarr",
        consolidated=False, storage_options=S3_STORAGE_OPTIONS,
    )
    try:
        t_osn, t_s3 = _first_time_value(ds_osn), _first_time_value(ds_s3)
        logging.info(f"CROSS asked '{date_str}' -> OSN says {t_osn}, "
                     f"S3 says {t_s3}")
        assert t_osn == t_s3 == _expected_time(date_str), (
            f"date '{date_str}': OSN time={t_osn}, S3 time={t_s3}"
        )
    finally:
        ds_osn.close()
        ds_s3.close()
