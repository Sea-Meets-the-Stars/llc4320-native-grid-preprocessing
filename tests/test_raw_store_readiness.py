"""
test_raw_store_readiness.py
---------------------------
Readiness checks for the migrated raw LLC4320 stores under
``s3://dbof/LLC4320_RAW/{DEPTH,SURFACE}`` (see ``docs/Data_Organization.md`` for the
bucket layout and available dates).

For each subset (DEPTH, SURFACE) this verifies exactly what the global
pipeline (``generate_global.py``) needs at startup:

1. **grid.zarr exists and opens** through the pipeline's own loader
   (``get_llc_depth_gridfile``), with the essential grid variables present
   and a sample value actually readable from S3.
2. **Every expected timestep store has a readable timestamp** through the
   pipeline's own loader (``get_llc_timestep_data``, with the same chunks /
   storage options each pipeline passes), and the stored ``time`` value
   matches the datetime encoded in the store name (``YYYYMMDDTHH.zarr``).

A discovery test additionally lists each prefix on S3 and checks that

* the expected date list (kept in sync with ``docs/Data_Organization.md``) is a
  subset of what is actually in the bucket, and
* any *extra* store found in the bucket also opens with a timestamp
  matching its name (so nothing unreadable is lurking under the prefix).

A hermetic test pins the pipeline's ``data_sources`` constants to the
``LLC4320_RAW`` prefixes, so a source-dict regression is caught offline.

NOTE: ``LLC_DEPTH_SOURCE["grid_folder"]`` still points the DEPTH pipeline's
*grid* read at the legacy ``dbof/LLC4320`` prefix.  Once
``test_grid_store_opens[DEPTH]`` passes against the new location, flip
``grid_folder`` to ``LLC4320_RAW/DEPTH`` in
``dbof/global_dataset_creation/data_sources.py``.

Running (skipped by default; needs network + NRP S3 credentials)::

    pytest --run-integration -m s3_dbof tests/test_raw_store_readiness.py
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pytest

from dbof.llc4320_ingestion import get_raw_data
from dbof.llc4320_ingestion.date_iterations import DATE_FMT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
BUCKET = "dbof"
RAW_PREFIX = "LLC4320_RAW"


def _surface_stamps() -> list[str]:
    """Expected SURFACE stores: 20111209T12 plus the 1st and 9th of each
    month Jan-Nov 2012 at 12:00 (see docs/Data_Organization.md)."""
    stamps = ["20111209T12"]
    for mm in range(1, 12):
        stamps.append(f"2012{mm:02d}01T12")
        stamps.append(f"2012{mm:02d}09T12")
    return stamps


# Per-subset: S3 folder, expected date stamps, and the exact chunks /
# storage options each pipeline passes to get_llc_timestep_data
# (DEPTH: generate_global.py DEPTH branch; SURFACE: the loader defaults
# used by the SURF pipeline's forcing-variable reads).
SUBSETS = {
    "DEPTH": {
        "folder": f"{RAW_PREFIX}/DEPTH",
        "expected_stamps": ["20121109T12"],
        "chunks": get_raw_data.llc_depth_timestep_chunks,
        "storage_options": get_raw_data._llc_depth_storage_options(S3_ENDPOINT),
    },
    "SURFACE": {
        "folder": f"{RAW_PREFIX}/SURFACE",
        "expected_stamps": _surface_stamps(),
        "chunks": get_raw_data.llc_surf_timestep_chunks,
        "storage_options": get_raw_data._llc_surf_storage_options(S3_ENDPOINT),
    },
}

# Grid variables generate_global cannot run without.
ESSENTIAL_GRID_VARS = ["XC", "YC", "Depth", "hFacC", "dxC", "dyC", "rA"]

STAMP_FMT = "%Y%m%dT%H"

DATE_PARAMS = [
    pytest.param(subset, stamp, id=f"{subset}-{stamp}")
    for subset, cfg in SUBSETS.items()
    for stamp in cfg["expected_stamps"]
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stamp_to_date_str(stamp: str) -> str:
    return datetime.strptime(stamp, STAMP_FMT).strftime(DATE_FMT)


def _list_zarr_stores(folder: str) -> set[str]:
    """Store names (``*.zarr``) directly under ``{BUCKET}/{folder}`` on S3,
    listed with the pipeline's own synchronous filesystem."""
    from dbof.io.filesystems import create_s3_filesystems

    _, fs_synch = create_s3_filesystems(S3_ENDPOINT)
    entries = fs_synch.ls(f"{BUCKET}/{folder}", detail=False)
    return {p.rstrip("/").rsplit("/", 1)[-1] for p in entries
            if p.rstrip("/").endswith(".zarr")}


def _open_time(subset: str, stamp: str):
    """Open ``{stamp}.zarr`` exactly the way the pipeline does and return
    its decoded timestamp as ``np.datetime64[ns]``."""
    cfg = SUBSETS[subset]
    ds = get_raw_data.get_llc_timestep_data(
        S3_ENDPOINT,
        BUCKET,
        cfg["folder"],
        _stamp_to_date_str(stamp),
        vars_requested=["time"],
        chunks=cfg["chunks"],
        storage_options=cfg["storage_options"],
    )
    try:
        assert "time" in ds, (
            f"{subset}/{stamp}.zarr has no 'time' variable — re-run "
            f"transfer_llc4320.py with 'time' in transfer.variables"
        )
        return np.datetime64(np.asarray(ds["time"].values).ravel()[0], "ns")
    finally:
        ds.close()


def _check_timestamp(subset: str, stamp: str) -> None:
    expected = np.datetime64(datetime.strptime(stamp, STAMP_FMT), "ns")
    actual = _open_time(subset, stamp)
    logging.info(f"{subset}/{stamp}.zarr: time={actual} (expected {expected})")
    assert actual == expected, (
        f"{subset}/{stamp}.zarr timestamp mismatch: store time={actual}, "
        f"store name implies {expected}"
    )


# ---------------------------------------------------------------------------
# Hermetic: pipeline source constants point at LLC4320_RAW
# ---------------------------------------------------------------------------

def test_pipeline_data_sources_point_at_raw():
    """generate_global reads from data_sources constants — pin them to the
    prefixes verified by the integration tests below."""
    from dbof.global_dataset_creation import data_sources

    assert data_sources.LLC_SURF_SOURCE["folder"] == f"{RAW_PREFIX}/SURFACE"
    assert data_sources.LLC_DEPTH_SOURCE["folder"] == f"{RAW_PREFIX}/DEPTH"


# ---------------------------------------------------------------------------
# (1) grid.zarr exists and opens via the pipeline loader
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.s3_dbof
@pytest.mark.parametrize("subset", list(SUBSETS))
def test_grid_store_opens(subset):
    cfg = SUBSETS[subset]
    grid = get_raw_data.get_llc_depth_gridfile(S3_ENDPOINT, BUCKET, cfg["folder"])
    try:
        assert grid.sizes.get("face") == 13, (
            f"{subset} grid.zarr: expected 13 faces, got {grid.sizes.get('face')}"
        )
        missing = [v for v in ESSENTIAL_GRID_VARS if v not in grid]
        assert not missing, f"{subset} grid.zarr missing variables: {missing}"

        # Actually pull one value from S3 — metadata alone can exist while
        # chunk objects are missing or corrupt.
        val = float(grid["XC"].isel(face=0, j=0, i=0).compute())
        assert np.isfinite(val), f"{subset} grid.zarr: XC sample is not finite"
        logging.info(f"{subset}/grid.zarr OK: vars={sorted(grid.data_vars)}")
    finally:
        grid.close()


# ---------------------------------------------------------------------------
# (2) every expected timestep store has a readable, correct timestamp
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.s3_dbof
@pytest.mark.parametrize("subset,stamp", DATE_PARAMS)
def test_expected_store_timestamp(subset, stamp):
    try:
        _check_timestamp(subset, stamp)
    except FileNotFoundError as exc:
        pytest.fail(
            f"{subset}/{stamp}.zarr not found at "
            f"s3://{BUCKET}/{SUBSETS[subset]['folder']}/ — expected per "
            f"docs/Data_Organization.md ({exc})"
        )


# ---------------------------------------------------------------------------
# Discovery: expected ⊆ bucket, and every extra store is also readable
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.s3_dbof
@pytest.mark.parametrize("subset", list(SUBSETS))
def test_bucket_matches_expected_inventory(subset):
    cfg = SUBSETS[subset]
    discovered = _list_zarr_stores(cfg["folder"])
    logging.info(f"{subset}: {len(discovered)} zarr stores under "
                 f"s3://{BUCKET}/{cfg['folder']}/")

    assert "grid.zarr" in discovered, f"{subset}: grid.zarr not in bucket"

    expected = {f"{s}.zarr" for s in cfg["expected_stamps"]}
    missing = sorted(expected - discovered)
    assert not missing, (
        f"{subset}: stores expected per docs/Data_Organization.md but absent from "
        f"bucket: {missing}"
    )

    # Stores in the bucket beyond the documented list: verify them too, and
    # surface them so docs/Data_Organization.md gets updated.
    extras = sorted(discovered - expected - {"grid.zarr"})
    if extras:
        logging.warning(
            f"{subset}: undocumented stores in bucket (add to "
            f"docs/Data_Organization.md?): {extras}"
        )
    failures = []
    for name in extras:
        stamp = name.removesuffix(".zarr")
        try:
            _check_timestamp(subset, stamp)
        except Exception as exc:  # noqa: BLE001 — collect all, report at end
            failures.append(f"{name}: {exc}")
    assert not failures, (
        f"{subset}: undocumented stores failed the timestamp check:\n  "
        + "\n  ".join(failures)
    )
