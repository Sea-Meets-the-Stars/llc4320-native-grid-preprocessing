"""
Integration tests for transferred-data integrity.

Test-suite version of the ad-hoc notebooks
(``notebooks_dev/check_ocetaux*.ipynb``, ``test_chunk_*_transfer.ipynb``):
the *assertions* here are numeric (all-NaN detection, exact equality,
matching shapes) — that is what CI or a reviewer runs.  The images the
notebooks produce are diagnostics, not tests, so they are opt-in artifacts:

    DBOF_SAVE_PLOTS=1 pytest --run-integration -m s3_dbof tests/test_transfer_integrity.py

writes side-by-side PNGs for every compared chunk to ``tests/artifacts/``
(using the headless Agg backend).  Look at them when a test fails; ignore
them when it passes.

To keep runtime sane, every check reads a single TILE x TILE chunk of one
face (defaults: face 1, 720x720 — the same guard the transfer pipeline runs
inline), for **every time-varying variable and every date** in
``configs/transfer/run_surface.yaml``.

Running (integration tests are skipped unless ``--run-integration`` is
passed — see conftest.py):

    pytest --run-integration -m s3_dbof tests/test_transfer_integrity.py  # NEW vs OLD S3
    pytest --run-integration -m mit     tests/test_transfer_integrity.py  # MIT-side
                                                                          # (on MIT machine)

If the dbof S3 bucket is down, only the MIT-source corruption check can run:

    pytest --run-integration -m "mit and not s3_dbof" tests/test_transfer_integrity.py

Environment overrides: ``DBOF_TEST_VAR`` (single variable instead of all),
``DBOF_TEST_FACE`` (default 1), ``DBOF_TEST_TILE`` (default 720),
``DBOF_SAVE_PLOTS=1`` (write PNG artifacts).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from dbof.llc4320_ingestion.date_iterations import DATE_FMT, mit_date_to_time_idx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
with open(REPO / "configs" / "transfer" / "run_surface.yaml") as _f:
    _CFG = yaml.safe_load(_f)

MIT_PATH = _CFG["data"]["MIT_data_path"]
DATES = list(dict.fromkeys(_CFG["data"]["date_iterations"]))
S3_ENDPOINT = _CFG["output"]["s3_endpoint"]

# Dates known to be corrupt in the MIT source (e.g. oceQnet all zeros/NaNs).
# These are reported as XFAIL rather than failures; remove a date here once
# the source data is repaired.
KNOWN_CORRUPT_DATES = {"2012-08-01 12:00:00"}
DATE_PARAMS = [
    pytest.param(d, marks=pytest.mark.xfail(
        reason="known-corrupt date in MIT source"))
    if d in KNOWN_CORRUPT_DATES else d
    for d in DATES
]

# All time-varying variables from the config, or a single one via env.
# SIarea is excluded: sea-ice is legitimately all zeros away from the poles.
VARS = ([os.environ["DBOF_TEST_VAR"]] if "DBOF_TEST_VAR" in os.environ
        else [v for v in _CFG["transfer"]["variables"]
              if v not in ("time", "SIarea")])
FACE = int(os.environ.get("DBOF_TEST_FACE", "1"))
TILE = int(os.environ.get("DBOF_TEST_TILE", "720"))
SAVE_PLOTS = os.environ.get("DBOF_SAVE_PLOTS", "0") == "1"
ARTIFACT_DIR = REPO / "tests" / "artifacts"

NEW_TMPL = "dbof/LLC4320_RAW/SURFACE/{stamp}.zarr"   # transferred copy
OLD_TMPL = "dbof/LLC4320/{stamp}.zarr"               # first S3 copy

S3_STORAGE_OPTIONS = {
    "anon": False,
    "client_kwargs": {"endpoint_url": S3_ENDPOINT},
    "config_kwargs": {"signature_version": "s3v4",
                      "s3": {"addressing_style": "path"}},
}

needs_mit = pytest.mark.skipif(
    not Path(MIT_PATH).exists(),
    reason=f"MIT source store not accessible at {MIT_PATH}",
)


def _stamp(date_str: str) -> str:
    return datetime.strptime(date_str, DATE_FMT).strftime("%Y%m%dT%H")


def _chunk(da) -> np.ndarray:
    """One TILE x TILE chunk of FACE, squeezed to 2D.  Slices whatever
    horizontal dims the variable has (handles staggered i_g/j_g grids)."""
    da = da.isel(face=FACE)
    sl = {d: slice(0, TILE) for d in da.dims if d[0] in "ij"}
    return np.squeeze(da.isel(sl).values)


def _load_s3_chunk(store: str, var: str) -> np.ndarray:
    import xarray as xr
    ds = xr.open_zarr("s3://" + store, consolidated=False,
                      storage_options=S3_STORAGE_OPTIONS)
    try:
        return _chunk(ds[var])
    finally:
        ds.close()


def _save_plot_pair(xa, xb, labels, var, date_str, tag):
    """Optional diagnostic artifact — never asserted on."""
    if not SAVE_PLOTS:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    finite = np.concatenate([x[np.isfinite(x)].ravel() for x in (xa, xb)])
    vmin, vmax = (np.percentile(finite, [1, 99]) if finite.size else (-1, 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, x, label in zip(axes, (xa, xb), labels):
        im = ax.imshow(x, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(f"{label}\n{var}  {date_str}  face {FACE}  "
                     f"NaN={np.isnan(x).mean():.1%}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    out = ARTIFACT_DIR / f"{tag}_{var}_face{FACE}_{_stamp(date_str)}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _assert_chunks_match(xa, xb, labels, var, date_str):
    """Shared numeric checks: not-all-NaN, same shape, exact equality."""
    la, lb = labels
    nan_a = float(np.isnan(xa).mean())
    nan_b = float(np.isnan(xb).mean())
    logging.info(f"{var} '{date_str}' face {FACE} [{TILE}x{TILE}]: "
                 f"NaN% {la}={nan_a:.1%} {lb}={nan_b:.1%}")
    assert nan_a < 1.0, f"{la} {var} is ALL-NaN for {date_str} (face {FACE})"
    assert nan_b < 1.0, f"{lb} {var} is ALL-NaN for {date_str} (face {FACE})"
    assert xa.shape == xb.shape, (
        f"shape mismatch for {var} {date_str}: {la}={xa.shape} {lb}={xb.shape}"
    )
    nan_mm = int((np.isnan(xa) != np.isnan(xb)).sum())
    identical = np.allclose(xa, xb, equal_nan=True, rtol=0, atol=0)
    if not identical:
        with np.errstate(invalid="ignore"):
            d = np.abs(xa - xb)
        max_d = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
        pytest.fail(
            f"{la} != {lb} for {var} {date_str} (face {FACE}): "
            f"max|diff|={max_d:.3e}, nan-mask mismatches={nan_mm}, "
            f"NaN%: {la}={nan_a:.1%} {lb}={nan_b:.1%}"
            + (f"  [plots in {ARTIFACT_DIR}]" if SAVE_PLOTS else
               "  [re-run with DBOF_SAVE_PLOTS=1 for plots]")
        )


# ---------------------------------------------------------------------------
# NEW S3 copy vs OLD S3 copy
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.s3_dbof
@pytest.mark.parametrize("var", VARS)
@pytest.mark.parametrize("date_str", DATE_PARAMS)
def test_new_s3_matches_old_s3(date_str, var):
    st = _stamp(date_str)
    xa = _load_s3_chunk(NEW_TMPL.format(stamp=st), var)
    xb = _load_s3_chunk(OLD_TMPL.format(stamp=st), var)
    labels = ("NEW (LLC4320_RAW/SURFACE)", "OLD (LLC4320)")
    _save_plot_pair(xa, xb, labels, var, date_str, tag="s3_new_vs_old")
    _assert_chunks_match(xa, xb, labels, var, date_str)


# ---------------------------------------------------------------------------
# MIT-side tests — run on the MIT machine
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
@pytest.mark.parametrize("var", VARS)
@pytest.mark.parametrize("date_str", DATE_PARAMS)
def test_mit_source_face_has_valid_data(ds_mit, date_str, var):
    """Corrupt-source guard (no S3 needed): the sampled face chunk must not be
    all zeros/NaNs — same check the transfer pipeline now runs inline."""
    from dbof.transfer.pipeline import date_has_valid_data

    tidx = mit_date_to_time_idx(date_str, ds_mit.sizes["time"])
    ok = date_has_valid_data(ds_mit, var, tidx, face=FACE, tile=TILE)
    logging.info(f"MIT {var} '{date_str}' face {FACE} sample "
                 f"{'looks valid' if ok else 'is ALL zeros/NaNs'}")
    assert ok, (
        f"{var} at {date_str} looks corrupt in the MIT source "
        f"(all zeros/NaNs in face-{FACE} sample)"
    )


@pytest.mark.integration
@pytest.mark.mit
@pytest.mark.s3_dbof
@needs_mit
@pytest.mark.parametrize("var", VARS)
@pytest.mark.parametrize("date_str", DATE_PARAMS)
def test_new_s3_matches_mit_source(ds_mit, date_str, var):
    tidx = mit_date_to_time_idx(date_str, ds_mit.sizes["time"])
    xa = _load_s3_chunk(NEW_TMPL.format(stamp=_stamp(date_str)), var)
    xb = _chunk(ds_mit[var].isel(time=tidx))
    labels = ("NEW (LLC4320_RAW/SURFACE)", "MIT (orcd source)")
    _save_plot_pair(xa, xb, labels, var, date_str, tag="s3_new_vs_mit")
    _assert_chunks_match(xa, xb, labels, var, date_str)
