"""
Integration tests for transferred-data integrity.

Test-suite version of the ad-hoc notebooks
(``notebooks_dev/check_ocetaux*.ipynb``, ``test_chunk_*_transfer.ipynb``):
the *assertions* here are numeric (all-NaN detection, exact equality,
matching shapes) — that is what CI or a reviewer runs.  The images the
notebooks produce are diagnostics, not tests, so they are opt-in artifacts:

    DBOF_SAVE_PLOTS=1 pytest -m s3 tests/test_transfer_integrity.py

writes side-by-side PNGs for every compared date to ``tests/artifacts/``
(using the headless Agg backend).  Look at them when a test fails; ignore
them when it passes.

Selection (plain ``pytest`` skips all of these — see pyproject.toml):

    pytest -m s3_dbof tests/test_transfer_integrity.py  # NEW vs OLD S3 stores
    pytest -m mit     tests/test_transfer_integrity.py  # MIT-side tests
                                                        # (on the MIT machine)

If the dbof S3 bucket is down, only the MIT-source corruption check can run:

    pytest tests/test_transfer_integrity.py -m "mit and not s3_dbof"

Environment overrides: ``DBOF_TEST_VAR`` (default oceTAUY),
``DBOF_TEST_FACE`` (default 1).
"""
from __future__ import annotations

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

VAR = os.environ.get("DBOF_TEST_VAR", "oceTAUY")
FACE = int(os.environ.get("DBOF_TEST_FACE", "1"))
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


def _load_s3_face(store: str) -> np.ndarray:
    import xarray as xr
    ds = xr.open_zarr("s3://" + store, consolidated=False,
                      storage_options=S3_STORAGE_OPTIONS)
    try:
        return np.squeeze(ds[VAR].isel(face=FACE).values)
    finally:
        ds.close()


def _save_plot_pair(xa, xb, labels, date_str, tag):
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
        ax.set_title(f"{label}\n{date_str}  face {FACE}  "
                     f"NaN={np.isnan(x).mean():.1%}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    out = ARTIFACT_DIR / f"{tag}_{VAR}_face{FACE}_{_stamp(date_str)}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _assert_faces_match(xa, xb, labels, date_str):
    """Shared numeric checks: not-all-NaN, same shape, exact equality."""
    la, lb = labels
    nan_a = float(np.isnan(xa).mean())
    nan_b = float(np.isnan(xb).mean())
    assert nan_a < 1.0, f"{la} is ALL-NaN for {date_str} (face {FACE})"
    assert nan_b < 1.0, f"{lb} is ALL-NaN for {date_str} (face {FACE})"
    assert xa.shape == xb.shape, (
        f"shape mismatch for {date_str}: {la}={xa.shape} {lb}={xb.shape}"
    )
    nan_mm = int((np.isnan(xa) != np.isnan(xb)).sum())
    identical = np.allclose(xa, xb, equal_nan=True, rtol=0, atol=0)
    if not identical:
        with np.errstate(invalid="ignore"):
            d = np.abs(xa - xb)
        max_d = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
        pytest.fail(
            f"{la} != {lb} for {date_str} (face {FACE}): "
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
@pytest.mark.parametrize("date_str", DATES)
def test_new_s3_matches_old_s3(date_str):
    st = _stamp(date_str)
    xa = _load_s3_face(NEW_TMPL.format(stamp=st))
    xb = _load_s3_face(OLD_TMPL.format(stamp=st))
    labels = ("NEW (LLC4320_RAW/SURFACE)", "OLD (LLC4320)")
    _save_plot_pair(xa, xb, labels, date_str, tag="s3_new_vs_old")
    _assert_faces_match(xa, xb, labels, date_str)


# ---------------------------------------------------------------------------
# NEW S3 copy vs original MIT source — run on the MIT machine
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
def test_mit_source_face_has_valid_data(ds_mit, date_str):
    """Corrupt-source guard (no S3 needed): the sampled face chunk must not be
    all zeros/NaNs — same check the transfer pipeline now runs inline."""
    from dbof.transfer.pipeline import date_has_valid_data

    tidx = mit_date_to_time_idx(date_str, ds_mit.sizes["time"])
    assert date_has_valid_data(ds_mit, VAR, tidx, face=FACE), (
        f"{VAR} at {date_str} looks corrupt in the MIT source "
        f"(all zeros/NaNs in face-{FACE} sample)"
    )


@pytest.mark.integration
@pytest.mark.mit
@pytest.mark.s3_dbof
@needs_mit
@pytest.mark.parametrize("date_str", DATES)
def test_new_s3_matches_mit_source(ds_mit, date_str):
    tidx = mit_date_to_time_idx(date_str, ds_mit.sizes["time"])
    xa = _load_s3_face(NEW_TMPL.format(stamp=_stamp(date_str)))
    xb = np.squeeze(ds_mit[VAR].isel(time=tidx, face=FACE).values)
    labels = ("NEW (LLC4320_RAW/SURFACE)", "MIT (orcd source)")
    _save_plot_pair(xa, xb, labels, date_str, tag="s3_new_vs_mit")
    _assert_faces_match(xa, xb, labels, date_str)
