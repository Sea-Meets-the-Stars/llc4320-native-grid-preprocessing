"""Vector handling for stitched LLC maps: Path A vs Path B.

Two ways to move staggered vectors (U/V, oceTAUX/Y) onto the stitched
lat-lon rectangle:

  A (production, ECCO-style*):  interp to tracer -> CS/SN rotate -> scalar stitch
  B (legacy, xmitgcm-style**):  interp to tracer -> 'mate' pairs -> vector stitch

Synthetic tests (offline, run by default):

  test_difference_A_vs_B          how the two outputs differ
  test_correctness_A_vs_B         which one reproduces a known vector
  test_registration_vs_scalars    which one co-registers with scalar channels
  test_B_is_valid_for_staggered_input  B fed staggered input (its design case)
                                  agrees with A -> the legacy bug was the
                                  interp-BEFORE-vector-stitch ordering, not xmitgcm

Real-data tests (opt-in; images to tests/output/):

  test_real_grid_cs_sn_convention  LLC4320 CS/SN match the stitch convention
                                   off-cap                  (DBOF_GRID_CHECK=1)
  test_real_grid_cs_sn_face_plot   CS/SN per face, LLC layout (DBOF_GRID_CHECK=1)
  test_real_snapshot_A_vs_B        both paths on one raw OSN wind snapshot
                                                            (DBOF_RAW_CHECK=1)

*  https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Gradient_calc_on_native_grid.html
** https://xmitgcm.readthedocs.io/en/latest/_modules/xmitgcm/llcreader/llcmodel.html
"""

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import dbof.utils.native_gradient as ng
from dbof.llc4320_ingestion.grid import set_xgcm_grid
from dbof.utils.faces_to_latlon import (
    faces_dataset_to_latlon,
    interp_staggered_to_tracer,
    set_vector_pair_attrs,
)

N_FACES = 13
CAP_FACE = 6
ROTATED_FACES = (7, 8, 9, 10, 11, 12)
OUTPUT_DIR = Path(__file__).parent / "output"

#: Stitched LLC4320 rectangle: rotated-facet columns start here.
_RECT_ROTATED_COL_START = 2 * 4320


# ---------------------------------------------------------------------------
# Synthetic grid helpers
# ---------------------------------------------------------------------------

def _make_synthetic_llc_dataset(n=90):
    """13-face dataset with smooth staggered U/V and xgcm comodo attrs."""
    i = xr.DataArray(np.arange(n), dims="i", attrs={"axis": "X"})
    i_g = xr.DataArray(np.arange(n), dims="i_g",
                       attrs={"axis": "X", "c_grid_axis_shift": -0.5})
    j = xr.DataArray(np.arange(n), dims="j", attrs={"axis": "Y"})
    j_g = xr.DataArray(np.arange(n), dims="j_g",
                       attrs={"axis": "Y", "c_grid_axis_shift": -0.5})
    face = xr.DataArray(np.arange(N_FACES), dims="face")

    ff = np.arange(N_FACES)[:, None, None]
    jj = np.arange(n)[None, :, None]
    ii = np.arange(n)[None, None, :]
    u = np.sin(2 * np.pi * ii / n) + 0.5 * np.cos(2 * np.pi * jj / n) + 0.10 * ff + 2.0
    v = np.cos(4 * np.pi * ii / n) - 0.7 * np.sin(2 * np.pi * jj / n) - 0.05 * ff - 3.0

    return xr.Dataset(
        {"U": xr.DataArray(u, dims=("face", "j", "i_g")),
         "V": xr.DataArray(v, dims=("face", "j_g", "i"))},
        coords={"face": face, "i": i, "i_g": i_g, "j": j, "j_g": j_g},
    )


def _stitch(ds_faces):
    """Face dataset -> lat-lon rectangle (repo's xmitgcm wrapper)."""
    return faces_dataset_to_latlon(ds_faces, metric_vector_pairs=[])


def _face_id_map(ds):
    """Scalar-stitch a per-face constant: which face each output pixel came from."""
    n_j, n_i = ds.sizes["j"], ds.sizes["i"]
    data = np.broadcast_to(np.arange(N_FACES, dtype=float)[:, None, None],
                           (N_FACES, n_j, n_i)).copy()
    da = xr.DataArray(data, dims=("face", "j", "i"),
                      coords={"face": ds.face, "j": ds.j, "i": ds.i},
                      name="face_id")
    return _stitch(xr.Dataset({"face_id": da}).chunk({"face": 1}))["face_id"].values


def _tracer_da(ds, data):
    return xr.DataArray(data, dims=("face", "j", "i"),
                        coords={"face": ds.face, "j": ds.j, "i": ds.i})


def _measure_stitch_rotation(ds, face_id):
    """Per-face (CS, SN) equivalent of xmitgcm's vector stitch, measured by
    stitching unit probe vectors.  Median per face skips the pad-seam zeros."""
    def _probe(u_val, v_val):
        shape = (N_FACES, ds.sizes["j"], ds.sizes["i"])
        dsp = xr.Dataset({"pu": _tracer_da(ds, np.full(shape, u_val)),
                          "pv": _tracer_da(ds, np.full(shape, v_val))})
        set_vector_pair_attrs(dsp, vector_pairs=[("pu", "pv")])
        out = _stitch(dsp.chunk({"face": 1}))
        return out["pu"].values, out["pv"].values

    e1, n1 = _probe(1.0, 0.0)   # (E, N) = ( CS, SN)
    e2, n2 = _probe(0.0, 1.0)   # (E, N) = (-SN, CS)

    cs = np.full(N_FACES, np.nan)
    sn = np.full(N_FACES, np.nan)
    for f in range(N_FACES):
        sel = face_id == f
        if not sel.any():                       # cap face: dropped by stitch
            continue
        cs[f], sn[f] = np.median(e1[sel]), np.median(n1[sel])
        assert np.isclose(np.median(e2[sel]), -sn[f], atol=1e-12)
        assert np.isclose(np.median(n2[sel]), cs[f], atol=1e-12)
    cs[CAP_FACE], sn[CAP_FACE] = 1.0, 0.0       # unused (cap dropped)
    return cs, sn


def _synthetic_case(n=90):
    """Synthetic dataset with CS/SN set to the stitch's measured convention.
    Returns (ds, face_id, grid)."""
    ds = _make_synthetic_llc_dataset(n)
    face_id = _face_id_map(ds)
    cs, sn = _measure_stitch_rotation(ds, face_id)
    shape = (N_FACES, n, n)
    ds["CS"] = _tracer_da(ds, np.broadcast_to(cs[:, None, None], shape).copy())
    ds["SN"] = _tracer_da(ds, np.broadcast_to(sn[:, None, None], shape).copy())
    grid = set_xgcm_grid(ds, use_connections=True)
    return ds, face_id, grid


# ---------------------------------------------------------------------------
# The two paths
# ---------------------------------------------------------------------------

def _path_a(u, v, ds, grid, interpolate=True):
    """A: (interp ->) CS/SN rotate -> scalar stitch.  Returns (east, north)."""
    e, n = ng.rotate_vector_to_geographic(u, v, ds, grid, interpolate=interpolate)
    e.attrs, n.attrs = {}, {}                   # no mate attrs -> scalar path
    out = _stitch(xr.Dataset({"E": e, "N": n}).chunk({"face": 1}))
    return out["E"].values, out["N"].values


def _path_b(u, v, grid, interpolate=True):
    """B: (interp ->) mate pairs -> vector stitch.  Returns (east, north)."""
    fields = {"U": u, "V": v}
    if interpolate:
        interp_staggered_to_tracer(fields, grid)
    dsb = xr.Dataset(fields)
    set_vector_pair_attrs(dsb)
    out = _stitch(dsb.chunk({"face": 1}))
    return out["U"].values, out["V"].values


def _save_panels(rows, path, suptitle, mark_halves=True):
    """rows: list of (label, list of (title, array, cmap, vmin, vmax)).
    mark_halves: divider + labels for the un-rotated (left, faces 0-5) and
    rotated (right, faces 7-12) halves of the stitched rectangle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), ncol, figsize=(5.5 * ncol, 4 * len(rows)),
                             constrained_layout=True, squeeze=False)
    for r, (_, panels) in enumerate(rows):
        for c, (title, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
            if mark_halves:
                h, w = arr.shape
                ax.axvline(w / 2, color="k", ls="--", lw=0.8)
                box = dict(facecolor="white", alpha=0.75, edgecolor="none",
                           pad=1.5)
                ax.text(0.25, 0.97, "faces 0–5\n(un-rotated)", fontsize=7,
                        ha="center", va="top", transform=ax.transAxes, bbox=box)
                ax.text(0.75, 0.97, "faces 7–12\n(rotated 90°)", fontsize=7,
                        ha="center", va="top", transform=ax.transAxes, bbox=box)
    fig.suptitle(suptitle, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Synthetic tests
# ===========================================================================

def test_difference_A_vs_B():
    """A and B agree for east.  B's north is displaced one pixel in latitude
    over the rotated half, with one zero-padded seam row per rotated facet."""
    n = 90
    ds, face_id, grid = _synthetic_case(n)
    a_e, a_n = _path_a(ds.U, ds.V, ds, grid)
    b_e, b_n = _path_b(ds.U, ds.V, grid)
    rot = np.isin(face_id, ROTATED_FACES)

    # east: identical
    np.testing.assert_allclose(a_e, b_e, rtol=0, atol=1e-12)

    # north: identical on the un-rotated half ...
    dn = a_n - b_n
    assert not (np.abs(dn) > 1e-12)[~rot].any()

    # ... one-pixel latitude displacement on the rotated half, plus pad seam
    resid = (np.abs(np.roll(a_n, 1, axis=0) - b_n) > 1e-12) & rot
    assert resid.sum() <= 2 * n                       # seam rows only
    assert np.allclose(b_n[resid], 0.0, atol=1e-15)   # ... and they are zeros
    assert np.abs(dn)[rot].max() > 0.01               # displacement is real

    vmax = np.nanmax(np.abs(a_n))
    dmax = np.abs(dn).max()
    _save_panels(
        [("E", [("east A", a_e, "RdBu_r", -vmax, vmax),
                ("east B", b_e, "RdBu_r", -vmax, vmax),
                ("east A-B (=0)", a_e - b_e, "PuOr", -dmax, dmax)]),
         ("N", [("north A", a_n, "RdBu_r", -vmax, vmax),
                ("north B", b_n, "RdBu_r", -vmax, vmax),
                ("north A-B", dn, "PuOr", -dmax, dmax)])],
        OUTPUT_DIR / "difference_A_vs_B.png",
        "A vs B: east identical; B north shifted 1 px on rotated half + zero seams")


def test_correctness_A_vs_B():
    """Known geographic vector at cell centres, expressed in model basis:
    A reproduces it exactly; B corrupts the north on the rotated half.
    (interpolate=False isolates the stitch; the interp step is shared.)"""
    n = 90
    ds, face_id, _ = _synthetic_case(n)
    cs, sn = ds.CS.values, ds.SN.values
    rot = np.isin(face_id, ROTATED_FACES)

    ff = np.arange(N_FACES)[:, None, None]
    jj = np.arange(n)[None, :, None]
    ii = np.arange(n)[None, None, :]
    e_true = np.sin(2 * np.pi * ii / n) + 0.3 * jj / n + 0.2 * ff + 1.0
    n_true = np.cos(2 * np.pi * jj / n) - 0.4 * ii / n - 0.1 * ff - 2.0

    # same vector in the model basis (inverse rotation), per cell
    u = _tracer_da(ds, e_true * cs + n_true * sn)
    v = _tracer_da(ds, -e_true * sn + n_true * cs)

    # reference: truth values relocated like every scalar in the product
    ref = _stitch(xr.Dataset({"E": _tracer_da(ds, e_true),
                              "N": _tracer_da(ds, n_true)}).chunk({"face": 1}))
    e_ref, n_ref = ref["E"].values, ref["N"].values

    a_e, a_n = _path_a(u, v, ds, None, interpolate=False)
    b_e, b_n = _path_b(u, v, None, interpolate=False)

    assert np.abs(a_e - e_ref).max() == 0.0
    assert np.abs(a_n - n_ref).max() == 0.0
    assert np.abs(b_e - e_ref).max() == 0.0
    bad = np.abs(b_n - n_ref) > 1e-12
    assert not bad[~rot].any() and bad[rot].all()

    err = np.abs(np.stack([a_n - n_ref, b_n - n_ref]))
    emax = np.percentile(err[err > 0], 98) if (err > 0).any() else 1e-16
    _save_panels(
        [("E", [("truth east", e_ref, "RdBu_r", None, None),
                ("A east err (=0)", a_e - e_ref, "PuOr", -emax, emax),
                ("B east err (=0)", b_e - e_ref, "PuOr", -emax, emax)]),
         ("N", [("truth north", n_ref, "RdBu_r", None, None),
                ("A north err (=0)", a_n - n_ref, "PuOr", -emax, emax),
                (f"B north err (max {np.abs(b_n - n_ref).max():.2f})",
                 b_n - n_ref, "PuOr", -emax, emax)])],
        OUTPUT_DIR / "correctness_A_vs_B.png",
        "Ground truth: A exact everywhere; B wrong for north on rotated half")


def test_registration_vs_scalars():
    """Every scalar channel is placed by the scalar stitch.  A's components
    land on the same pixel as their own cell's scalar; B's north lands one
    cell off on the rotated half."""
    n = 90
    ds, face_id, _ = _synthetic_case(n)
    cs, sn = ds.CS.values, ds.SN.values
    rot = np.isin(face_id, ROTATED_FACES)

    # unique per-cell marker, treated like any scalar channel (Theta, mask...)
    marker = (np.arange(N_FACES)[:, None, None] * 1e5
              + np.arange(n)[None, :, None] * 1e2
              + np.arange(n)[None, None, :]).astype(float)
    e_true = np.sin(marker * 1e-3)              # vector defined via the marker
    n_true = np.cos(marker * 1e-3)

    u = _tracer_da(ds, e_true * cs + n_true * sn)
    v = _tracer_da(ds, -e_true * sn + n_true * cs)

    marker_ll = _stitch(xr.Dataset(
        {"m": _tracer_da(ds, marker)}).chunk({"face": 1}))["m"].values

    a_e, a_n = _path_a(u, v, ds, None, interpolate=False)
    b_e, b_n = _path_b(u, v, None, interpolate=False)

    # co-registration: value at each pixel must match the function of the
    # scalar marker that landed on that same pixel
    assert np.abs(a_e - np.sin(marker_ll * 1e-3)).max() == 0.0
    assert np.abs(a_n - np.cos(marker_ll * 1e-3)).max() == 0.0
    assert np.abs(b_e - np.sin(marker_ll * 1e-3)).max() == 0.0
    mis = np.abs(b_n - np.cos(marker_ll * 1e-3)) > 1e-12
    assert not mis[~rot].any() and mis[rot].all()

    # ... and B's misplaced north values belong to the adjacent cell
    shifted = np.cos(np.roll(marker_ll, 1, axis=0) * 1e-3)
    resid = (np.abs(b_n - shifted) > 1e-12) & rot
    assert resid.sum() <= 2 * n                       # pad seam only

    exp_n = np.cos(marker_ll * 1e-3)
    err = np.abs(b_n - exp_n)
    emax = np.percentile(err[err > 0], 98)
    _save_panels(
        [("E", [("east expected from co-located scalar", np.sin(marker_ll * 1e-3),
                 "RdBu_r", -1, 1),
                ("A east − expected (=0)", a_e - np.sin(marker_ll * 1e-3),
                 "PuOr", -emax, emax),
                ("B east − expected (=0)", b_e - np.sin(marker_ll * 1e-3),
                 "PuOr", -emax, emax)]),
         ("N", [("north expected from co-located scalar", exp_n,
                 "RdBu_r", -1, 1),
                ("A north − expected (=0)", a_n - exp_n, "PuOr", -emax, emax),
                ("B north − expected", b_n - exp_n, "PuOr", -emax, emax)])],
        OUTPUT_DIR / "registration_vs_scalars.png",
        "Registration vs scalar channels: A co-registered everywhere; "
        "B north carries the adjacent cell's value on the rotated half")


def test_B_is_valid_for_staggered_input():
    """B applied to RAW staggered input (its design case), then interpolated
    to centres on the rectangle, matches A.  The legacy bug was the ordering
    (interp before the vector stitch), not xmitgcm's stitch itself."""
    n = 90
    ds, face_id, grid = _synthetic_case(n)

    a_e, a_n = _path_a(ds.U, ds.V, ds, grid)              # production path

    # B as designed: vector-stitch the raw staggered pair ...
    b_e_stag, b_n_stag = _path_b(ds.U, ds.V, grid, interpolate=False)
    # ... then interp to cell centres on the rectangle (zonally periodic;
    # staggered points sit on the 'left' face -> average with next index)
    c_e = 0.5 * (b_e_stag + np.roll(b_e_stag, -1, axis=1))
    c_n = 0.5 * (b_n_stag + np.roll(b_n_stag, -1, axis=0))

    # compare away from rows where the rectangle has no valid neighbour
    # (southern edge pad/no-connection rows, northern cap boundary)
    interior = np.s_[2:-2, :]
    de = np.abs(c_e - a_e)[interior]
    dn = np.abs(c_n - a_n)[interior]
    frac_e = (de < 1e-12).mean()
    frac_n = (dn < 1e-12).mean()
    print(f"\nstaggered-B vs A agreement (interior): east {frac_e:.4%}, "
          f"north {frac_n:.4%}")
    assert frac_e > 0.99 and frac_n > 0.99

    # contrast: B on interpolated input (the legacy misuse)
    b_e_i, b_n_i = _path_b(ds.U, ds.V, grid)
    err = np.abs(np.stack([c_n - a_n, b_n_i - a_n]))
    emax = np.percentile(err[err > 0], 98)
    _save_panels(
        [("E", [("east A (reference)", a_e, "RdBu_r", None, None),
                ("B staggered input − A (≈0)", c_e - a_e, "PuOr", -emax, emax),
                ("B interpolated input − A (=0)", b_e_i - a_e, "PuOr", -emax, emax)]),
         ("N", [("north A (reference)", a_n, "RdBu_r", None, None),
                ("B staggered input − A (≈0)", c_n - a_n, "PuOr", -emax, emax),
                ("B interpolated input − A", b_n_i - a_n, "PuOr", -emax, emax)])],
        OUTPUT_DIR / "b_staggered_vs_interpolated_input.png",
        "B is valid for its design input (staggered) and wrong for "
        "interpolated input — the legacy bug was the ordering")


# ===========================================================================
# Real-data tests (opt-in)
# ===========================================================================

def _load_real_grid():
    import dbof.llc4320_ingestion.get_raw_data as get_raw_data
    import dbof.preprocessing.preproc_llc_core_data as preproc
    from dbof.global_dataset_creation.data_sources import OSN_ENDPOINT
    co = get_raw_data.get_remote_gridfile(OSN_ENDPOINT)
    return preproc.process_llc4320_grid(co)


_needs_grid = pytest.mark.skipif(
    not os.environ.get("DBOF_GRID_CHECK"),
    reason="needs OSN grid access; set DBOF_GRID_CHECK=1")

_needs_raw = pytest.mark.skipif(
    not os.environ.get("DBOF_RAW_CHECK"),
    reason="needs OSN data access; set DBOF_RAW_CHECK=1 "
           "(optional DBOF_DATE, within 2011-11-01..2012-07-15)")


@_needs_grid
def test_real_grid_cs_sn_convention():
    """LLC4320 CS/SN over ocean must equal the stitch convention on every
    non-cap face: (1, 0) on faces 0-5, (0, ±1) on faces 7-12."""
    ds_grid = _load_real_grid()
    cs, sn = ds_grid["CS"].values, ds_grid["SN"].values
    hfac = ds_grid["hFacC"].values
    if hfac.ndim == 4:
        hfac = hfac[:, 0]
    ocean = hfac > 0

    for f in range(N_FACES):
        if f == CAP_FACE or not ocean[f].any():
            continue
        cs_f, sn_f = cs[f][ocean[f]], sn[f][ocean[f]]
        exp_cs = 1.0 if f < CAP_FACE else 0.0
        exp_sn = 0.0 if f < CAP_FACE else np.sign(np.median(sn_f))
        dev = max(np.abs(cs_f - exp_cs).max(), np.abs(sn_f - exp_sn).max())
        print(f"face {f:2d}: CS≈{exp_cs:+.0f} SN≈{exp_sn:+.0f} "
              f"max dev {dev:.3e}")
        assert dev < 1e-6, f"face {f} deviates from the stitch convention"


@_needs_grid
def test_real_grid_cs_sn_face_plot():
    """Plot CS and SN per face in the LLC face layout (visual check of the
    rotation convention).  Writes cs_sn_faces.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_grid = _load_real_grid()
    # (row, col) face positions — same layout as plotting.llc_plotting
    layout = {0: (4, 0), 1: (3, 0), 2: (2, 0), 3: (4, 1), 4: (3, 1),
              5: (2, 1), 6: (1, 1), 7: (1, 2), 8: (1, 3), 9: (1, 4),
              10: (0, 2), 11: (0, 3), 12: (0, 4)}

    for name in ("CS", "SN"):
        var = ds_grid[name]
        fig, axes = plt.subplots(5, 5, figsize=(14, 14))
        for ax in axes.flatten():
            ax.axis("off")
        for f, (r, c) in layout.items():
            ax = axes[r, c]
            im = ax.imshow(np.asarray(var.isel(face=f).values), origin="lower",
                           cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"face {f}", fontsize=9)
        fig.colorbar(im, ax=axes, orientation="horizontal",
                     fraction=0.04, pad=0.02, label=name)
        fig.suptitle(f"LLC4320 {name} per face", fontsize=13)
        out = OUTPUT_DIR / f"cs_sn_faces_{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


@_needs_raw
def test_real_snapshot_A_vs_B():
    """Both paths on one raw OSN wind snapshot (oceTAUX/Y).  Expect: east
    identical; B's north = A's north rolled one pixel in latitude on the
    rotated half.  Writes real_snapshot_A_vs_B.png."""
    import dbof.llc4320_ingestion.get_raw_data as get_raw_data
    import dbof.preprocessing.preproc_llc_core_data as preproc
    from dbof.global_dataset_creation.data_sources import OSN_ENDPOINT
    from dbof.global_dataset_creation.grid_setup import set_up_grid
    from dbof.global_dataset_creation.iterations import LLC_FACES, osn_date_to_iteration
    from dbof.preprocessing.calculate_additional_fields import geographic_wind_stress

    date = os.environ.get("DBOF_DATE", "2012-01-15 12:00:00")
    ds_grid, _, grid = set_up_grid("OSN", None)
    it = osn_date_to_iteration(date)
    ds = get_raw_data.get_remote_llc_data(OSN_ENDPOINT, it, LLC_FACES)
    ds_merge = preproc.process_llc4320(ds, ds_grid)
    ds_merge = ds_merge.merge(
        get_raw_data.get_remote_llc_wind_data(OSN_ENDPOINT, it, LLC_FACES))

    a_e, a_n = _path_a(ds_merge.oceTAUX, ds_merge.oceTAUY, ds_merge, grid)
    b_e, b_n = _path_b(ds_merge.oceTAUX, ds_merge.oceTAUY, grid)
    a_e, a_n = np.asarray(a_e, np.float32), np.asarray(a_n, np.float32)
    b_e, b_n = np.asarray(b_e, np.float32), np.asarray(b_n, np.float32)

    rot = np.zeros(a_e.shape, bool)
    rot[:, _RECT_ROTATED_COL_START:] = True
    scale = np.nanstd(a_n)

    de = np.abs(a_e - b_e)
    dn = np.abs(a_n - b_n)
    dn_roll = np.abs(np.roll(a_n, 1, axis=0) - b_n)
    print(f"\neast  max|A-B| = {np.nanmax(de):.3e}")
    print(f"north max|A-B|           unrot / rot: "
          f"{np.nanmax(dn[~rot]):.3e} / {np.nanmax(dn[rot]):.3e}")
    print(f"north max|roll(A,1)-B|   rot        : {np.nanmax(dn_roll[rot]):.3e}")

    sub = np.s_[::20, ::20]
    dmax = np.nanmax(dn) or 1
    _save_panels(
        [("N", [("north A (rotated+scalar)", a_n[sub], "RdBu_r", -3*scale, 3*scale),
                ("north B (mate+vector)", b_n[sub], "RdBu_r", -3*scale, 3*scale),
                ("A-B", (a_n - b_n)[sub], "PuOr", -dmax, dmax),
                ("roll(A,1,lat)-B", (np.roll(a_n, 1, axis=0) - b_n)[sub],
                 "PuOr", -dmax, dmax)])],
        OUTPUT_DIR / "real_snapshot_A_vs_B.png",
        f"oceTAU {date}: B north = A north shifted 1 px (lat) on rotated half")

    assert np.nanmax(de) < 1e-4 * scale                    # east identical
    assert np.nanmax(dn[~rot]) < 1e-4 * scale              # north, unrot half
    assert np.nanmedian(dn_roll[rot]) < np.nanmedian(dn[rot])  # shift explains B


if __name__ == "__main__":
    test_difference_A_vs_B()
    test_correctness_A_vs_B()
    test_registration_vs_scalars()
    test_B_is_valid_for_staggered_input()
    print("Synthetic tests passed. Images in", OUTPUT_DIR)
