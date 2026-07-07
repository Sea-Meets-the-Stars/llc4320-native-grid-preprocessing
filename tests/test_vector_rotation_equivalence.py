"""
Comparison of the two vector-handling paths in the global pipeline.

Path A -- "computed channel" path (depth-pipeline native_fields U/V after
the rotation fix, and all jacobian/geographic derived fields):

    rotate_vector_to_geographic()        [interp to tracer + CS/SN rotation]
    -> faces_dataset_to_latlon()         [stitched as SCALARS, no mate attrs]

Path B -- "model channel" path (SURF native_fields U/V, oceTAUX/Y in both
pipelines):

    interp_staggered_to_tracer()         [interp to tracer, model basis]
    -> set_vector_pair_attrs()           [tag mate]
    -> faces_dataset_to_latlon()         [vector-aware face stitch]

VERIFIED RELATIONSHIP (this test asserts it, and writes an image):

* The EAST component is pixel-identical between the two paths.
* The NORTH component is identical on the un-rotated (left) half of the
  stitched map, but on the rotated half (source faces 7-12) Path B's north
  component is displaced by exactly ONE PIXEL along the latitude axis
  relative to Path A, and one row per rotated facet is zero-filled.

Why: xmitgcm's ``transform_u_to_v`` (used by the vector face stitch for
the u->north mapping on rotated faces) applies a one-pixel shift-and-pad
that re-registers *staggered* (i_g) components onto the rotated C-grid.
The production pipeline interpolates to tracer points BEFORE the stitch,
so for its input the shift is a misregistration.  Path A keeps each
cell's vector at its own cell and is the correct treatment for
tracer-point data.  (At LLC4320 resolution the artifact is a ~2 km
displacement of V/oceTAUY over the rotated half of the map.)

The rotation convention itself is not hard-coded here: the test MEASURES
the effective per-face rotation implemented by the vector stitch (unit
probe vectors) and adopts it as the synthetic grid's CS/SN.  A separate
opt-in test (``test_real_grid_cs_sn_convention``) checks that the real
LLC4320 AngleCS/AngleSN match this convention on every non-cap face; it
needs network access, so it is skipped unless ``DBOF_GRID_CHECK=1``.

Outputs: ``tests/output/vector_rotation_equivalence.png``

Run:
    pytest tests/test_vector_rotation_equivalence.py -v
    DBOF_GRID_CHECK=1 pytest tests/test_vector_rotation_equivalence.py -v


    DBOF_PRODUCT_CHECK=1 \
    DBOF_BUCKET=dbof \
    DBOF_FOLDER=surface_fields \
    DBOF_RUN_ID=vtest1 \
    DBOF_DATE_PREFIX=20120501_120000 \
    DBOF_DATASET=native_fields.zarr \
    pytest tests/test_vector_rotation_equivalence.py::test_real_product_zero_lines -s

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


# ---------------------------------------------------------------------------
# Synthetic LLC dataset
# ---------------------------------------------------------------------------

def _make_synthetic_llc_dataset(n=90):
    """Synthetic 13-face LLC dataset with staggered U/V and comodo attrs."""
    i = xr.DataArray(np.arange(n), dims="i",
                     attrs={"axis": "X"})
    i_g = xr.DataArray(np.arange(n), dims="i_g",
                       attrs={"axis": "X", "c_grid_axis_shift": -0.5})
    j = xr.DataArray(np.arange(n), dims="j",
                     attrs={"axis": "Y"})
    j_g = xr.DataArray(np.arange(n), dims="j_g",
                       attrs={"axis": "Y", "c_grid_axis_shift": -0.5})
    face = xr.DataArray(np.arange(N_FACES), dims="face")

    # Smooth, face-dependent fields so any component swap, sign error, or
    # registration shift is visible and asymmetric.
    ff = np.arange(N_FACES)[:, None, None]
    jj = np.arange(n)[None, :, None]
    ii = np.arange(n)[None, None, :]

    u_data = (np.sin(2 * np.pi * ii / n) + 0.5 * np.cos(2 * np.pi * jj / n)
              + 0.10 * ff + 2.0)
    v_data = (np.cos(4 * np.pi * ii / n) - 0.7 * np.sin(2 * np.pi * jj / n)
              - 0.05 * ff - 3.0)

    ds = xr.Dataset(
        {
            "U": xr.DataArray(u_data, dims=("face", "j", "i_g")),
            "V": xr.DataArray(v_data, dims=("face", "j_g", "i")),
        },
        coords={"face": face, "i": i, "i_g": i_g, "j": j, "j_g": j_g},
    )
    return ds


# ---------------------------------------------------------------------------
# Stitch helpers
# ---------------------------------------------------------------------------

def _stitch(ds_faces):
    """Stitch a face dataset to the lat-lon rectangle (repo wrapper)."""
    return faces_dataset_to_latlon(ds_faces, metric_vector_pairs=[])


def _face_id_map(ds):
    """Stitch a per-face constant scalar to identify output pixel origins."""
    n_j, n_i = ds.sizes["j"], ds.sizes["i"]
    data = np.broadcast_to(
        np.arange(N_FACES, dtype=float)[:, None, None],
        (N_FACES, n_j, n_i)).copy()
    face_id = xr.DataArray(
        data, dims=("face", "j", "i"),
        coords={"face": ds.face, "j": ds.j, "i": ds.i}, name="face_id")
    ds_id = xr.Dataset({"face_id": face_id})
    return _stitch(ds_id.chunk({"face": 1}))["face_id"].values


def _measure_stitch_rotation(ds, face_id):
    """Measure the per-face (CS, SN) equivalent of the vector face stitch.

    Stitches the probe vector (u, v) = (1, 0) through the mate-paired
    vector path: at each output pixel the stitch yields
    (E, N) = (CS_face, SN_face) for the pixel's source face.  The median
    over each face's pixels recovers the coefficients (median, not mean,
    because the stitch zero-pads one row per rotated facet -- the same
    seam artifact the main test documents).  A second probe
    (0, 1) -> (-SN, CS) is used as a consistency check.

    Returns
    -------
    cs, sn : np.ndarray, shape (13,)
        Effective rotation coefficients per face.  Cap face entries
        (dropped by the stitch) are set to identity (unused).
    """
    def _probe(u_val, v_val):
        shape = (N_FACES, ds.sizes["j"], ds.sizes["i"])
        pu = xr.DataArray(np.full(shape, u_val), dims=("face", "j", "i"),
                          coords={"face": ds.face, "j": ds.j, "i": ds.i})
        pv = xr.DataArray(np.full(shape, v_val), dims=("face", "j", "i"),
                          coords={"face": ds.face, "j": ds.j, "i": ds.i})
        dsp = xr.Dataset({"pu": pu, "pv": pv})
        set_vector_pair_attrs(dsp, vector_pairs=[("pu", "pv")])
        out = _stitch(dsp.chunk({"face": 1}))
        return out["pu"].values, out["pv"].values

    e1, n1 = _probe(1.0, 0.0)   # (E, N) = ( CS, SN)
    e2, n2 = _probe(0.0, 1.0)   # (E, N) = (-SN, CS)

    cs = np.full(N_FACES, np.nan)
    sn = np.full(N_FACES, np.nan)
    for f in range(N_FACES):
        sel = face_id == f
        if not sel.any():          # cap face -- not in the rectangle
            continue
        cs_f = np.median(e1[sel])
        sn_f = np.median(n1[sel])
        # Consistency between the two probes (rotation-matrix structure).
        assert np.isclose(np.median(e2[sel]), -sn_f, atol=1e-12), \
            f"face {f}: probe (0,1) E-component inconsistent with -SN"
        assert np.isclose(np.median(n2[sel]), cs_f, atol=1e-12), \
            f"face {f}: probe (0,1) N-component inconsistent with CS"
        cs[f], sn[f] = cs_f, sn_f

    cs[CAP_FACE], sn[CAP_FACE] = 1.0, 0.0
    return cs, sn


# ---------------------------------------------------------------------------
# The two production paths
# ---------------------------------------------------------------------------

def _path_a_rotate_then_scalar_stitch(ds, grid):
    """Computed-channel path: rotate_vector_to_geographic -> scalar stitch."""
    u_east, v_north = ng.rotate_vector_to_geographic(
        ds["U"], ds["V"], ds, grid)
    u_east.attrs, v_north.attrs = {}, {}          # no mate attrs -> scalars
    ds_a = xr.Dataset({"U": u_east, "V": v_north})
    out = _stitch(ds_a.chunk({"face": 1}))
    return out["U"].values, out["V"].values


def _path_b_interp_mate_vector_stitch(ds, grid):
    """Model-channel path: interp_staggered_to_tracer + mate -> vector stitch."""
    fields = {"U": ds["U"], "V": ds["V"]}
    interp_staggered_to_tracer(fields, grid)
    ds_b = xr.Dataset(fields)
    set_vector_pair_attrs(ds_b)                   # mate -> vector stitch
    out = _stitch(ds_b.chunk({"face": 1}))
    return out["U"].values, out["V"].values


# ---------------------------------------------------------------------------
# Image output
# ---------------------------------------------------------------------------

def _save_comparison_image(a_u, a_v, b_u, b_v, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    rows = [("U (east)", a_u, b_u), ("V (north)", a_v, b_v)]
    for r, (label, a, b) in enumerate(rows):
        diff = a - b
        vmax = np.nanmax(np.abs(np.concatenate([a.ravel(), b.ravel()])))
        im0 = axes[r, 0].imshow(a, origin="lower", cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
        axes[r, 0].set_title(f"{label} — Path A\nrotate_vector_to_geographic"
                             " + scalar stitch")
        im1 = axes[r, 1].imshow(b, origin="lower", cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
        axes[r, 1].set_title(f"{label} — Path B\ninterp + mate attrs"
                             " + vector stitch")
        dmax = np.nanmax(np.abs(diff))
        im2 = axes[r, 2].imshow(diff, origin="lower", cmap="PuOr",
                                vmin=-max(dmax, 1e-16),
                                vmax=max(dmax, 1e-16))
        axes[r, 2].set_title(f"{label} — difference (A − B)\n"
                             f"max |diff| = {dmax:.2e}")
        for ax, im in zip(axes[r], (im0, im1, im2)):
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "Vector-path comparison, synthetic 13-face LLC grid (cap dropped).\n"
        "East: identical.  North: Path B displaced 1 pixel in latitude on "
        "the rotated half (staggered-grid shift applied to tracer-point "
        "data) + zero-filled seam rows.", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vector_paths_east_identical_north_shifted():
    n = 90
    ds = _make_synthetic_llc_dataset(n=n)
    face_id = _face_id_map(ds)

    # Measure the rotation convention the vector stitch actually implements
    # and adopt it as the synthetic grid's CS/SN.
    cs, sn = _measure_stitch_rotation(ds, face_id)

    # Document/verify the convention: identity on lat-lon faces, a pure
    # 90-degree swap on the rotated faces.
    for f in range(N_FACES):
        if f == CAP_FACE:
            continue
        assert np.isclose(cs[f] ** 2 + sn[f] ** 2, 1.0, atol=1e-12)
        if f < CAP_FACE:
            assert np.isclose(cs[f], 1.0) and np.isclose(sn[f], 0.0), \
                f"lat-lon face {f}: expected identity, got CS={cs[f]}, SN={sn[f]}"
        else:
            assert np.isclose(cs[f], 0.0) and np.isclose(abs(sn[f]), 1.0), \
                f"rotated face {f}: expected 90° swap, got CS={cs[f]}, SN={sn[f]}"

    shape = (N_FACES, n, n)
    ds["CS"] = xr.DataArray(
        np.broadcast_to(cs[:, None, None], shape).copy(),
        dims=("face", "j", "i"),
        coords={"face": ds.face, "j": ds.j, "i": ds.i})
    ds["SN"] = xr.DataArray(
        np.broadcast_to(sn[:, None, None], shape).copy(),
        dims=("face", "j", "i"),
        coords={"face": ds.face, "j": ds.j, "i": ds.i})

    grid = set_xgcm_grid(ds, use_connections=True)

    a_u, a_v = _path_a_rotate_then_scalar_stitch(ds, grid)
    b_u, b_v = _path_b_interp_mate_vector_stitch(ds, grid)

    _save_comparison_image(
        a_u, a_v, b_u, b_v,
        OUTPUT_DIR / "vector_rotation_equivalence.png")

    rotated_half = np.isin(face_id, ROTATED_FACES)

    # 1. East component: the two paths are pixel-identical everywhere.
    np.testing.assert_allclose(
        a_u, b_u, rtol=0, atol=1e-12,
        err_msg="EAST component differs between the two paths")

    # 2. North component: identical on the un-rotated half of the map.
    dv = a_v - b_v
    assert not (np.abs(dv) > 1e-12)[~rotated_half].any(), \
        "NORTH component differs on the un-rotated half"

    # 3. North component, rotated half: Path B equals Path A displaced by
    #    exactly one pixel along the latitude axis -- the staggered-grid
    #    shift that transform_u_to_v applies -- except for the zero-filled
    #    pad rows (one per rotated facet).
    a_v_shifted = np.roll(a_v, 1, axis=0)
    mismatch = (np.abs(a_v_shifted - b_v) > 1e-12) & rotated_half
    n_seam = mismatch.sum()
    assert n_seam <= 2 * n, (
        f"rotated half: {n_seam} pixels differ even after the 1-pixel "
        f"shift (expected only the {2 * n} zero-pad seam pixels)")
    assert np.allclose(b_v[mismatch], 0.0, atol=1e-15), \
        "seam pixels in Path B north component are not zero-filled"

    # The displacement is real and O(field): confirm the raw difference on
    # the rotated half is NOT small (i.e. the shift matters).
    assert np.abs(dv)[rotated_half].max() > 0.01, \
        "expected a visible 1-pixel displacement signal on the rotated half"

    print(
        f"\nEast: identical.  North: identical on un-rotated half; "
        f"1-pixel latitude displacement on rotated half "
        f"(max |A-B| = {np.abs(dv)[rotated_half].max():.3f}), "
        f"{n_seam} zero-pad seam pixels."
    )


@pytest.mark.skipif(
    not os.environ.get("DBOF_GRID_CHECK"),
    reason="needs network access to the OSN grid endpoint; "
           "set DBOF_GRID_CHECK=1 to run",
)
def test_real_grid_cs_sn_convention():
    """Real LLC4320 AngleCS/AngleSN must match the stitch convention off-cap.

    This is the empirical half of the equivalence argument: on every
    non-cap face the true rotation coefficients must be (to numerical
    precision) the identity (faces 0-5) or the pure 90-degree swap
    (faces 7-12) that the vector face stitch applies.  Ocean-only cells
    are checked (CS/SN can be arbitrary over land in some grid files).
    """
    import dbof.llc4320_ingestion.get_raw_data as get_raw_data
    import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
    from dbof.global_dataset_creation.data_sources import OSN_ENDPOINT

    co = get_raw_data.get_remote_gridfile(OSN_ENDPOINT)
    ds_grid = preproc_llc_core_data.process_llc4320_grid(co)

    cs = ds_grid["CS"].values
    sn = ds_grid["SN"].values
    hfac = ds_grid["hFacC"].values
    if hfac.ndim == 4:                 # (face, k, j, i) -> surface
        hfac = hfac[:, 0]
    ocean = hfac > 0

    for f in range(N_FACES):
        if f == CAP_FACE:
            continue
        sel = ocean[f]
        if not sel.any():
            continue
        cs_f, sn_f = cs[f][sel], sn[f][sel]
        if f < CAP_FACE:
            exp_cs, exp_sn = 1.0, 0.0
        else:
            exp_cs, exp_sn = 0.0, np.sign(np.median(sn_f)) * 1.0
        dev = max(np.abs(cs_f - exp_cs).max(), np.abs(sn_f - exp_sn).max())
        print(f"face {f:2d}: CS≈{exp_cs:+.0f}, SN≈{exp_sn:+.0f}, "
              f"max deviation = {dev:.3e}")
        assert dev < 1e-6, (
            f"face {f}: real AngleCS/AngleSN deviate from the stitch "
            f"convention by up to {dev:.3e} — raw model-channel vectors "
            f"(vector stitch, no CS/SN) would be biased on this face."
        )


# ---------------------------------------------------------------------------
# Ground-truth registration test
# ---------------------------------------------------------------------------

def test_ground_truth_registration():
    """Prove which path is correct, not just that they differ.

    The registration of the output product is DEFINED by the scalar
    stitch: the land mask, tracers, and every computed field are placed
    on the rectangle by it, so a vector component is correct iff cell
    (f, j, i)'s value lands on the same output pixel as that cell's
    scalars.

    Construction: a known geographic vector (E, N) is defined at every
    CELL CENTER, converted to the model basis per cell with the measured
    CS/SN, and pushed through both paths.  Reference = the truth arrays
    relocated by the scalar stitch.  ``interpolate=False`` is used so the
    (identical, shared) interp step is excluded and the stitch question
    is isolated.

    Expected: Path A == truth exactly, both components.  Path B == truth
    for east, but misregistered at every pixel of the rotated half for
    north.

    Writes ``tests/output/vector_stitch_ground_truth.png``.
    """
    n = 90
    ds = _make_synthetic_llc_dataset(n=n)
    face_id = _face_id_map(ds)
    cs, sn = _measure_stitch_rotation(ds, face_id)

    shape = (N_FACES, n, n)
    cs_a = np.broadcast_to(cs[:, None, None], shape)
    sn_a = np.broadcast_to(sn[:, None, None], shape)
    coords = {"face": ds.face, "j": ds.j, "i": ds.i}
    ds["CS"] = xr.DataArray(cs_a.copy(), dims=("face", "j", "i"),
                            coords=coords)
    ds["SN"] = xr.DataArray(sn_a.copy(), dims=("face", "j", "i"),
                            coords=coords)

    def _da(x):
        return xr.DataArray(x, dims=("face", "j", "i"), coords=coords)

    # Known geographic vector at every cell center.
    ff = np.arange(N_FACES)[:, None, None]
    jj = np.arange(n)[None, :, None]
    ii = np.arange(n)[None, None, :]
    e_true = np.sin(2 * np.pi * ii / n) + 0.3 * jj / n + 0.2 * ff + 1.0
    n_true = np.cos(2 * np.pi * jj / n) - 0.4 * ii / n - 0.1 * ff - 2.0

    # Same vector expressed in the model basis per cell (inverse rotation).
    u_model = e_true * cs_a + n_true * sn_a
    v_model = -e_true * sn_a + n_true * cs_a

    # Reference registration: truth relocated exactly like every scalar
    # channel (land mask, Theta, computed fields) in the product.
    ref = _stitch(xr.Dataset(
        {"E": _da(e_true), "N": _da(n_true)}).chunk({"face": 1}))
    e_ref, n_ref = ref["E"].values, ref["N"].values

    # Path A on cell-centred input (interpolate=False isolates the stitch).
    e_a, n_a = ng.rotate_vector_to_geographic(
        _da(u_model), _da(v_model), ds, None, interpolate=False)
    e_a.attrs, n_a.attrs = {}, {}
    out_a = _stitch(xr.Dataset({"E": e_a, "N": n_a}).chunk({"face": 1}))

    # Path B on the same cell-centred model-basis input.
    ds_b = xr.Dataset({"U": _da(u_model), "V": _da(v_model)})
    set_vector_pair_attrs(ds_b)
    out_b = _stitch(ds_b.chunk({"face": 1}))

    err_a_e = out_a["E"].values - e_ref
    err_a_n = out_a["N"].values - n_ref
    err_b_e = out_b["U"].values - e_ref
    err_b_n = out_b["V"].values - n_ref

    # ---- image ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for r, (label, truth, ea, eb) in enumerate([
            ("E (east)", e_ref, err_a_e, err_b_e),
            ("N (north)", n_ref, err_a_n, err_b_n)]):
        vmax = np.nanmax(np.abs(truth))
        im0 = axes[r, 0].imshow(truth, origin="lower", cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
        axes[r, 0].set_title(f"{label} — ground truth\n(scalar-stitch "
                             "registration = product registration)")
        # Scale error panels to the 98th percentile of the *nonzero* error
        # so the pervasive 1-pixel displacement error is visible (the pad
        # seam is ~50x larger and would otherwise swamp the color scale).
        nz = np.abs(np.stack([ea, eb]))
        nz = nz[nz > 0]
        emax = np.percentile(nz, 98) if nz.size else 1e-16
        im1 = axes[r, 1].imshow(ea, origin="lower", cmap="PuOr",
                                vmin=-emax, vmax=emax)
        axes[r, 1].set_title(f"{label} — Path A error\n"
                             f"max |err| = {np.abs(ea).max():.2e}")
        im2 = axes[r, 2].imshow(eb, origin="lower", cmap="PuOr",
                                vmin=-emax, vmax=emax)
        axes[r, 2].set_title(f"{label} — Path B error "
                             f"(color clipped at {emax:.2g})\n"
                             f"max |err| = {np.abs(eb).max():.2e}")
        for ax, im in zip(axes[r], (im0, im1, im2)):
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "Ground-truth registration: known geographic vector at cell centres, "
        "expressed in model basis, through both paths.\n"
        "Path A (CS/SN rotate + scalar stitch) reproduces truth exactly; "
        "Path B (mate + vector stitch) misregisters NORTH on the rotated "
        "half.", fontsize=12)
    out_png = OUTPUT_DIR / "vector_stitch_ground_truth.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---- assertions ----
    rotated_half = np.isin(face_id, ROTATED_FACES)

    assert np.abs(err_a_e).max() == 0.0, "Path A east deviates from truth"
    assert np.abs(err_a_n).max() == 0.0, "Path A north deviates from truth"
    assert np.abs(err_b_e).max() == 0.0, "Path B east deviates from truth"

    bad_b_n = np.abs(err_b_n) > 1e-12
    assert not bad_b_n[~rotated_half].any(), \
        "Path B north wrong on the UN-rotated half (unexpected)"
    assert bad_b_n[rotated_half].all(), (
        "expected Path B north to be misregistered at every rotated-half "
        "pixel for this smooth truth field")

    print(f"\nPath A == truth exactly (both components).  Path B: east "
          f"exact, north wrong at {bad_b_n.sum()}/{rotated_half.sum()} "
          f"rotated-half pixels (max err {np.abs(err_b_n).max():.3f}).")


# ---------------------------------------------------------------------------
# Real-product zero-line scanner (opt-in: needs S3 access to a product store)
# ---------------------------------------------------------------------------

#: Stitched-rectangle geometry (LLC4320).
_RECT_H, _RECT_W = 12960, 17280
_FACET_W = 4320
#: Column ranges of the two rotated facets in the stitched rectangle.
_ROTATED_COL_START = 2 * _FACET_W          # columns >= 8640 come from faces 7-12


def _scan_zero_lines(field, name, min_frac=0.5, min_count=500):
    """Find rows/columns dominated by EXACT zeros over ocean (non-NaN) pixels.

    Returns (rows, cols): lists of (index, n_zero, n_ocean, frac) tuples.
    """
    finite = ~np.isnan(field)
    zero = finite & (field == 0.0)

    def _lines(axis):
        n_zero = zero.sum(axis=axis)
        n_ocean = finite.sum(axis=axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n_ocean > 0, n_zero / n_ocean, 0.0)
        hits = np.where((frac >= min_frac) & (n_zero >= min_count))[0]
        return [(int(k), int(n_zero[k]), int(n_ocean[k]), float(frac[k]))
                for k in hits]

    cols = _lines(axis=0)   # per-column stats -> vertical (longitude) lines
    rows = _lines(axis=1)   # per-row stats    -> horizontal (latitude) lines
    if rows or cols:
        print(f"\n{name}: exact-zero lines found")
        for k, nz, no, fr in rows:
            print(f"  ROW  {k:>6d}  zeros={nz:>7d}/{no:>7d} ({fr:.1%})")
        for k, nz, no, fr in cols:
            side = "ROTATED half" if k >= _ROTATED_COL_START else "un-rotated half"
            seam = "  <-- facet seam" if k % _FACET_W in (0, _FACET_W - 1) else ""
            print(f"  COL  {k:>6d}  zeros={nz:>7d}/{no:>7d} ({fr:.1%})"
                  f"  [{side}]{seam}")
    else:
        print(f"\n{name}: no exact-zero lines detected")
    return rows, cols


@pytest.mark.skipif(
    not os.environ.get("DBOF_PRODUCT_CHECK"),
    reason="needs S3 access to a generated product store; set "
           "DBOF_PRODUCT_CHECK=1 (plus DBOF_RUN_ID / DBOF_DATE_PREFIX / "
           "optionally DBOF_DATASET, DBOF_BUCKET, DBOF_FOLDER, "
           "DBOF_S3_ENDPOINT, DBOF_CHANNELS) to run",
)
def test_real_product_zero_lines():
    """Scan a real generated product for the Path-B artifacts.

    Looks for lines of EXACT zeros in the stitched vector channels of an
    existing store (default ``native_fields.zarr``; set
    ``DBOF_DATASET=surface_wind.zarr`` to scan wind stress).

    Expected Path-B signatures:

    * the zero-pad seam: a horizontal (latitude) line confined to the
      rotated-half columns (>= 8640), in the NORTH component (V /
      oceTAUY) only;
    * NO such line in the east component from the stitch itself --
      zero lines in U / oceTAUX (especially vertical ones, or ones on
      the rotated half) point to artifacts already present in the RAW
      model-y data, which the 90-degree swap deposits into the plotted
      east component on that half.

    Writes ``tests/output/product_zero_lines_<dataset>.png``.
    """
    from dbof.io.filesystems import create_s3_filesystems
    from dbof.global_dataset_creation.zarr_dataset_global import (
        GlobalZarrDatasetReader,
    )

    endpoint = os.environ.get("DBOF_S3_ENDPOINT",
                              "https://s3-west.nrp-nautilus.io")
    bucket = os.environ.get("DBOF_BUCKET", "dbof")
    folder = os.environ.get("DBOF_FOLDER")
    run_id = os.environ.get("DBOF_RUN_ID")
    date_prefix = os.environ.get("DBOF_DATE_PREFIX")
    dataset = os.environ.get("DBOF_DATASET", "native_fields.zarr")
    assert folder and run_id, \
        "set DBOF_FOLDER and DBOF_RUN_ID (and usually DBOF_DATE_PREFIX)"

    default_channels = ("oceTAUX,oceTAUY" if "wind" in dataset else "U,V")
    channels = os.environ.get("DBOF_CHANNELS", default_channels).split(",")

    fs, _ = create_s3_filesystems(endpoint)
    reader = GlobalZarrDatasetReader(
        bucket, folder, run_id, dataset, fs, date_prefix=date_prefix)
    print(f"\nstore channels: {reader.channel_names}  "
          f"iteration: {reader.iteration}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    fig, axes = plt.subplots(len(channels), 3, figsize=(20, 5 * len(channels)),
                             constrained_layout=True, squeeze=False)
    for r, ch in enumerate(channels):
        field = reader.get_channel_snapshot(ch)     # (H, W) float32
        rows, cols = _scan_zero_lines(field, ch)
        results[ch] = (rows, cols)

        finite = ~np.isnan(field)
        zero = finite & (field == 0.0)
        col_frac = np.where(finite.sum(0) > 0,
                            zero.sum(0) / np.maximum(finite.sum(0), 1), 0)
        row_frac = np.where(finite.sum(1) > 0,
                            zero.sum(1) / np.maximum(finite.sum(1), 1), 0)

        sub = field[::20, ::20]
        vmax = np.nanpercentile(np.abs(sub), 99)
        axes[r, 0].imshow(sub, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        for k, *_ in rows:
            axes[r, 0].axhline(k / 20, color="lime", lw=1.0)
        for k, *_ in cols:
            axes[r, 0].axvline(k / 20, color="magenta", lw=1.0)
        axes[r, 0].axvline(_ROTATED_COL_START / 20, color="k", ls="--", lw=0.8)
        axes[r, 0].set_title(
            f"{ch} (20x downsampled) — zero-lines marked "
            f"(rows: green, cols: magenta; dashed = start of rotated half)")

        axes[r, 1].plot(col_frac, lw=0.5)
        axes[r, 1].axvline(_ROTATED_COL_START, color="k", ls="--", lw=0.8)
        axes[r, 1].set_title(f"{ch} — fraction of ocean pixels EXACTLY 0, "
                             "per column (longitude)")
        axes[r, 1].set_ylim(0, 1.05)

        axes[r, 2].plot(row_frac, lw=0.5)
        axes[r, 2].set_title(f"{ch} — fraction of ocean pixels EXACTLY 0, "
                             "per row (latitude)")
        axes[r, 2].set_ylim(0, 1.05)

    out_png = OUTPUT_DIR / f"product_zero_lines_{dataset.replace('.zarr','')}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nimage written to {out_png}")

    # Informational, not a hard failure: report whether the Path-B pad-seam
    # signature (horizontal zero line, rotated half, north component) exists.
    north = channels[-1]
    rows_n, _ = results[north]
    if rows_n:
        print(f"pad-seam candidate row(s) in {north}: "
              f"{[k for k, *_ in rows_n]}")


if __name__ == "__main__":
    test_vector_paths_east_identical_north_shifted()
    test_ground_truth_registration()
    print("Tests passed. Images written to", OUTPUT_DIR)
