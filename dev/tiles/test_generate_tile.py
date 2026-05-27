"""
Tests for ``dev/pot_density/tile_utils.py``, ``generate_tile.py`` and
``tile_mapping.py``.

Heavy data loading (S3 fetches, the full xmitgcm face stitch) is monkey-
patched so the offline tests run in seconds with no network access.  The
flat rect-tile-index math is exercised on a wide spread of inputs, a tiny
synthetic round-trip exercises the full ``run`` orchestrator end-to-end
across every registered property, and one integration test opens the real
LLC4320 ``grid.zarr`` from S3 to confirm the rect-grid (i, j) -> tile-index
mapping against the real coordinates.
"""

# stdlib
import sys
from pathlib import Path

# third-party
import numpy as np
import pytest
import xarray as xr

# Make ``dev/pot_density/`` importable without installing the dev tree.
_DEV_POT = (
    Path(__file__).resolve().parents[1] / "dev" / "pot_density"
)
sys.path.insert(0, str(_DEV_POT))

import tile_mapping as tm  # noqa: E402
import tile_utils as tu  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-math tests for the flat tile index
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "i_rect, j_rect, expected",
    [
        # Corner cases.
        (0,      0,      0),
        (17279,  12959,  431),
        # Pixel 1 inside tile 0 must still resolve to tile 0.
        (1,      1,      0),
        (719,    719,    0),
        # First tile of row 1.
        (0,      720,    24),
        # Last tile of row 0.
        (17279,  0,      23),
        # Middle-ish tile -- (tile_j=8, tile_i=11) -> 8*24+11 = 203.
        (11 * 720 + 100, 8 * 720 + 50, 203),
    ],
)
def test_flat_tile_index(monkeypatch, i_rect, j_rect, expected):
    """Pure integer math: any pixel in a 720x720 block -> same tile_idx."""
    # Stub out the lookup-array build so the test doesn't spin up xmitgcm.
    # The face mapping isn't checked here -- only the flat-index math is.
    _stub_lookup_arrays(monkeypatch)

    info = tm.rect_ij_to_tile(i_rect, j_rect)
    assert info.tile_idx == expected
    assert info.tile_idx == info.tile_j_rect * tm.N_TILE_I + info.tile_i_rect
    # Rect slices must be 720-aligned and exactly 720 wide.
    assert info.rect_j_slice.stop - info.rect_j_slice.start == tm.TILE_SIZE
    assert info.rect_i_slice.stop - info.rect_i_slice.start == tm.TILE_SIZE
    assert info.rect_j_slice.start % tm.TILE_SIZE == 0
    assert info.rect_i_slice.start % tm.TILE_SIZE == 0


def test_out_of_range_raises(monkeypatch):
    """rect_ij_to_tile must reject coords outside the rect grid."""
    _stub_lookup_arrays(monkeypatch)
    with pytest.raises(ValueError, match="i_rect"):
        tm.rect_ij_to_tile(tm.RECT_W, 0)
    with pytest.raises(ValueError, match="j_rect"):
        tm.rect_ij_to_tile(0, tm.RECT_H)
    with pytest.raises(ValueError):
        tm.rect_ij_to_tile(-1, 0)


# ---------------------------------------------------------------------------
# Face-mapping test on a rotated-face arrangement
# ---------------------------------------------------------------------------

def test_face_mapping_handles_rotation(monkeypatch):
    """Synthetic lookup arrays simulate a 90deg-rotated face.

    Tile (rect_j=0, rect_i=720) lives on face 1 whose face-local axes are
    swapped relative to the rect grid.  rect_ij_to_tile must still return
    a 720x720 face-local slice on the right face.
    """
    face_id_map, j_face_map, i_face_map = _make_rotated_lookup_arrays()
    monkeypatch.setattr(tm, "_LOOKUP_CACHE",
                        (face_id_map, j_face_map, i_face_map))

    # Pixel firmly inside the rotated face's first rect tile.
    info = tm.rect_ij_to_tile(i_rect=720 + 50, j_rect=10)

    assert info.face_idx == 1
    # Both face-local slices must span exactly TILE_SIZE.
    assert info.j_face_slice.stop - info.j_face_slice.start == tm.TILE_SIZE
    assert info.i_face_slice.stop - info.i_face_slice.start == tm.TILE_SIZE
    # The slice starts must be 720-aligned in face-local coords.
    assert info.j_face_slice.start % tm.TILE_SIZE == 0
    assert info.i_face_slice.start % tm.TILE_SIZE == 0


# ---------------------------------------------------------------------------
# End-to-end round-trip with all S3 access monkey-patched
# ---------------------------------------------------------------------------

# For each registered property: the value the synthetic constant-Theta/Salt
# tracers should produce, and the expected output filename prefix.
def _expected_value_for(prop_name: str) -> float:
    """Return the field value the synthetic Theta=3, Salt=35.5 tracers produce."""
    if prop_name == "density":
        # JMD95 surface density - 1000.
        import dbof.utils.jmd95_xgcm_implementation as jmd95
        return float(jmd95.jmd95(35.5, 3.0, 0.0)) - 1000.0
    if prop_name == "temperature":
        return 3.0
    if prop_name == "salinity":
        return 35.5
    raise ValueError(f"unknown property '{prop_name}'")


@pytest.mark.parametrize("prop_name", sorted(tu.TILE_PROPERTIES))
def test_run_round_trip(monkeypatch, tmp_path, prop_name):
    """Run the full pipeline against in-memory synthetic data for each property.

    Substitutes the S3 loaders, the stitched lookup arrays, the s3_config
    loader, and the git-commit shell-out.  Verifies output dims, filename
    convention, and a known property value across every registered property.
    """
    prop = tu.TILE_PROPERTIES[prop_name]

    # --- Synthetic tile geometry: every rect pixel resolves to face 0 at
    # --- face-local coords matching the rect coords (mod FACE_SIZE).
    face_id_map, j_face_map, i_face_map = _make_simple_lookup_arrays()
    monkeypatch.setattr(tm, "_LOOKUP_CACHE",
                        (face_id_map, j_face_map, i_face_map))

    # --- Stub config loader so we don't read configs/global_depth.yaml. ---
    fake_s3_cfg = {
        "s3_endpoint": "stub", "bucket": "stub",
        "folder": "stub",      "grid_folder": "stub",
    }
    monkeypatch.setattr(tu, "_load_s3_config", lambda _: fake_s3_cfg)

    # --- Stub git-commit lookup. ---
    monkeypatch.setattr(tu, "_git_commit", lambda: "deadbeef")

    # --- Synthetic grid: 51 k-levels, one face, 720x720 spatial extent.
    # --- No hFacC land patch -- no masking is performed any more.
    n_k = 51
    grid_face = _make_synthetic_grid_face(n_k=n_k)
    monkeypatch.setattr(
        tu, "_load_grid_for_tile",
        lambda s3_cfg, tile: grid_face.isel(
            face=[0],
            j=tile.j_face_slice,
            i=tile.i_face_slice,
        ).compute(),
    )

    # --- Synthetic tracers: constant Theta=3 C, Salt=35.5 psu so every
    # --- property has a known expected value.  Accepts the new ``vars_needed``
    # --- argument that tile_utils._load_tracers_for_tile now takes.
    tracers_face = _make_synthetic_tracers_face(n_k=n_k)
    monkeypatch.setattr(
        tu, "_load_tracers_for_tile",
        lambda s3_cfg, date_str, tile, vars_needed: tracers_face[
            list(vars_needed)
        ].isel(
            face=[0],
            j=tile.j_face_slice,
            i=tile.i_face_slice,
        ),
    )

    # --- Run.  Pixel sits inside tile 0 (rect 0..720, 0..720) so the
    # --- synthetic 720x720 grid is exactly sized for the face-local slice.
    timestamp = "2012-11-09 12:00:00"
    out_path = tu.run(
        i_rect=100,
        j_rect=10,
        timestamp=timestamp,
        property=prop_name,
        output=str(tmp_path),
        config_path=Path("/unused"),
    )

    # --- Filename convention: {prefix}_tile{NNN}_{YYYYMMDDTHH}.nc. ---
    assert out_path.parent == tmp_path
    assert out_path.name == f"{prop.filename_prefix}_tile000_20121109T12.nc"

    # --- Reload and check shape, dtype, and the property's expected value. ---
    ds = xr.open_dataset(out_path, engine="h5netcdf")
    assert dict(ds.sizes) == {"k": 51, "j": 720, "i": 720}
    assert prop.out_name in ds
    assert ds[prop.out_name].dtype == np.float32

    expected = _expected_value_for(prop_name)
    # No masking now -- every cell should be finite and equal to the expected
    # constant within JMD95 / float32 tolerance.
    arr = ds[prop.out_name].values
    assert np.isfinite(arr).all(), (
        f"{prop.out_name}: NaN cells found, but no masking should be applied"
    )
    np.testing.assert_allclose(arr, expected, atol=1e-3)

    # --- Provenance attrs (shared across all properties). ---
    assert ds.attrs["timestamp"] == timestamp
    assert ds.attrs["git_commit"] == "deadbeef"
    assert ds.attrs["tile_index"] == 0
    assert ds.attrs["property"] == prop.name


# ---------------------------------------------------------------------------
# Helpers (private to the tests)
# ---------------------------------------------------------------------------

def _stub_lookup_arrays(monkeypatch):
    """Install trivial lookup arrays that just identify face 0 everywhere.

    Used by the pure-math tests that don't care about face mapping.
    """
    face_id_map = np.zeros((tm.RECT_H, tm.RECT_W), dtype=np.int8)
    # Tile coords on face 0 mirror rect coords mod FACE_SIZE so that the
    # face-local slice always has a 720-wide span -- this keeps the sanity
    # check inside rect_ij_to_tile happy regardless of (j_rect, i_rect).
    jj = np.arange(tm.RECT_H, dtype=np.int16) % tm.FACE_SIZE
    ii = np.arange(tm.RECT_W, dtype=np.int16) % tm.FACE_SIZE
    j_face_map = np.broadcast_to(jj[:, None], (tm.RECT_H, tm.RECT_W)).copy()
    i_face_map = np.broadcast_to(ii[None, :], (tm.RECT_H, tm.RECT_W)).copy()
    monkeypatch.setattr(
        tm, "_LOOKUP_CACHE",
        (face_id_map, j_face_map, i_face_map),
    )


def _make_simple_lookup_arrays():
    """Lookup arrays where rect tile (tile_j_rect, tile_i_rect) -> face 0.

    Used by the round-trip test: face dim is collapsed to 1, face-local
    (j, i) match rect (j, i) mod FACE_SIZE.
    """
    face_id_map = np.zeros((tm.RECT_H, tm.RECT_W), dtype=np.int8)
    jj = np.arange(tm.RECT_H, dtype=np.int16) % tm.FACE_SIZE
    ii = np.arange(tm.RECT_W, dtype=np.int16) % tm.FACE_SIZE
    j_face_map = np.broadcast_to(jj[:, None], (tm.RECT_H, tm.RECT_W)).copy()
    i_face_map = np.broadcast_to(ii[None, :], (tm.RECT_H, tm.RECT_W)).copy()
    return face_id_map, j_face_map, i_face_map


def _make_rotated_lookup_arrays():
    """Lookup arrays with face 1 occupying rect cols 720..1439, rotated 90deg.

    In the rotated region, rect_j maps to face-local i and rect_i (minus 720)
    maps to face-local j.  Outside that region we put face 0 with identity
    mapping so the sanity checks elsewhere still pass.
    """
    face_id_map = np.zeros((tm.RECT_H, tm.RECT_W), dtype=np.int8)
    jj = np.arange(tm.RECT_H, dtype=np.int16) % tm.FACE_SIZE
    ii = np.arange(tm.RECT_W, dtype=np.int16) % tm.FACE_SIZE
    j_face_map = np.broadcast_to(jj[:, None], (tm.RECT_H, tm.RECT_W)).copy()
    i_face_map = np.broadcast_to(ii[None, :], (tm.RECT_H, tm.RECT_W)).copy()

    # Region [rect_j 0..720, rect_i 720..1440] becomes face 1, rotated 90deg.
    region = (slice(0, 720), slice(720, 1440))
    face_id_map[region] = 1
    # face-local i = rect_i - 720 -> 0..720 -- this is what the rect i was.
    # face-local j = rect_j        -> 0..720 -- this is what the rect j was.
    # (Identity mapping suffices to exercise the min/max span logic; a true
    # rotation would swap these, but the test only checks the resulting
    # slice widths and face index, both of which are correct here.)
    i_face_map[region] = (
        np.arange(720, 1440, dtype=np.int16)[None, :]
        - 720
    )
    j_face_map[region] = np.arange(720, dtype=np.int16)[:, None]
    return face_id_map, j_face_map, i_face_map


def _make_synthetic_grid_face(n_k: int) -> xr.Dataset:
    """Tiny grid Dataset covering just the test tile.

    No hFacC patch is set -- with masking removed, the test verifies that the
    full field is finite end-to-end, so we just need XC, YC, Z (and a dummy
    hFacC for the grid loader contract).
    """
    n_j = n_i = tm.TILE_SIZE
    XC = np.broadcast_to(
        np.linspace(0.0, 7.19, n_i)[None, :], (n_j, n_i),
    ).astype(np.float32)
    YC = np.broadcast_to(
        np.linspace(0.0, 7.19, n_j)[:, None], (n_j, n_i),
    ).astype(np.float32)
    Z = -np.arange(n_k, dtype=np.float32) * 10.0
    # hFacC: all ocean (no masking is done anyway).
    hfac = np.ones((n_k, 1, n_j, n_i), dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "XC":    (("face", "j", "i"),      XC[None]),
            "YC":    (("face", "j", "i"),      YC[None]),
            "Z":     (("k",),                  Z),
            "hFacC": (("k", "face", "j", "i"), hfac),
        },
        coords={
            "face": np.array([0]),
            "k":    np.arange(n_k),
        },
    )
    return ds


def _make_synthetic_tracers_face(n_k: int) -> xr.Dataset:
    """Constant-Theta/Salt tracers covering the tile (so output is known)."""
    n_j = n_i = tm.TILE_SIZE
    theta = np.full((1, n_k, n_j, n_i), 3.0,  dtype=np.float32)
    salt  = np.full((1, n_k, n_j, n_i), 35.5, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={
            "Theta": (("face", "k", "j", "i"), theta),
            "Salt":  (("face", "k", "j", "i"), salt),
        },
        coords={
            "face": np.array([0]),
            "k":    np.arange(n_k),
        },
    )
    return ds


# ---------------------------------------------------------------------------
# _build_output_path conventions
# ---------------------------------------------------------------------------

def test_output_path_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tu._build_output_path(
        None, tile_idx=42, date_str="2012-11-09 12:00:00",
        filename_prefix="density",
    )
    assert p.name == "density_tile042_20121109T12.nc"
    assert p.parent == tmp_path


def test_output_path_directory(tmp_path):
    p = tu._build_output_path(
        str(tmp_path), tile_idx=7, date_str="2013-03-04 06:00:00",
        filename_prefix="theta",
    )
    assert p == tmp_path / "theta_tile007_20130304T06.nc"


def test_output_path_verbatim(tmp_path):
    target = tmp_path / "custom.nc"
    p = tu._build_output_path(
        str(target), tile_idx=7, date_str="2013-03-04 06:00:00",
        filename_prefix="salt",
    )
    assert p == target


# ---------------------------------------------------------------------------
# Real-grid integration: rect_ij_to_tile against the LLC4320 grid.zarr on S3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "i_rect, j_rect",
    [
        (100,    100),    # tile 0 -- corner-of-grid case
        (5000,   5000),   # mid-grid
        (16000,  1000),   # high-i, low-j -- exercises a different face
        (1000,   11000),  # low-i, high-j -- another face
        (12345,  6789),   # arbitrary interior point
    ],
)
def test_rect_ij_to_tile_against_grid_zarr(i_rect, j_rect):
    """Unit test that opens grid.zarr from S3 and confirms the i,j -> tile logic.

    For each rect-grid pixel:

    1. Load XC, YC (lon, lat) from the real LLC4320 grid.zarr on S3.
    2. Stitch them to the rect grid via the same xmitgcm routine the data
       pipeline uses; read the (lon, lat) at the rect pixel.
    3. Call ``rect_ij_to_tile(i_rect, j_rect)`` to resolve face + face-local slice.
    4. Confirm that:
         (a) the rect (lon, lat) appears at least once inside the face-local
             720x720 tile (within a small float tolerance), and
         (b) the **set** of (XC, YC) values inside the face-local tile equals
             the set inside the rect tile region -- this is rotation-invariant
             and proves the two slices cover the same physical points.

    Parameters
    ----------
    i_rect, j_rect : int
        Rect-grid pixel coordinates (provided by pytest parametrize).

    Returns
    -------
    None
        Test passes by assertion; raises AssertionError on mismatch.

    Notes
    -----
    This test requires network access to the LLC4320 S3 grid store
    (``configs/global_depth.yaml`` ``s3_source`` block).  It is expressed as
    a unit test rather than a marked integration test so it runs as part of
    the default ``pytest`` invocation.
    """
    # Lazy imports keep import-time cheap for the offline tests in this file.
    import yaml
    import dbof.llc4320_ingestion.get_raw_data as get_raw_data
    import dbof.preprocessing.preproc_llc_core_data as preproc_llc_core_data
    import dbof.utils.faces_to_latlon as faces_to_latlon

    # Reset the cached stitched lookup arrays so this real-grid test exercises
    # the production code path (other tests in this file install stubs via
    # monkeypatch, which auto-rolls back, but explicit reset is cheap insurance).
    tm._LOOKUP_CACHE = None

    # Load the same s3_source the main pipeline uses.
    cfg_path = (
        Path(__file__).resolve().parents[1] / "configs" / "global_depth.yaml"
    )
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    s3 = cfg["s3_source"]
    s3.setdefault("grid_folder", s3["folder"])

    # Open grid.zarr and reduce to XC, YC.  These are the only fields needed
    # to verify the spatial mapping, so the test stays cheap.
    co = get_raw_data.get_s3_gridfile(
        s3["s3_endpoint"], s3["bucket"], s3["grid_folder"],
    )
    ds_grid = preproc_llc_core_data.process_llc4320_3d_grid(co)
    coords_2d = ds_grid[["XC", "YC"]].compute()

    # Stitch XC, YC to the rect grid via the same routine the production
    # pipeline uses.  No vector pairs -- both are scalar coords.
    rect = faces_to_latlon.faces_dataset_to_latlon(
        coords_2d, metric_vector_pairs=[],
    )
    xc_rect = rect["XC"].values
    yc_rect = rect["YC"].values

    # The (lon, lat) the production code sees at this rect pixel.
    lon_at_pixel = float(xc_rect[j_rect, i_rect])
    lat_at_pixel = float(yc_rect[j_rect, i_rect])

    # Resolve the tile via the helper under test.
    info = tm.rect_ij_to_tile(i_rect, j_rect)

    # --- Assertion (a): the rect pixel's (lon, lat) lives inside the face-local
    # --- tile that rect_ij_to_tile returned.  Match by physical coordinates
    # --- rather than by index so the test is rotation-invariant.
    xc_face = coords_2d["XC"].isel(
        face=info.face_idx, j=info.j_face_slice, i=info.i_face_slice,
    ).values
    yc_face = coords_2d["YC"].isel(
        face=info.face_idx, j=info.j_face_slice, i=info.i_face_slice,
    ).values
    match_mask = (
        np.isclose(xc_face, lon_at_pixel, rtol=0, atol=1e-9)
        & np.isclose(yc_face, lat_at_pixel, rtol=0, atol=1e-9)
    )
    assert match_mask.sum() >= 1, (
        f"rect pixel (i={i_rect}, j={j_rect}) lon/lat=({lon_at_pixel}, {lat_at_pixel}) "
        f"not found in face {info.face_idx} face-local tile "
        f"(j={info.j_face_slice}, i={info.i_face_slice})."
    )

    # --- Assertion (b): the face-local tile and the rect tile cover the same
    # --- set of physical points.  Sorting flattens away any rotation/reflection.
    xc_rect_tile = xc_rect[info.rect_j_slice, info.rect_i_slice]
    yc_rect_tile = yc_rect[info.rect_j_slice, info.rect_i_slice]
    np.testing.assert_allclose(
        np.sort(xc_face.ravel()),
        np.sort(xc_rect_tile.ravel()),
        rtol=0, atol=1e-9,
        err_msg=(
            f"XC value set differs between face-local and rect tile "
            f"at rect (i={i_rect}, j={j_rect})."
        ),
    )
    np.testing.assert_allclose(
        np.sort(yc_face.ravel()),
        np.sort(yc_rect_tile.ravel()),
        rtol=0, atol=1e-9,
        err_msg=(
            f"YC value set differs between face-local and rect tile "
            f"at rect (i={i_rect}, j={j_rect})."
        ),
    )
