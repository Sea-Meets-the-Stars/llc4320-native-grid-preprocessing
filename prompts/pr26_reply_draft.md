# Draft reply to the PR #26 review (post after pushing the branch)

Thanks for the careful review — every point is addressed below. The two
notebooks were substantially rewritten, and several new unit tests were
added where a comment exposed a genuine gap in coverage.

## General

- **PR title** — renamed as suggested.
- **Where on the grid cell operations go** — both notebooks now open
  with an annotated Arakawa C-grid schematic (new shared helper
  `tests/cgrid_schematic.py`) using the ECCO/xmitgcm names: tracer
  point `(j, i)`, U point `(j, i_g)` on the west face (with `U`,
  `dxC`), V point `(j_g, i)` on the south face (with `V`, `dyC`), and
  the exact path diff → divide-by-metric → interp-to-centre → rotate.
  The text states explicitly that **all gradient outputs land on
  tracer/cell-centre points**, and that this is a choice downstream
  calculations must be aware of. (Attaching machine-readable
  location attributes to the production outputs was deemed out of
  scope for this PR.)

## `native_gradient_tracer_tests.ipynb`

- **Gradient in lat, lon or both?** Both. The function returns the
  zonal (λ, eastward) and meridional (φ, northward) components; the
  notebook now plots the two components separately (not just the
  magnitude), and the ±5° Eastern Pacific panel makes the equatorial
  symmetry of the meridional component visible.
- **ECCO naming / where things end up** — adopted throughout (see the
  schematic above); every output is at the tracer point.
- **Definitions** — `p`, `q` were renamed `slope_x`, `slope_y`
  (prescribed synthetic tracer slopes, tracer units per metre — not
  pressure); "geographic axes" is defined as local east/north.
- **λ/φ convention** — the code (from the ECCO v4 tutorial) uses
  λ = longitude, φ = latitude, the standard geographic convention;
  the notebook now defines both symbols at first use and in every
  axis label ("zonal (λ, eastward)" etc.).
- **Constant tracer** — yes, the whole domain filled with one number;
  now stated explicitly.
- **Synthetic 13-face grid** — now described in full (LLC topology +
  `face_connections`, uniform metrics, constant rotation, why the
  interior is analytic). It was isotropic; per your anisotropy point,
  new tests (`test_*_anisotropic_grid_metrics` in both test modules)
  use `dyC = 2.4 dxC`, which would expose a metric swap as a 2.4×
  error. The notebook shows this (Section 4).
- **Edges look off** — explained: each synthetic face carries its own
  linear field, so the cross-face halo stitches inconsistent
  neighbours at every seam; the edge ring is wrong *by construction*
  and excluded by `EDGE_MARGIN` (marked with a dashed box in the
  figures).
- **Downsampled / continent-heavy face** — replaced by
  **full-native-resolution (1/48°)** regional images of the three
  regions from issue #24 (W. North Atlantic 0–60°N, Atlantic sector
  of the Southern Ocean, ±5° Eastern Pacific), each showing the
  signed zonal and meridional components plus the magnitude.
- **"Front" wording** — replaced by "SST gradients".
- **Theta** — yes, MITgcm `Theta` = potential temperature; the store
  metadata (`long_name: Potential Temperature`, `units: degC`) is now
  printed from the data itself, and the notebook notes why the
  distinction matters at depth.
- **"Geographic vs analytic" stats** — that label was wrong; the
  comparison is geographic-frame vs **model-frame** gradient magnitude
  (rotation invariance). Text and labels fixed.
- **Are the faces doing what they should?** — new test
  (`test_tracer_real_grid_face_consistency_analytic_field`): tracers
  built from the grid's own XC/YC with closed-form geographic
  gradients (`sin φ` and `sin λ cos φ`) are checked **per face**; a
  per-face staggering/sign/component error would appear as an O(1)
  normalised error on that face. Observed per-face medians are ~1e-4
  (the float32 lon/lat noise floor) with thresholds 1e-3 (median) and
  5e-3 (95th pct). The notebook shows per-face statistics and
  computed-vs-analytic maps on the Arctic cap face (the most rotated
  one).

## `vertical_helpers_tests.ipynb`

- **Why flip the sign of depth?** The positive-down convention is
  baked into the downstream MLD/vertical code, so it stays — but the
  flip is no longer silent: `_get_depth_coord` now emits a
  `UserWarning` whenever it inverts the native negative-up `Z` (the
  "big flag" for users of other ECCO packages), the notebook opens
  with a warning banner, and a new test asserts the warning fires
  (and does not fire when no flip happens).
- **Top description** — rewritten; a definitions table gives `Z`,
  `Zl`, `Zp1`, `drF` with units and where each is defined (`drF[k]` is
  the tracer-cell thickness between interfaces `Zp1[k]` and
  `Zp1[k+1]`).
- **"Linear profile"** — defined: a field varying linearly with depth,
  `f(z) = slope·z`; used because finite differences are exact on it.
- **Section 3 (ML mean)** — now explains the thickness-weighted mean
  and prints the full hand computation (per-level depth, `drF`, field,
  in-ML flag) next to the helper output.
- **The "800 m" figure** — the depth axis was already depth-on-y; the
  confusing part was the range: the DBOF store spans the **upper ocean
  only (51 levels, interfaces to 968.6 m)** — that is the intended
  DBOF domain, now stated prominently at the top and in the figure
  captions.
- **Depth profile first / depth on y-axis** — the real vertical
  structure is now Section 1, and every profile figure has depth on
  the y-axis increasing downward.
- **Getting to the middle of the cell** — new test
  (`test_select_at_depth_targets_cell_centres`) and a dedicated figure:
  interfaces (`Zp1`), centres (`Z`), requested targets, and arrows to
  the level actually selected — targets resolve to cell centres, with
  the nearest-centre caveat on non-uniform cells documented.

All tests: 39 offline + 11 online pass (`RUN_LLC_NETWORK_TESTS=1`).
Both notebooks execute end-to-end and finish by running their test
suites.
