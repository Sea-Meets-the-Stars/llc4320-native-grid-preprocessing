# tiles-fields: generalize the tile workflow to any field

**Branch**: ``tiles-depth-fields`` (fresh from current ``main`` —
the old ``tiles_field`` branch is retired as a reference only; do
NOT rebase it, per the PR strategy in ``field_validation.md``).

**Status: PLANNING — no code changes yet.**

## Goal

Make ``src/dbof/tiles`` able to extract a 720×720(×51) tile of ANY
calculated or native field, not just density/temperature/salinity.
Changes must be small, simple, and built on the existing tile code
and the existing (post-migration) field functions — the tile module
gains plumbing only, never physics.

Primary consumer: the depth-phase field-validation notebooks
(``field_validation.md`` §4/§6 — profiles, cross-sections, suffix
cross-checks from regional tiles).

## Question answered first: can surface(-only) fields be tiled?

**Yes — and mostly for free.**  Three cases:

1. **"Surface" versions of depth-resolved fields** (Theta_sfc,
   gradb2_sfc, ζ_sfc, ...): need NO tile support at all.  The tile
   delivers the full (k, j, i) column; surface is ``k=0`` (and
   z25m/mld/mld_mean come from ``depth_strategies`` applied to the
   same tile, in the notebook).  The field functions are
   dimension-agnostic (proven by ``tests/test_calculate_fields.py``),
   so one 3D tile serves every suffix.
2. **Inherently-2D fields from tracer-point inputs** (Eta →
   gradeta2, ug, vg; SIarea, oceQnet passthroughs): computable,
   because the LLC_DEPTH timestep stores carry ALL variables
   (verified: ``configs/transfer/run_depth.yaml`` transfers Theta,
   Salt, U, V, W, Eta, oceTAUX, oceTAUY, SIarea, oceQnet).  Only
   requirement: the output builder must accept (j, i) as well as
   (k, j, i) — one small change.
3. **Wind-stress fields** (rotated τ, curl, ekman): also computable
   (data present; staggered handling identical to U/V), but LOW
   priority — no depth-validation consumer.  Supported by the same
   machinery; entries can be added later without further plumbing.

So: no separate "surface tiles" pathway is needed.  We tile 3D
fields (+ the few true-2D fields); surface is a level selection
downstream.

## Current state (what exists, ``src/dbof/tiles``)

- ``tile_mapping.py``: 432 fixed rect-grid tiles → face-local
  slices.  **No changes needed.**
- ``tile_utils.py``:
  - ``_load_grid_for_tile`` → tile-extent grid via
    ``process_llc4320_3d_grid``, which keeps XC, YC, dxC, dyC, dxG,
    dyG, rA(z), Depth, hFacC, SN, CS, Z(l/u/p1), drF — i.e. ALL
    metrics/rotation coefficients the field functions need. ✔
  - ``_load_tracers_for_tile`` → lazy per-face slice of requested
    vars; currently slices dims ``j``/``i`` only (staggered dims
    ``j_g``/``i_g``/``k_l`` pass through UNsliced — gap #2 below).
  - ``TileProperty`` dataclass with ``compute(ds_tracers_tile) →
    DataArray`` — no grid argument (gap #1 below).
  - ``TILE_PROPERTIES`` = {density, temperature, salinity}.
  - ``compute_tile_property`` → single ``.compute()``, squeeze face,
    float32, attrs.  ``_build_output_dataset`` → NetCDF with XC/YC/Z
    coords + provenance attrs; assumes (k, j, i) (gap #3).
  - ``run()`` end-to-end + QA plot.  **CLI shape unchanged.**
- ``llc4320_ingestion.grid.set_xgcm_grid(ds, use_connections=False)``
  already supports a single-face local grid. ✔

## Design — the minimal change set

1. **Tile ds_merge + local xgcm grid** (the one genuinely new piece):

   ``_build_tile_context(ds_tracers_tile, ds_grid_tile) →
   (ds_merge, grid)`` — ``xr.merge`` of tracers + grid vars, then
   ``set_xgcm_grid(ds_merge, use_connections=False)``.
   Consequence (documented, accepted): horizontal-gradient fields
   are edge-contaminated in a ~3-cell rim at the tile boundary
   (same EDGE_MARGIN as the synthetic tests).  Interior of a 720²
   tile is unaffected.  The rim is NOT trimmed by default; the
   output carries an ``edge_margin`` attr and consumers trim.

2. **Slice staggered dims in ``_load_tracers_for_tile``**: apply the
   tile's j/i slices to ``j_g``/``i_g`` as well (identical ranges —
   tiles are chunk-aligned); leave ``k``/``k_l`` full (whole column).
   Boundary consequence: outer-edge staggered differences produce a
   1-cell NaN rim via xgcm padding without connections — subsumed by
   the edge-margin convention above.

3. **New ``field_registry.py``** (name kept from the old
   ``tiles_field`` branch): moves ``TileProperty`` +
   ``TILE_PROPERTIES`` out of ``tile_utils.py`` and generalizes the
   compute signature to ``compute(ds_merge, grid) → DataArray``.
   The three existing entries are re-expressed in the new signature
   (passthroughs ignore ``grid``); their CLI names stay valid.
   Units/long_names are sourced per ``docs/Fields.md``.

   One registry entry per OUTPUT FIELD.  Functions that return
   multiple arrays (``ertel_pv_terms`` → 3, ``advective_buoyancy_
   fluxes`` → 3, ``geographic_velocity`` → 2, ``geostrophic_
   velocity`` → 2) get one entry per component with a tiny
   extraction wrapper.  A tile run computes ONE field; the
   recompute cost of shared intermediates across separate runs is
   accepted (simplicity > efficiency for validation tiles).

4. **2D-output support**: ``compute_tile_property`` /
   ``_build_output_dataset`` accept (j, i) results (MLD,
   ml_heat_content, Eta-based fields, ug/vg, wind/ice) — skip the
   Z coordinate when there is no ``k`` dim.  QA plot: use k=0 when
   3D, the field itself when 2D.

5. **KE promotion** (already agreed — field_validation.md
   Clarification 4): move the KE formula from
   ``depth_subsets.compute_energetics`` into
   ``calculate_fields_at_depth.kinetic_energy`` so the registry can
   cite one canonical callable.  Only registry-relevant refactor;
   dispatcher calls the new function.

6. **No other changes**: tile_mapping, S3 config handling, output
   paths/filenames, NetCDF format, provenance attrs, ``run()`` CLI,
   QA plot machinery all stay as-is.

## Vector rotation on tiles (LH question, answered)

Raw U/V (and oceTAUX/oceTAUY) in the stores are staggered and in
MODEL x/y directions; we want zonal/meridional.  Rotation is
required on tiles and is ALREADY provided by the existing code:
it is a POINTWISE CS/SN linear map inside
``native_gradient.rotate_vector_to_geographic`` (and the tensor
version inside ``calculate_jacobian``), called internally by every
vector-consuming canonical function (geographic_velocity,
geographic_wind_stress, gradients, vertical_shear).  It is
independent of face stitching (``faces_to_latlon`` rearranges
already-rotated components; it never rotates).  Tile requirements —
CS/SN present (✔ kept by ``process_llc4320_3d_grid``) and a local
xgcm grid for the staggered→tracer interp (✔ design item #1).
NOTE: this is not a small correction — the Pacific sits on the
ROTATED LLC faces where model-x points ~north, so an unrotated
Kuroshio-tile "U" would show the jet in V.  Acceptance test added
below.

## Field compatibility (every DEPTH subset base field)

| Subset | Field(s) | Callable (canonical) | Tile dims | Notes |
|---|---|---|---|---|
| native_fields | Theta, Salt | passthrough | 3D | exists (rename-compatible) |
| native_fields | Eta | passthrough | 2D | needs #4 |
| native_fields | U, V | CF.geographic_velocity | 3D | needs #1, #2; 2 entries |
| native_fields | W | VH._interp_w_to_tracer_levels | 3D | k_l → k centring |
| stratification | N2 | CFAD.buoyancy_frequency_squared | 3D | vertical only — no rim |
| stratification | mixed_layer_depth | CFAD.mixed_layer_depth | 2D | needs #4 |
| stratification | ml_heat_content | CFAD.mixed_layer_heat_content | 2D | needs #4 |
| vertical_shear | vertical_shear | CFAD.vertical_shear_magnitude | 3D | staggered #2 |
| vertical_shear | Ri | CFAD.richardson_number | 3D | |
| mixing_parameters | Fr, Bu, R_ib | CFAD.{froude,burger,balanced_richardson}_number | 3D | Fr/Bu use MLD internally |
| mixing_parameters | Ro | CF.rossby_number | 3D | rim |
| ertel_pv | ertel_pv, _vertical, _tilt | CFAD.ertel_pv_terms | 3D | 3 entries, extraction wrappers; rim |
| buoyancy_fluxes | uB, vB, wB | CFAD.advective_buoyancy_fluxes | 3D | 3 entries |
| energetics | KE | CFAD.kinetic_energy (after #5) | 3D | rim (∇b) |
| frontal_structure | gradb2, gradtheta2, gradsalt2, gradrho2, turner_angle, density, buoyancy | CF.* | 3D | rim for grad*; density/buoyancy rim-free |
| frontal_structure | gradeta2 | CF.grad_eta2 | 2D | needs #4; rim |
| kinematic | ζ, strain_n/s/mag, divergence, okubo_weiss, rossby_number | CF.* | 3D | rim |
| kinematic | coriolis_f | CF.coriolis_parameter | 2D | needs #4 |
| frontogenesis | tendency, geo, Wstar | CF.* | 3D | rim |
| frontogenesis | ageo | tendency − geo (wrapper) | 3D | small wrapper (dispatcher-inline) |
| frontogenesis | ug, vg | CF.geostrophic_velocity | 2D | needs #4; 2 entries |
| surface_wind | oceTAUX, oceTAUY, wind_stress_curl, ekman_pumping, u_ekman, v_ekman, oceQnet | CF.* / passthrough | 2D | IN SCOPE (LH answer 3); staggered τ handled like U/V |
| icearea | SIarea | passthrough | 2D | IN SCOPE (LH answer 3) |

Suffix reductions (``_sfc``/``_z25m``/``_mld``/``_mld_mean``) are
NOT tile properties: they are applied downstream by the validation
notebooks via ``depth_strategies`` on the 3D tile (one tile serves
all suffixes, and cross-checks the Route-A 2D products).

## docs/Tiles.md edits (with the implementation)

- Property table → replaced by a pointer to ``field_registry.py``
  plus the field-compatibility table above.
- New subsection "Merged tile dataset & local xgcm grid": no face
  connections ⇒ ~3-cell edge rim on horizontal-gradient fields
  (``edge_margin`` attr); staggered-dim slicing convention.
- New subsection "2D properties" (MLD, Eta-based, wind/ice).
- CLI examples extended with a calculated field
  (``--property gradb2``, ``--property N2``).

## Testing

- Reuse ``tests/test_calculate_fields.py`` guarantees (dimension-
  agnostic, single implementation) — no re-testing of physics.
- One new test module ``tests/test_tile_field_registry.py``:
  synthetic 32×32×5 single-face dataset → assert every registry
  entry (a) runs through ``compute_tile_property``, (b) returns the
  declared dims, (c) horizontal-gradient fields are finite in the
  interior beyond EDGE_MARGIN.  Mirrors the synthetic-test pattern
  already used for the global pipeline.
- ROTATION acceptance test: synthetic grid with CS=0/SN=1 (90°
  rotated, i.e. Pacific-face-like) and purely "model-x" flow →
  assert the U registry entry returns ~zero and V returns the flow
  (rotation applied, not passthrough).  Real-data spot check for
  the depth notebooks: a Kuroshio tile's rotated U must contain the
  eastward jet.

## Deliverables (in order)

1. ``field_registry.py`` + tile ds_merge/xgcm helper + staggered
   slicing (#1–#3), with the original three properties passing
   through the new signature unchanged (backward-compatible CLI).
2. 2D-output support (#4) + registry entries for all fields in the
   table above (wind/ice deferred unless free).
3. KE promotion (#5) + dispatcher update.
4. ``tests/test_tile_field_registry.py``.
5. ``docs/Tiles.md`` update + this plan's Logs entry.

## Out of scope

- Multi-date tiles, tile mosaics/adjacent-tile stitching (future).
- Any change to the global pipelines or stores.
- Suffix reduction inside tiles (downstream concern).
- Performance work (per-field recompute of shared intermediates is
  accepted).

## Open questions — RESOLVED (LH, 2026-08-03)

1. Registry naming: **match subset channel names exactly** (``N2``,
   ``mixed_layer_depth``, ``relative_vorticity``, …);
   ``density``/``temperature``/``salinity`` kept as legacy aliases.
2. Edge rim: **keep 720×720 shape, explicitly NaN the rim** (per
   registry-declared margin, ≤3 cells; ``edge_margin`` attr on the
   output) for horizontal-gradient fields.  Self-documenting (bad
   cells cannot be used silently, same convention as land NaNs),
   shapes stay uniform and pixel-aligned with the Route-A products;
   rim-free fields (verticals, passthroughs, N², MLD) untouched.
   Each ``TileProperty`` declares its own ``edge_margin`` (0 for
   rim-free, 2–3 for gradient/Jacobian fields).
3. Wind/ice entries: **build now** (oceTAUX, oceTAUY,
   wind_stress_curl, ekman_pumping, u_ekman, v_ekman, oceQnet,
   SIarea — all 2D; data confirmed present in LLC_DEPTH stores).
   Table rows move from "deferred" to in-scope.

## Logs

### 2026-08-04 — Implementation complete (branch `tiles-depth-fields`)

**Code.**

- NEW `src/dbof/tiles/field_registry.py`: `TileProperty` dataclass
  (adds `edge_margin`, uniform `compute(ds_merge, grid)` signature) +
  ~50 entries keyed by exact subset channel names, covering every
  SURF and DEPTH channel incl. wind/ice. Physics never lives in the
  registry: entries point at canonical CF/CFAD functions; only small
  adapters are local (`_passthrough`, `_no_grid`, `_pick` for
  tuple/dict multi-output functions, `_w_centred`,
  `_frontogenesis_ageo`). `ALIASES` keeps `temperature`/`salinity`
  working; `resolve_property()` is the single lookup.
- `src/dbof/tiles/tile_utils.py`: local registry replaced by
  re-exports from `field_registry` (backward-compatible imports);
  NEW `_build_tile_context()` (merge tile tracers+grid, local xgcm
  grid via `set_xgcm_grid(use_connections=False)`);
  `_load_tracers_for_tile` now slices the staggered dims `j_g`/`i_g`
  with the same tile slices (k/k_l kept full);
  `compute_tile_property(ds_merge, grid, prop)` squeezes the face
  dim, casts float32, NaNs the `edge_margin` boundary rim, and
  records `edge_margin` as an output attr; `_build_output_dataset`
  and `_qa_plot` are 2D-aware (Z coord only when `k` present;
  `isel(k=0)` only for 3D); `run()` resolves via
  `resolve_property` and builds the tile context.
- KE promotion (plan Clarification 4): canonical
  `kinetic_energy(ds_merge, grid, mld=None)` appended to
  `calculate_fields_at_depth.py`;
  `depth_subsets.compute_energetics` now delegates to it; the KE
  registry entry points at the same function.

**Tests** (all offline, all passing).

- NEW `tests/test_tile_field_registry.py` (59 tests): every registry
  entry runs end-to-end through `compute_tile_property` on a
  single-face slice of the shared `tests/synthetic_llc.py` fixtures
  (dims/dtype/attrs contract, NaN edge rim, finite interior beyond
  the rim); coverage test asserts EVERY channel of every SURF and
  DEPTH subset resolves to a registry entry (registry cannot fall
  behind `subset_definitions`); alias + unknown-name tests; the
  rotation acceptance test — synthetic CS=0/SN=1 (Pacific-face-like)
  tile with purely model-x flow must yield U≈0 / V≈flow, plus the
  CS=1/SN=0 identity check.
- `tests/test_generate_tile.py` updated for the generalized
  registry: `test_run_round_trip` now parametrizes the cheap
  constant-field cases (`density`, both legacy aliases, `Theta`,
  `Salt`, and `Eta` exercising the 2D output path end-to-end
  through the REAL `run()` with S3 mocked — the full per-entry sweep
  lives in the new file); synthetic tracers are dask-backed
  (production-faithful, bounded test memory) and now include `Eta`;
  the density-delegation test looks through the `_no_grid` adapter
  closure for the canonical `potential_density_anomaly`.

**Docs.** `docs/Tiles.md` rewritten where stale: module table gains
`field_registry.py`; Properties section now points at the registry +
subset_definitions as ground truth and documents the tile context
(local xgcm grid), `edge_margin` NaN rim, vector rotation (never
passthrough; CS/SN), 2D outputs, aliases; CLI/API examples show
arbitrary channels; output-format section documents 2D shape and the
`edge_margin` attr; testing section covers both test files.

**Notes / deviations from plan.**

- Rim policy implemented exactly as resolved: full 720×720 shape
  kept, rim NaN'd per-property, width recorded as attr (0 rim-free,
  1 staggered-interp, 3 horizontal-derivative chains).
- `strain` returns `(mag, n, s)` and `ertel_pv_terms`/
  `ekman_transport` return dicts — handled by the `_pick` adapter,
  no canonical-function changes needed.
- `frontogenesis_ageo` has no canonical single function (it is the
  tendency−geo residual inline in `surface_subsets`); the registry
  mirrors that residual in a local adapter.
- The network integration test
  `test_balanced_richardson_against_real_tile` (S3) was not run
  here (offline environment) — unchanged by this work.

### 2026-08-04 (addendum) — grid-loader staggered-dim gap, caught in review

LH review question ("did we account for i_g/j_g like the old
tiles_field branch?") surfaced a real gap: `_load_tracers_for_tile`
sliced the staggered dims, but `_load_grid_for_tile` sliced only
`face/j/i` — on real data the grid metrics (`dxC`/`dyG` on
`(j, i_g)`, `dyC`/`dxG` on `(j_g, i)`, `rAz` on `(j_g, i_g)`) would
have come down full-face (4320) and misaligned with the 720-sliced
tracers at the tile-context merge. The synthetic test grids carried
no staggered dims, so tests missed it.

Fix: ported `_tile_indexer(ds, tile)` verbatim from the retired
`tiles_field` branch (one shared indexer covering tracer AND
staggered horizontal dims, restricted to dims present in the
dataset; k/k_l untouched) and used it in BOTH loaders. Added
`test_tile_indexer_slices_staggered_dims` to
`tests/test_generate_tile.py` (26 offline tests passing).
`docs/Tiles.md` tile-context section now documents the staggered
slicing explicitly (restored old-branch caveat #1).

Second old-branch difference, resolved by LH (port it): `tiles_field`
land-masked tiles via `hFacC == 0 → NaN` (`mask_land=True` default) —
a cheap safety since derivative fields are pathological over/near
land. Ported: `compute_tile_property(..., mask_land=True)` applies the
mask after the edge rim (hFacC surface-collapsed, residual face/k dims
handled defensively); `run(..., mask_land=True)` passes it through;
CLI gains `--no-mask-land`. New test
`test_compute_tile_property_land_mask` (mask on/off); full offline
tile suites re-run green (27 + 59). `docs/Tiles.md` masking section
updated (default mask, `--no-mask-land`, raw physical values note).

Third old-branch difference (LH review): old `_build_tile_xgcm_grid`
vs new direct `set_xgcm_grid` import. Same core (the old helper also
called `set_xgcm_grid(use_connections=False)` — it was a wrapper, not
a parallel implementation), but it additionally dropped the vertical
coordinate vars (Z/Zl/Zu/Zp1/drF) before the xgcm build, mirroring
`grid_setup.set_up_grid_depth` ("xgcm only needs the horizontal
stencil"). Ported into `_build_tile_context`: the drop applies only
to what xgcm sees (`_VERTICAL_VARS` imported from `grid_setup` —
defined once); the returned `ds_merge` keeps Z/drF for the compute
callbacks (N2, MLD need them). Synthetic tests can't catch this
(their k/k_l coords carry no comodo attrs, so xgcm never builds a Z
axis) — real-data parity motivated the port. Suites green (86).
