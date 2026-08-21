# Depth field validation — plan

Companion to `prompts/field_validation.md` (the original surface+depth
plan).  That document is history; **this one is what we are building.**
The surface phase is done and taught us things — see
`docs/Gradients.md`.

---

## 1. What we are making

One notebook per DEPTH subset:

    notebooks/notebooks_field_validation/depth_fields/{subset}.ipynb

Each notebook validates every channel in its subset the same way the
surface notebooks do — maps, PDFs, then a literature comparison — but
sliced by **depth** instead of by **region**.

Subsets, in build order:

| # | Notebook | Fields |
|---|---|---|
| 1 | `stratification.ipynb` | N2 (+ mixed_layer_depth, ml_heat_content) |
| 2 | `native_fields.ipynb` | Theta, Salt, U, V, W, Eta_sfc |
| 3 | `vertical_shear.ipynb` | vertical_shear, Ri |
| 4 | `frontal_structure.ipynb` | gradb2, gradtheta2, gradsalt2, gradrho2, gradeta2_sfc, turner_angle |
| 5 | `kinematic.ipynb` | relative_vorticity, strain_n/s/mag, divergence, rossby_number, okubo_weiss, coriolis_f |
| 6 | `frontogenesis.ipynb` | frontogenesis_tendency/geo/ageo, ug, vg, Wstar |
| 7 | `mixing_parameters.ipynb` | Fr, Ro, Bu, R_ib |
| 8 | `ertel_pv.ipynb` | ertel_pv, ertel_pv_vertical, ertel_pv_tilt |
| 9 | `buoyancy_fluxes.ipynb` | uB, vB, wB |
| 10 | `energetics.ipynb` | KE |

`surface_wind` and `icearea` are surface-only — one-line pointers to the
surface notebooks, no new work.

All ten are built.  `surface_wind` and `icearea` are pointer
notebooks.

---

## 2. The one big change from the surface notebooks

Surface: **rows were regions**, columns were raw → intermediate → final.

Depth: **rows are depth levels.**  Columns stay the same.

Figure 1 for a field is therefore 8 rows:

| Row | What |
|---|---|
| 1 | sfc, whole tile |
| 2 | z25m, whole tile |
| 3 | mld, whole tile |
| 4 | mld_mean, whole tile |
| 5 | sfc, 200 × 200 km zoom |
| 6 | z25m, zoom |
| 7 | mld, zoom |
| 8 | mld_mean, zoom |

Rows 1–4 carry a **crimson square** showing where the zoom is.

**One region per notebook, chosen by the user at the top of the
notebook.  Default: Gulf Stream.**

Every column in a row is at that row's depth level, so the chain reads
left to right at a single level.  Fields that are inherently 2D
(`Eta`, `gradeta2`, `ug`, `vg`, `coriolis_f`, `mixed_layer_depth`,
`ml_heat_content`) repeat down the rows and are labelled as such.

Fields that **do not vary with depth** — `mixed_layer_depth`,
`ml_heat_content`, `coriolis_f` — get **two** rows instead of eight:
whole tile and zoom.  Four identical rows say nothing.

Figure 2 (PDFs) uses only rows 1–4 — the whole tile at each level.
The zoom boxes are too small to make an honest histogram.  Bins are
shared down a column so you can see the distribution change with
depth.

Figure 3 is **depth profiles**.  Five ocean columns, spread across the
tile and fixed by a seed so every field profiles the same water.  The
left panel shows where they are as numbered, colour-coded ×; then one
panel per 3D field in the chain, surface at the top, depth increasing
down, each location in its own colour.  Dashed lines mark each
location's MLD.  Colours are Okabe–Ito, and the numbers repeat in the
legend so the five never depend on colour alone.

---

## 3. Where the data comes from

**Not** from running `generate-global`.  That would compute the whole
planet to look at one place.

Each notebook works on **one tile** — the 720 × 720 × 51 block the
`dbof.tiles` workflow already defines: one LLC face, the full water
column, about 1400 × 1400 km.  From
`s3://dbof/LLC4320_RAW/DEPTH/20121109T12.zarr`:

1. The region's anchor lon/lat picks the tile
   (`tile_utils.latlon_to_rect_ij` → `tile_mapping.rect_ij_to_tile`).
2. That tile is loaded, plus the matching piece of `grid.zarr`.
3. `tile_utils._build_tile_context` merges them and builds a **local**
   xgcm grid.
4. The **production** compute function for the subset
   (`depth_subsets.compute_*`) runs on it and internally applies the
   four depth strategies.  Same code as production — that is the point.
5. The edge rim is NaN'd, land is masked, and the result is plotted on
   the tile's own XC/YC.

Why a tile and not something bigger:

- A tile is **exactly one chunk** of the depth store, so step 2 costs
  one S3 GET per variable.  About 106 MB per 3D field.
- Tiles are 720-aligned and faces are 6 × 720, so a tile **can never
  straddle two LLC faces**.  There is no geometry that breaks it.
- All of it already exists.  Nothing new is needed to get the data.

The one thing to keep in mind: **a tile samples a region, it does not
cover it.**  "Gulf Stream" means the tile around 60°W, 37°N, not the
whole 80–40°W box the surface notebooks use.  The 200 km zoom sits at
the same anchor.

And one cost we accept: a local xgcm grid has no face connections, so
cells within `edge_margin` of the tile boundary have no neighbours and
their gradients are wrong.  We NaN that rim — the same convention and
the same per-field numbers the tile registry already records (0 for
purely vertical fields, 1 for staggered interpolation, 3 for gradient
and Jacobian chains).

---

## 4. Sparkle rules — carried over unchanged

From `docs/Gradients.md`.  Nothing here changes with depth; the depth
notebooks inherit the same four cases and should say so rather than
re-litigating them:

- **Squares** (`gradb2`, `gradtheta2`, …): squared before interpolating
  → clean.
- **Magnitude of a Jacobian** (`strain_mag`, `okubo_weiss`, `Wstar`):
  squared at their native points before interpolating → clean.
- **Plain Jacobian** (`relative_vorticity`, `divergence`): ECCO recipe,
  artifact present but not amplified.  Kept deliberately so everything
  lands on cell centres.
- **Products of two different components** (`frontogenesis` — bx·by):
  must interpolate before multiplying, so the artifact is unavoidable.

Practical effect on these notebooks: when a sparkle shows up in
`kinematic` or `frontogenesis` at any depth, it is expected and
already explained — note it, do not chase it.

### 4b. And two the depth pipeline adds

Both have their own notebook, because both show up across many subsets
and neither is worth re-arguing in each one.

**The centred vertical stencil** (`vertical_gradients.ipynb`).  The
production vertical derivative is
`(f[k+1] − f[k−1]) / (z[k+1] − z[k−1])`, which never reads level k.
That averaging does two different things:

- *same-sign slopes* (a monotone kink, e.g. the mixed-layer base) — the
  centred form returns their mean, a correct second-order estimate
  where the derivative is not well defined.  **Not an error.**
  Reported as `dz_asym`.
- *opposite-sign slopes* (a vertical extremum) — they annihilate and
  the centred form collapses toward zero.  **This is the artifact.**
  Reported as `dz_signflip`, and it must be **measured directly**: an
  `asym > 0.5` proxy undercounts it badly on this strongly non-uniform
  vertical grid.

Measured on the Gulf Stream tile, ∂ρ/∂z: **5.5% of cells flipped at
25 m and 7.2% at the MLD**, peaking near 14% around 40–50 m and falling
to ~0 below 200 m.  So the MLD row is the worst of the four for
cancellation as well as for curvature — it samples close to the peak.

`dz_signflip` reduces differently per level.  At `sfc`, `z25m` and
`mld` one level is picked, so it stays 0/1 and its mean is the fraction
of cells affected.  At `mld_mean` it is thickness-weighted over the
layer, arrives as a fraction, and is essentially never exactly 1 —
average it, never threshold it.

`vertical_gradients.ipynb` runs the diagnostic across six field types
(raw tracer → buoyancy → horizontal gradient → Jacobian → vertical
gradient → vertical×horizontal product), because a field that is
*already a derivative* is rougher in the vertical and cancels more.
It also settles the `mld_mean` order-of-operations question:
`mean(a·b) ≠ mean(a)·mean(b)`, and the difference is the mixed-layer
covariance — a real quantity, not an error.  Production computes
`mean(a·b)`; `docs/Fields.md` should say so.

**The MLD staircase** (`mixed_layer_depth.ipynb`).  A separate
mechanism with similar symptoms, and the likelier cause of blotchy
`_mld` rows.  `mixed_layer_depth` returns *the deepest model level*
satisfying the density criterion, so MLD can only take the 51 discrete
values in `Z` — 10–20 m apart near typical mixed layers.  The `_mld`
maps are not noisy, they are **quantised**.

Not an extraction bug: given that definition, `_extract_at_mld`'s
nearest-k lookup is exact.  Making the extraction interpolate would
change nothing.  The MLD itself has to become continuous first.

**Three possible orders, and they are not equivalent:**

| | Order | Meaning |
|---|---|---|
| A | compute in 3D → snap to nearest level | production today |
| B | compute in 3D → interpolate in z | removes the staircase, keeps the along-level meaning |
| C | interpolate inputs to the MLD → compute | gradient along the tilted MLD *surface* |

C is a trap for anything with a horizontal gradient: ∇ along the MLD
surface adds `(∂b/∂z)·∇(MLD)`, and with a staircase MLD that term is a
field of spikes at the level boundaries.  Verified on synthetic data —
C comes out 16× rougher, correlating at 0.96 with the predicted
`A + (∂b/∂z·∇MLD)²`.  **Rule C out explicitly.**

`mixed_layer_depth_DI` — **DI = Depth Integration** — defines the MLD
as the N²-weighted mean depth over a fixed integration depth.  It maps
smoothest of the three, but *because it is an integral*: it averages
over the whole profile by construction.  That is not evidence it is a
better MLD, and its values sit well below the threshold definitions.
Choose it, if at all, on a physics argument about what the consumer
needs — not on a smoothness contest.

---

## 5. Notebook structure

Same skeleton as the surface notebooks, so they read alike.

1. **Setup** — pick the region (default Gulf Stream), the date, the
   depth levels.  One cell, all the knobs at the top.
2. **Load** — resolve and load the tile, then run the subset's
   production compute function on it.
3. **Subset** — the channel list, verbatim from
   `subset_definitions.DEPTH_SUBSETS`.
4. **Field & dependency table** — field, units, equation, dependencies,
   where the code lives.
5. **Per field** — Figure 1 (8-row maps), Figure 2 (4-row PDFs),
   Figure 3 (depth profiles at 5 locations).
5b. **Vertical stencil check** — only in `stratification`,
   `vertical_shear`, `mixing_parameters` and `ertel_pv`, the four whose
   fields divide by or square a vertical derivative.  See §4b.
6. **Literature comparison** — *left open.*  LH picks a figure from a
   paper, drops the PNG in `literature_figures/`, and only then do we
   decide which of our panels to put beside it.  Until then the section
   is a placeholder saying exactly that.
7. **Summary** — coverage and range per channel, plus physical checks
   that assert.

New shared code: **one module**,
`src/dbof/plotting/depth_figures.py` — the figure layouts plus the
small amount of shaping they need (reduce to levels, NaN the rim, mask
land, pack).  Everything else is reused as-is: `dbof.tiles` for the
data, `regions.py` for the anchors, `field_cmaps.yaml`,
`global_maps.py`, `pdfs.py`.

Note on the word "grid": in `depth_figures.py` a *grid* is the rows ×
columns array of panels in a figure.  The model's Arakawa C-grid is a
different thing entirely — that is `docs/Grid.md` and
`native_gradient.py`.

---

## 6. Open items

- `R_ib` has no `field_cmaps.yaml` entry — add one before #7.
- Depth-suffixed intermediate keys (`db_dx_mld`, `rho_theta_mld`, …)
  are missing from `field_cmaps.yaml` — add as each notebook needs
  them.
- The f-normalised fields (Ro, R_ib, W*) go NaN/extreme near the
  equator.  The surface notebooks filter `|lat| > 2°` in the PDFs; the
  depth figures need the same guard before #5, #6 and #7 if anyone
  points them at an equatorial tile.
- Only one DEPTH date exists (20121109T12); the notebooks are
  parameterised by date for when more arrive.
- **`ml_heat_content` returns exactly 0, not NaN**, for any column
  where the MLD mask catches no model level — `.where(mask).sum()`
  uses xarray's default `skipna=True` and the sum of nothing is 0.
  The stratification notebook counts and excludes those cells; the
  production field in `calculate_fields_at_depth.py` still has them.
  One-line fix: `min_count=1` on that sum.

*Generated by LH and Claude.*
