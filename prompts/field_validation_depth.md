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

Build #1 first, review it, then roll out the rest.

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

Figure 2 (PDFs) uses only rows 1–4 — the whole tile at each level.
The zoom boxes are too small to make an honest histogram.  Bins are
shared down a column so you can see the distribution change with
depth.

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
5. **Per field** — Figure 1 (8-row maps) then Figure 2 (4-row PDFs).
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

*Generated by LH and Claude.*
