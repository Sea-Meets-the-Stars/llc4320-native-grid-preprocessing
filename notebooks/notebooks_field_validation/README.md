# Field-validation notebooks

One end-to-end validation notebook per calculated-field subset (plan:
`prompts/field_validation.md`; field definitions: `docs/Fields.md`).
Each notebook RUNs the pipeline for its subset at the single validation
timestep **2012-11-09 12:00:00** (run_id `field_validation_v1`, config
`configs/global/run/field_validation_surface.yaml`), LOADs its own
output, and validates every field with dependency-chain maps
(Figure 1), PDFs (Figure 2), and a literature comparison (Figure 3).

Layout: `surface_fields/{subset}.ipynb` (SURF) and
`depth_fields/{subset}.ipynb` (DEPTH).  Surface phase first.
Literature reference images go in `literature/{subset}/{field}.png`.

Shared plotting modules: `src/dbof/plotting/` — `regions.py`
(validation domains), `pipeline_grids.py` (map grids),
`pdfs.py` (PDF grids), `literature_comparison.py`.

## Validation domains (rows of Figures 1–2)

| Row | Region | Box |
|---|---|---|
| A | Global | downsampled `[::12, ::12]` |
| B | Gulf Stream (W. North Atlantic) | 80–40°W, 25–45°N |
| C | Kuroshio (NW Pacific) | 135–165°E, 25–45°N |
| D | Southern Ocean (Atlantic sector) | 20°W–20°E, 65–40°S |
| E | Eastern tropical Pacific | 140–90°W, 5°S–5°N |

## Surface subsets (SURF / OSN) — `surface_fields/`

Channel lists are verbatim from
`global_dataset_creation/subset_definitions.SURFACE_SUBSETS`
(raw = model output channel; all others computed).

| Notebook | Fields (channels) | Status |
|---|---|---|
| `native_fields.ipynb` | Theta (raw), Salt (raw), Eta (raw), W (raw), U, V | planned |
| `surface_wind.ipynb` | oceQnet (raw, SURF only), oceTAUX, oceTAUY, wind_stress_curl, ekman_pumping, u_ekman, v_ekman | planned |
| `icearea.ipynb` | SIarea (raw) | planned |
| `frontal_structure.ipynb` | gradb2, gradsalt2, gradtheta2, gradeta2, gradrho2, turner_angle, density, buoyancy | **template — built** |
| `kinematic.ipynb` | relative_vorticity, strain_n, strain_s, strain_mag, divergence, coriolis_f, rossby_number, okubo_weiss | planned |
| `frontogenesis.ipynb` | frontogenesis_tendency, ug, vg, frontogenesis_geo, frontogenesis_ageo, Wstar | planned |

## Depth subsets (DEPTH) — `depth_fields/`

Base channels expand with the active depth suffixes
(`_sfc`, `_z25m`, `_mld`, `_mld_mean` by default; bases in
`SURFACE_ONLY_BASES` — Eta, gradeta2, ug, vg — emit `_sfc` only).
Extra channels are inherently 2D.

| Notebook | Base fields (× suffixes) | Extra channels | Status |
|---|---|---|---|
| `stratification.ipynb` | N2 | mixed_layer_depth, ml_heat_content | planned |
| `vertical_shear.ipynb` | vertical_shear, Ri | — | planned |
| `mixing_parameters.ipynb` | Fr, Ro, Bu, R_ib | — | planned |
| `ertel_pv.ipynb` | ertel_pv, ertel_pv_vertical, ertel_pv_tilt | — | planned |
| `buoyancy_fluxes.ipynb` | uB, vB, wB | — | planned |
| `energetics.ipynb` | KE | — | planned |
| `frontal_structure.ipynb` | gradb2, gradtheta2, gradsalt2, gradrho2, gradeta2, turner_angle | — | planned |
| `kinematic.ipynb` | relative_vorticity, strain_n, strain_s, strain_mag, divergence, rossby_number, okubo_weiss | coriolis_f | planned |
| `frontogenesis.ipynb` | frontogenesis_tendency, frontogenesis_geo, frontogenesis_ageo, ug, vg, Wstar | — | planned |
| `native_fields.ipynb` | Theta, Salt, Eta, U, V, W | — | planned |
| `surface_wind.ipynb` | surface-only; identical to SURF — one-line reference to the surface notebook | oceQnet (raw) | planned |
| `icearea.ipynb` | SIarea (raw, surface-only) | — | planned |

## Conventions (fixed across all notebooks)

- Maps: one shared colour scale per dependency column; global row on
  cartopy Robinson with regional-extent boxes; land/halo NaNs shown
  gray; log colour scale for ∝-squared gradient fields.
- PDFs: probability density; land + halo-rim NaNs removed before
  binning; log10-x for ∝-squared fields; bins shared per field across
  the four domains; |lat|>2° filter for f-normalised fields in Row D
  (stated in-figure).
- Cross-references instead of re-validation: σ₀/b →
  `surface_fields/frontal_structure.ipynb`; MLD → DEPTH
  `stratification.ipynb`; rotation/Jacobian machinery →
  `surface_fields/kinematic.ipynb`.
