# DBOF Calculated Fields

Reference list of every output field, organised by subset. Pipeline
mechanics (running subsets, depth-suffix expansion, re-run behaviour) live
in `Global_Maps.md`; this document is the field-level reference and the
checklist for the per-subset verification notebooks (planned; one notebook
per subset).

Function names refer to `preprocessing/calculate_fields.py` (single
lazy, dimension-agnostic implementations — the same functions serve 2D
and 3D inputs), `preprocessing/calculate_fields_at_depth.py`
(vertical-structure fields only), and inline in the subset dispatchers.
There is exactly ONE implementation per field (field migration,
`prompts/field_migration.md` — complete).  Conventions: the single
reference density RHO0_REFERENCE = 1000 kg/m³ and g = 9.81 m/s² apply
throughout; buoyancy is anomaly-based (b = g·σ₀/ρ₀); the `density`
channel is the potential density ANOMALY σ₀ (not σ₀ + 1000).  These
invariants are pinned by `tests/test_calculate_fields.py`.

## Conventions

- All computed fields are on tracer points; vectors are rotated to
  geographic (eastward/northward) components via the grid `CS`/`SN`
  coefficients.
- Density is potential density: JMD95 evaluated at p = 0 with potential
  temperature (surface-referenced; equals in-situ density at the surface
  only).  The output channel carries the anomaly σ₀ = ρ_θ − 1000.
- Buoyancy is b = g·σ₀/ρ₀ (anomaly-based; constant offsets vanish under
  the gradients/derivatives that consume it).
- In the DEPTH pipeline, each base field is expanded across the active
  depth suffixes (`sfc`, `z25m`, `mld`, `mld_mean` by default). Bases in
  `SURFACE_ONLY_BASES` (`Eta`, `gradeta2`, `ug`, `vg`) only ever emit
  `_sfc`. Extra channels are emitted as-is (inherently 2D).

### Staggered-grid stencils: squared fields are square-BEFORE-interp

LLC4320 is an Arakawa C-grid — tracers at cell centres, `U`/`V` on
staggered edge points, vorticity naturally on cell corners (see the
[MITgcm horizontal-grid documentation](
https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html)).
Two stencil families coexist in `utils/native_gradient.py`:

- **Vector components** (∇b for frontogenesis, the velocity Jacobian,
  geographic `U`/`V`) follow the ECCO recipe: finite-difference on the
  staggered points, 2-point-interpolate to the tracer centre, rotate
  to geographic via `CS`/`SN`
  (`calculate_native_gradient_tracer`, `calculate_jacobian`).
- **Squared magnitudes** (`grad*2`, `strain_mag`, `okubo_weiss`, and
  everything built on them: `turner_angle`, `R_ib`, `KE`, `Wstar`)
  are built **square-before-interp**: each difference is squared ON
  its native C-grid point and only the non-negative squares are moved
  between grid locations (`calculate_grad_squared_tracer`,
  `kinematic_invariants` + `interp_corner_squared`).

Why: the 2-point interpolation has a null space at the grid scale —
at a local extremum the two flanking one-sided differences are
equal-and-opposite and cancel in the average.  Squaring the
interpolated components therefore manufactures near-zero pixels
('sparkle' on log-scaled maps; spurious huge values where a squared
gradient sits in a denominator, e.g. `R_ib`).  A mean of
non-negative squares cannot cancel.  The squared magnitudes are
rotation-invariant, so the square-first forms never apply `CS`/`SN`.
Consequences: `strain_mag` ≠ `sqrt(strain_n² + strain_s²)`
pixelwise, and `okubo_weiss` uses the corner (MITgcm `momVort3`)
vorticity stencil rather than the centred `relative_vorticity`
channel — state this wherever the channels are compared pixel by
pixel.  A/B validation:
`notebooks/notebooks_dev/field_validation_sparkle.ipynb`; decision
log: `prompts/field_validation.md` (2026-08-05).

---

## Surface pipeline subsets (SURF / OSN)

All surface subsets are 2D (k = 0 input data, no depth suffixes).

### `native_fields` (native_fields.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `Theta` | model output | Potential temperature | °C |
| `Salt` | model output | Salinity | psu |
| `Eta` | model output | Sea-surface height anomaly | m |
| `W` | model output | Vertical velocity | m s⁻¹ |
| `U` | `geographic_velocity` | Eastward velocity (interpolated to tracer points, rotated to geographic) | m s⁻¹ |
| `V` | `geographic_velocity` | Northward velocity (interpolated to tracer points, rotated to geographic) | m s⁻¹ |

### `surface_wind` (surface_wind.zarr) — requires wind data

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `oceTAUX` | `geographic_wind_stress` | Eastward wind stress on tracer points | N m⁻² |
| `oceTAUY` | `geographic_wind_stress` | Northward wind stress on tracer points | N m⁻² |
| `wind_stress_curl` | `wind_stress_curl` | ∂τy/∂x − ∂τx/∂y | N m⁻³ |
| `ekman_pumping` | `ekman_pumping` | w_E = curl(τ)/(ρ₀ f); NaN at the equator | m s⁻¹ |
| `u_ekman` | `ekman_transport` | Zonal Ekman transport τ_φ/(ρ₀ f) | m² s⁻¹ |
| `v_ekman` | `ekman_transport` | Meridional Ekman transport −τ_λ/(ρ₀ f) | m² s⁻¹ |
| `oceQnet` | model output (SURF only) | Net surface heat flux | W m⁻² |

### `icearea` (icearea.zarr) — requires ice data

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `SIarea` | model output | Sea-ice area fraction | 0–1 |

### `frontal_structure` (frontal_structure.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `gradb2` | `grad_b2` | Squared horizontal buoyancy-gradient magnitude \|∇b\|² | s⁻⁴ |
| `gradsalt2` | `grad_salt2` | \|∇S\|² | (psu m⁻¹)² |
| `gradtheta2` | `grad_theta2` | \|∇θ\|² | (°C m⁻¹)² |
| `gradeta2` | `grad_eta2` | \|∇η\|² | (m m⁻¹)² |
| `gradrho2` | `grad_rho2` | \|∇ρ\|² (potential density) | (kg m⁻⁴)² |
| `turner_angle` | `turner_angle` | Tu_h = arctan(ρ₀(β²\|∇S\|² − α²\|∇θ\|²) / (−\|∇ρ\|²/ρ₀)); NaN where \|∇ρ\|² = 0 | degrees |
| `density` | `potential_density_anomaly` | Potential density anomaly σ₀ = ρ_θ(p=0) − 1000 | kg m⁻³ |
| `buoyancy` | `buoyancy_of_field` | b = g·σ₀/ρ₀ | m s⁻² |

### `kinematic` (kinematic.zarr)

All velocity-gradient fields share one Jacobian computed per subset run.

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `relative_vorticity` | `relative_vorticity` | ζ = ∂v/∂x − ∂u/∂y | s⁻¹ |
| `strain_n` | `strain` | Normal strain Sn = ∂u/∂x − ∂v/∂y | s⁻¹ |
| `strain_s` | `strain` | Shear strain Ss = ∂u/∂y + ∂v/∂x | s⁻¹ |
| `strain_mag` | `strain` | √(Sn² + Ss²) | s⁻¹ |
| `divergence` | `divergence` | δ = ∂u/∂x + ∂v/∂y | s⁻¹ |
| `coriolis_f` | `coriolis_parameter` | f = 2Ω sin(lat) | s⁻¹ |
| `rossby_number` | `rossby_number` | Ro = ζ/f | — |
| `okubo_weiss` | `okubo_weiss_parameter` | W = Sn² + Ss² − ζ² (positive ⇒ strain-dominated) | s⁻² |

### `frontogenesis` (frontogenesis.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `frontogenesis_tendency` | `frontogenesis_tendency` | F(u, v) = −(∂u/∂x·bx² + (∂u/∂y + ∂v/∂x)·bx·by + ∂v/∂y·by²) | s⁻⁵ |
| `ug` | `geostrophic_velocity` | ug = −(g/f)·∂η/∂y; NaN/inf in a narrow equatorial band | m s⁻¹ |
| `vg` | `geostrophic_velocity` | vg = (g/f)·∂η/∂x | m s⁻¹ |
| `frontogenesis_geo` | `frontogenesis_geo` | F(ug, vg) — same formula with geostrophic velocities | s⁻⁵ |
| `frontogenesis_ageo` | subset dispatcher (inline) | F(u, v) − F(ug, vg) | s⁻⁵ |

---

## Depth pipeline subsets (DEPTH)

Base fields expand across depth suffixes (default: `_sfc`, `_z25m`,
`_mld`, `_mld_mean`); extra channels are inherently 2D and emitted as-is.

### `stratification` (stratification.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `N2_{sfx}` | `buoyancy_frequency_squared` | N² = (g/ρ₀)·∂ρ/∂z (positive-down z convention) | s⁻² |
| `mixed_layer_depth` (extra) | `mixed_layer_depth` | Threshold MLD: deepest z where σ₀ − σ₀(10 m) ≤ 0.03 kg m⁻³ (Bodner et al.); positive metres | m |
| `ml_heat_content` (extra) | `mixed_layer_heat_content` | Q_ml = ∫₀^MLD cp·ρ₀·θ dz | J m⁻² |

Note: `mixed_layer_depth_DI` (N²-weighted Depth Integration estimator)
exists as a function but is not currently an output channel.

### `vertical_shear` (vertical_shear.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `vertical_shear_{sfx}` | `vertical_shear_magnitude` | \|S\| = √(uz² + vz²), geographic components | s⁻¹ |
| `Ri_{sfx}` | `richardson_number` | Ri = N²/(uz² + vz²); N² floored at 0; NaN where shear² = 0 | — |

### `mixing_parameters` (mixing_parameters.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `Fr_{sfx}` | `froude_number` | Fr = speed/(N·MLD); NaN where N·MLD ≤ 0 | — |
| `Ro_{sfx}` | `rossby_number` | Ro = ζ/f | — |
| `Bu_{sfx}` | `burger_number` | Bu = (Ro/Fr)²; NaN where Fr = 0 | — |
| `R_ib_{sfx}` | `balanced_richardson_number` | R_ib = N²f²/\|∇_h b\|² (Thomas, Tandon & Mahadevan 2013); N² floored at 0; NaN where \|∇_h b\|² = 0 | — |

### `ertel_pv` (ertel_pv.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `ertel_pv_{sfx}` | `ertel_pv_terms` | q = (ζ + f)·b_z + (w_y − v_z)·b_x + (u_z − w_x)·b_y | s⁻³ |
| `ertel_pv_vertical_{sfx}` | `ertel_pv_terms` | q_vert = (ζ + f)·b_z | s⁻³ |
| `ertel_pv_tilt_{sfx}` | `ertel_pv_terms` | q_tilt = (w_y − v_z)·b_x + (u_z − w_x)·b_y | s⁻³ |

### `buoyancy_fluxes` (buoyancy_fluxes.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `uB_{sfx}` | `advective_buoyancy_fluxes` | u·b (geographic u, W interpolated to tracer levels for wB) | m² s⁻³ |
| `vB_{sfx}` | `advective_buoyancy_fluxes` | v·b | m² s⁻³ |
| `wB_{sfx}` | `advective_buoyancy_fluxes` | w·b | m² s⁻³ |

### `energetics` (energetics.zarr)

| Channel | Source / function | Definition | Units |
|---|---|---|---|
| `KE_{sfx}` | subset dispatcher (inline) | Mixed-layer eddy scaling KE = ½·(MLD·\|∇b\|/f)² | m² s⁻² |

### `frontal_structure` (frontal_structure.zarr)

Same definitions as the surface subset, evaluated in 3D and reduced at
each suffix. `gradeta2` is surface-only (emits `_sfc` only). Note: the
DEPTH variant has no `density`/`buoyancy` output channels.

| Channel | Source / function |
|---|---|
| `gradb2_{sfx}` | `grad_b2` |
| `gradtheta2_{sfx}` | `grad_theta2` |
| `gradsalt2_{sfx}` | `grad_salt2` |
| `gradrho2_{sfx}` | `grad_rho2` |
| `gradeta2_sfc` | `grad_eta2` |
| `turner_angle_{sfx}` | `turner_angle` |

### `kinematic` (kinematic.zarr)

Same definitions as the surface subset, evaluated in 3D and reduced at
each suffix; `coriolis_f` is an extra (2D) channel.

| Channel | Source / function |
|---|---|
| `relative_vorticity_{sfx}` | `relative_vorticity` |
| `strain_n_{sfx}`, `strain_s_{sfx}`, `strain_mag_{sfx}` | `strain` |
| `divergence_{sfx}` | `divergence` |
| `rossby_number_{sfx}` | `rossby_number` |
| `okubo_weiss_{sfx}` | `okubo_weiss_parameter` |
| `coriolis_f` (extra) | `coriolis_parameter` |

### `frontogenesis` (frontogenesis.zarr)

Same definitions as the surface subset; `ug`/`vg` are surface-only bases
(emit `_sfc` only, since Eta is inherently 2D).

| Channel | Source / function |
|---|---|
| `frontogenesis_tendency_{sfx}` | `frontogenesis_tendency` |
| `frontogenesis_geo_{sfx}` | `frontogenesis_geo` |
| `frontogenesis_ageo_{sfx}` | subset dispatcher (inline) |
| `ug_sfc`, `vg_sfc` | `geostrophic_velocity` |
| `Wstar_{sfx}` | `modified_okubo_weiss` |

### `native_fields` (native_fields.zarr)

Model variables at each depth suffix; `Eta` is surface-only.

| Channel | Source / function |
|---|---|
| `Theta_{sfx}`, `Salt_{sfx}` | model output |
| `Eta_sfc` | model output |
| `U_{sfx}`, `V_{sfx}` | `geographic_velocity` |
| `W_{sfx}` | model output (interpolated to tracer levels) |

### `surface_wind` (surface_wind.zarr) — surface_only

Identical channel list to the SURF `surface_wind` subset (including
`oceQnet`); uses the depth pipeline's S3 access but no 3D machinery.

### `icearea` (icearea.zarr) — surface_only

| Channel | Source / function |
|---|---|
| `SIarea` | model output |

---

## Verification notebooks (planned)

One notebook per subset, verifying each channel against: expected units
and magnitude ranges, known spatial structure (e.g. equatorial NaN bands
for f-normalised fields, sign conventions for vorticity/OW), and — after
the field migration — the expected ~2.6% rescaling of buoyancy-based
dimensional fields under RHO0 = 1000 / g = 9.81.
