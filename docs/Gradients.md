# Gradients on the Native LLC Grid

Reference documentation for `src/dbof/utils/native_gradient.py` — the
single home of all horizontal differencing on the LLC4320 C-grid.
The function docstrings are deliberately short; the reasoning,
geometry, and usage rules live here.

Companions: [Fields.md](Fields.md) (per-channel reference),
`prompts/field_validation.md` (decision log), and the A/B notebooks
(`notebooks/notebooks_dev/field_validation_sparkle.ipynb`,
`notebooks/notebooks_field_validation/surface_fields/turner_angle.ipynb`).

---

## The C-grid, in one paragraph

LLC4320 is an Arakawa C-grid ([MITgcm horizontal-grid
documentation](https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html)):
tracers live at cell **centres** `(j, i)`, `U` on the west cell faces
`(j, i_g)`, `V` on the south faces `(j_g, i)`, and vorticity naturally
on cell **corners** `(j_g, i_g)`.  A finite difference moves a
quantity by half a cell: differencing a centre field lands on a face;
differencing a face field lands on a centre or a corner, depending on
the direction.  Every design rule below follows from that one fact.

## Two stencil families

**1. Vector components — the ECCO recipe** (`rotate_vector_to_geographic`,
`calculate_native_gradient_tracer`, `calculate_jacobian`): difference
on the staggered points, 2-point-interpolate to the tracer centre,
rotate to geographic east/north via the grid `CS`/`SN` coefficients

    u_east  = u·CS − v·SN
    v_north = u·SN + v·CS

This is the standard recipe from the [ECCO v4 tutorial](
https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Gradient_calc_on_native_grid.html)
and is the right tool whenever the **signed vector** is needed:
geographic `U`/`V`, ∇b for frontogenesis, the signed strain
channels, geostrophic shear.

> **Note on wind stress (`oceTAUX`/`oceTAUY`):** in the Arakawa
> C-grid, wind stress acts on horizontal velocities which are
> staggered relative to the tracer cells, with indexing such that
> +oceTAUY(i, j_g) corresponds to +y momentum fluxes at the 'v'
> edge of the tracer cell at (i, j, k=0) (and +oceTAUX(i_g, j) to
> +x fluxes at the 'u' edge).  Also, the model +y direction does
> not necessarily correspond to the geographical north–south
> direction, because the x and y axes of the model's curvilinear
> lat-lon-cap (llc) grid have arbitrary orientations which vary
> within and across tiles.  This is why the `oceTAUX`/`oceTAUY`
> output channels go through `rotate_vector_to_geographic`
> (interp to tracer points + CS/SN rotation) exactly like `U`/`V`,
> and are eastward/northward stress in the stores.
>
> *Provenance:* the note above is adapted from the model output
> metadata (`comments_2` of `oceTAUX`/`oceTAUY`).  Beware that in
> the metadata itself, `oceTAUY`'s `comments_2` carries a
> copy-paste indexing typo — it says `+oceTAUY(i_g, j)`, which
> contradicts its own dims declaration `oceTAUY(time, j_g, i)` and
> `comments_1` ("centered over the 'v' side"); the dims are
> authoritative.
>
> *ECCO-tutorial caveat:* the [ECCO gradient tutorial](
> https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Gradient_calc_on_native_grid.html)
> states that "surface winds/wind stress are located at the grid
> cell centers" — that refers to ECCO's ATMOSPHERIC forcing fields
> (`EXFuwind`/`EXFvwind`), which are indeed cell-centred.  The
> ocean-stress output `oceTAUX`/`oceTAUY` is EDGE-located (see the
> dims above), so the tutorial's closing exercise ("replace
> EXFuwind/EXFvwind with oceTAUX/oceTAUY" in the centre-based curl
> recipe) is misleading if followed naively.  Our
> `wind_stress_curl` does the right thing: it feeds the RAW
> staggered τ into `calculate_jacobian`, whose first step
> interpolates from the u/v edges before rotating and
> differencing.

**2. Squared magnitudes — square/multiply BEFORE interpolating**
(`calculate_grad_squared_tracer`, `calculate_grad_dot_tracer`,
`calculate_native_strain_vorticity` + `interp_corner_squared`):
form squares and products **on** the staggered points where their
factors natively live, and move only the (non-negative) results.

## Why two families: the 'sparkle' mechanism

The 2-point interpolation of the ECCO recipe has a null space at the
grid scale: at a local extremum of a field, the two flanking one-sided
differences are equal-and-opposite and **cancel in the average**.  The
interpolated component is fine for signed uses (a near-zero is an
unremarkable mid-scale value), but *squaring* it manufactures pixels
orders of magnitude below their neighbours — white 'sparkle' on
log-scaled maps, and spurious huge values wherever a squared gradient
sits in a denominator (`R_ib`).  A mean of **non-negative** squares
cannot cancel, so squaring first preserves grid-scale gradient
variance (the variance-preserving form).  Both families are consistent
O(Δx²) discretizations of the same continuum quantities; they differ
only in how they treat the 2Δx scale.  A/B evidence:
`field_validation_sparkle.ipynb`; float32 round-tripping was ruled out
first (store ≡ live to ~1e-7).

## Velocity combinations at their natural points

`calculate_native_strain_vorticity` is eight finite differences,
bundled.  On the C-grid some velocity-derivative combinations can be
computed **without any interpolation** if they are evaluated at the
right spot: differencing U along x lands naturally on cell CENTERS,
so normal strain (∂u/∂x − ∂v/∂y) and divergence (∂u/∂x + ∂v/∂y) are
"free" there (flux form over `rA`); differencing U along y lands on
cell CORNERS, so vorticity (∂v/∂x − ∂u/∂y) and shear strain
(∂v/∂x + ∂u/∂y) are free THERE (circulation form over `rAz`).  The
corner vorticity is MITgcm's own `momVort3` stencil.

`interp_corner_squared` then averages a corner value onto the cell
centres (a 2-point mean in x, then in y).  Use it **after** squaring:
the moved quantity is non-negative and cannot cancel.  Never move
signed corner quantities with it expecting their grid-scale structure
to survive — that reintroduces the null space.

## Basis and rotation rules

`CS`/`SN` rotate the model's horizontal x/y axes into geographic
east/north — a rotation **about the local vertical axis**.  The
native-point outputs behave differently under it (none of them is a
scalar in the general 3D sense — vorticity is a pseudovector, strain
a tensor):

- **vorticity** measures horizontal rotation around the vertical
  axis — the vertical component of the vorticity vector — and is
  unchanged by a rotation about that same axis;
- **divergence** measures horizontal expansion/contraction — the
  trace of the horizontal velocity-gradient tensor — likewise
  unchanged;
- **normal and shear strain** are the deviatoric tensor components:
  any rotation of the axes mixes the two into each other (at angle
  2φ), so the native-point pair is **model-basis** and must NEVER be
  output as the signed strain channels (`strain_n`/`strain_s` come
  from the rotated Jacobian).  Their **sum of squares**, however, is
  invariant under the rotation — which is exactly how `strain_mag`
  and `okubo_weiss` consume them.

The same logic covers the tracer functions: |∇s|² and ∇a·∇b are
rotation-invariant (vector norms and dot products), so
`calculate_grad_squared_tracer` and `calculate_grad_dot_tracer` never
apply `CS`/`SN`; the individual *geographic* component squares cannot
be produced square-first at all (the rotation cross term is destroyed
by squaring — verified empirically: on a rotated face the model-axis
squares appear swapped relative to the geographic ones).

## Products need co-located factors

A product can only be formed where both factors live.  `a²` is always
safe (a factor is co-located with itself).  For two *different*
fields, the same-direction gradient products are co-located
(`a_x·b_x` on the u-points, `a_y·b_y` on the v-points) — that is
`calculate_grad_dot_tracer`, used by the Turner angle so its
numerator and denominator come from one consistent measurement route.
Cross-direction products (e.g. `b_x·b_y` in frontogenesis) have no
native co-location; at best each factor takes one 2-point interp —
such quantities cannot be made fully cancellation-free.

## Consequences to state when comparing channels

- `strain_mag` ≠ `sqrt(strain_n² + strain_s²)` pixelwise (different
  stencils by design).
- `okubo_weiss` uses the corner (`momVort3`) vorticity stencil, not
  the centred `relative_vorticity` channel.
- `turner_angle` (projection form, Johnson et al. 2012 / Whalen &
  Drushka 2025) is built from measured-∇ρ dot products and is
  independent of the `grad*2` channels; `gradrho2` keeps full-EOS ρ
  differencing (the gap to the linearized form is the cabbeling
  content).

*Generated by LH and Claude.*
