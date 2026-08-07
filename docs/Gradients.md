# Gradients on the Native LLC Grid

Reference documentation for `src/dbof/utils/native_gradient.py` — the
single home of all horizontal differencing on the LLC4320 C-grid.

Companions: [Fields.md](Fields.md) (per-channel reference).

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

## Gradient Artifacts

Averaging/interpolating a signed gradient onto the cell centres can erase real
grid-scale bumps (the slopes on either side of a bump are equal and
opposite, and average to zero).  Whether that erasure ever becomes a
visible artifact — and whether we can do anything about it — depends
entirely on what you are computing:

| What you're computing | Can the artifact be avoided? | What we do |
|---|---|---|
| **Magnitudes and same-direction products** — `grad*2`, `strain_mag`, `okubo_weiss`, `∇a·∇b` | **Yes, completely.** Square/multiply at the staggered points *first*; an average of positive numbers cannot cancel. Any small values that remain are real features. | `calculate_grad_squared_tracer`, `calculate_grad_dot_tracer`, `calculate_native_strain_vorticity` + `interp_corner_squared` |
| **Signed vector components** — geographic `U`/`V`, ζ, δ, `strain_n`/`strain_s`, ∇b for frontogenesis | **No — but it's harmless.** The averaging still smooths grid-scale bumps, but a near-zero in a signed field on a linear colour scale is invisible and does no damage downstream. | The ECCO recipe: `calculate_native_gradient_tracer`, `calculate_jacobian` |
| **Cross-direction products** — e.g. `b_x·b_y` inside frontogenesis | **No — genuinely stuck.** The two factors never share a grid point, so at least one must be averaged (while still signed) before multiplying. The artifact can be minimized, never removed. | Component path for now (watch-list; see `prompts/field_validation.md`) |

Everything below is the detail behind this table.

## The two recipes: signed components vs squared magnitudes

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
gradients natively live, and move only the already-squared (non-negative) 
results to the cell centers.

## Why two recipes: the 'sparkle' mechanism

Interpolating a value from the staggered points to a cell center means averaging
the two neighboring values. This can become problematic if one is sitting at 
a local minima or maxima in a field. This leads to a case where, for example,
the slope in the left side is +a, the slope on the right side is -a, and averaging
the two gives zero. This erases a real gradient that exists on either side 
of the cell. For a signed field this hardly matters: one near-zero pixel among many mid-range values is
invisible. But square that averaged value and the erasure
becomes glaring — the pixel is now orders of magnitude smaller than
its neighbours, which shows up as white speckle ('sparkle') on a
log-scale map, and as absurdly large values anywhere a squared
gradient sits in a denominator (R_ib).

Squaring first avoids the trap entirely: (+a)² and (−a)² are both
positive, and an average of positive numbers can't cancel to zero.
That's the whole fix. Both orders of operation are legitimate
approximations of the same physics — they only disagree about
features at the very smallest (two-grid-cell) scale, which the
average-first version silently erases and the square-first version
keeps. Evidence: field_validation_sparkle.ipynb (side-by-side
A/B); float32 precision was ruled out as the cause first (store and
full-precision recompute agree to ~1e-7).

## Velocity combinations at their natural points

`calculate_native_strain_vorticity` is just eight finite
differences, bundled. The trick is where each one is evaluated.
On the C-grid, taking a difference moves you half a cell — so if
you pick the right combination, the result lands exactly on a grid
point and no interpolation is needed at all:
  - differencing U along its own axis (x) lands on the cell
  centres — so normal strain (∂u/∂x − ∂v/∂y) and divergence
  (∂u/∂x + ∂v/∂y) come out there for free;
  - differencing U across the other axis (y) lands on the cell
  corners — so vorticity (∂v/∂x − ∂u/∂y) and shear strain
  (∂v/∂x + ∂u/∂y) come out there for free. The corner vorticity
  is the same stencil MITgcm itself uses (momVort3).

`interp_corner_squared` then moves a corner value to the cell
centres by simple averaging. Use it only after squaring — an
average of positive numbers can't cancel. Moving a signed corner
quantity with it walks straight back into the trap above.

## Basis and rotation rules

`CS`/`SN` rotate the model's horizontal x/y axes into geographic
east/north — a rotation **about the local vertical axis**.   
The four native-point outputs react differently to that rotation:

- **vorticity** measures horizontal rotation around the vertical
  axis — the vertical component of the vorticity vector — and is
  unchanged by a rotation about that same axis (spinning doesn't 
  care which way your map is turned — same number in any basis.);
- **divergence** measures horizontal expansion/contraction — the
  trace of the horizontal velocity-gradient tensor, or how fast 
  the water spreads apart or squeezes together — likewise
  unchanged;
- **normal and shear strain** describe stretching along particular
  directions — turn the map and what looked like "stretching
  north–south" becomes partly "shearing", and vice versa.
  So the native-point pair is **model-basis** and must NEVER be
  output as the signed strain channels (`strain_n`/`strain_s` come
  from the rotated Jacobian).  The total amount of stretching (the 
  **sum of squares**), however, is invariant under the rotation — 
  which is exactly how `strain_mag` and `okubo_weiss` consume them.

The same logic covers the tracer functions: the length of a gradient
vector (|∇s|²) and their projection/dot products (∇a·∇b) are
rotation-invariant, so
`calculate_grad_squared_tracer` and `calculate_grad_dot_tracer` never
apply `CS`/`SN`. What you cannot get from the square-first route is
the individual *geographic* component squares (east2, north2) separately, 
as squaring throws away direction information needed by the rotation.
(This is verified empirically in `field_validation_sparkles.ipynb`, where
the model-axis squares come out squapped relative to the geographic ones
on a rotated face).

## Products need co-located factors

You can only multiply two numbers that live at the same locations.  
`a²` is always safe (a factor is co-located with itself).  For two *different*
fields, the same-direction gradient products are co-located
(`a_x·b_x` on the u-points, `a_y·b_y` on the v-points) — that is
`calculate_grad_dot_tracer`, used by the Turner angle so its
numerator and denominator come from one consistent measurement route.
Cross-direction products (e.g. `b_x·b_y` in frontogenesis) have no
native co-location; at best at least one must be averaged/interpolated
before multiplying. These quantities can never be made fully cancellation-free.

## Consequences to state when comparing channels

- `strain_mag` ≠ `sqrt(strain_n² + strain_s²)` pixelwise (different
  stencils by design). 
    - strain_mag: Normal strain lives on cell centres (differencing U along its own
      axis lands there — ZERO interpolation); shear strain lives on
      cell corners.  The shear is squared AT the corners, and only
      the non-negative square is moved to the centres (a mean of
      non-negatives cannot cancel). 
    - Signed geographic components, strain_n² & strain_s²: ECCO path 
      (vectors need the rotation; near-zeros are benign on signed fields).
- `okubo_weiss` uses the corner (`momVort3`) vorticity stencil, not
  the centred `relative_vorticity` channel.
    - normal strain on cell centres (zero interpolation); 
      vorticity and shear strain on cell corners (MITgcm's own momVort3 stencil); 
      each squared AT its native point, and only the non-negative squares are
      moved corner -> centre.
- `turner_angle` (projection form, Johnson et al. 2012 / Whalen &
  Drushka 2025) is built from measured-∇ρ dot products and is
  independent of the `grad*2` channels; `gradrho2` keeps full-EOS ρ
  differencing (the gap to the linearized form is the cabbeling
  content).

*Generated by LH and Claude.*
