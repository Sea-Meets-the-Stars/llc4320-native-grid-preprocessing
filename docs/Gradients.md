# Gradients on the Native LLC Grid

Reference for `src/dbof/utils/native_gradient.py` — all horizontal
differencing on the LLC4320 C-grid.  Companion:
[Fields.md](Fields.md) (per-channel reference).

---

## The C-grid

LLC4320 is an Arakawa C-grid ([MITgcm horizontal-grid
documentation](https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html)):
tracers live at cell **centres** `(j, i)`, `U` on the west cell faces
`(j, i_g)`, `V` on the south faces `(j_g, i)`, and vorticity naturally
on cell **corners** `(j_g, i_g)`.  A finite difference moves a
quantity by half a cell: differencing a centre field lands on a face;
differencing a face field lands on a centre or a corner, depending on
the direction.  

## Gradient / Interpolation artifacts

Any time we interpolate a gradient we run the risk of getting an
artifact.  This comes from the case where we have an extrema at a
given cell, which would cause, for example, slopes on either side of
that cell to be −a / +a.  Any interpolation at this point would
average together these two slopes and produce zero.  Taking the
square then magnifies that artifact.

To combine any two components we need them to exist at the same
location.  So we have four cases:

- **Squares.** If we take the square before we interpolate, we can
  avoid that error entirely (yay, all `grad_squared` components are
  good!) — `calculate_grad_squared_tracer`,
  `calculate_grad_dot_tracer`.
- **Magnitude of a Jacobian.** Without interpolation, different
  Jacobian components live at different parts of the grid.  If we
  square them THERE before interpolating, we can avoid the artifact
  (yay `strain_magnitude` and `okubo_weiss`!) —
  `calculate_native_strain_vorticity` + `interp_corner_squared`.
- **Plain Jacobian** (relative vorticity, divergence): we use the
  ECCO code exactly.  The artifact is still there (interpolation
  happens twice), just not exaggerated by squaring.  I'm assuming it
  is benign if it is what is used by ECCO folks.  We keep this
  version deliberately so that we can store all calculated fields at
  the cell centres — `calculate_jacobian`.
- **Multiplication of two different directional components** (e.g.
  frontogenesis multiplies bx·by): we have to interpolate before
  multiplying them to get them on the same point, so the artifact is
  unavoidable.

Evidence for all of this: `field_validation_sparkle.ipynb`.




# Some downwtream impacts: 

## Strain, vorticity, and rotation

**Why this section exists:** the square-first functions skip the
CS/SN rotation entirely — this explains why that is correct, and
which quantities you therefore must NOT take from them.

`calculate_native_strain_vorticity` gives four combinations, at 
particular locations on the grid:

- **cell centres** — differencing U along its own axis (x): normal
  strain (∂u/∂x − ∂v/∂y) and divergence (∂u/∂x + ∂v/∂y);
- **cell corners** — differencing U across the other axis (y):
  vorticity (∂v/∂x − ∂u/∂y) and shear strain (∂v/∂x + ∂u/∂y).  The
  corner vorticity is the same stencil MITgcm itself uses
  (`momVort3`).

`interp_corner_squared` moves a corner value to the centres by simple
averaging — use it **only after squaring**.

`CS`/`SN` rotate model x/y into geographic east/north (a rotation
about the local vertical).  The ECCO functions apply it
(`u_east = u·CS − v·SN`, `v_north = u·SN + v·CS`); the square-first
functions never need to:

- **vorticity** (spin about the vertical) and **divergence**
  (spreading/squeezing) are the same number in any basis (model vs geo);
- **normal and shear strain** describe stretching along particular
  directions, so turning the map converts one into the other.  The
  native-point pair is therefore **model-basis** and must NEVER be
  output as `strain_n`/`strain_s` (those come from the rotated
  Jacobian) — but their **sum of squares** IS rotation-invariant,
  which is exactly how `strain_mag` and `okubo_weiss` use them;
- likewise |∇s|² and ∇a·∇b.  What you *can't* get square-first is the
  east² and north² parts separately — squaring throws away the
  direction information the rotation needs (verified in
  `field_validation_sparkle.ipynb`: on a rotated face the model-axis
  squares come out swapped relative to the geographic ones).

Consequences: `strain_mag` ≠ `sqrt(strain_n² + strain_s²)` pixelwise,
and `okubo_weiss` uses the corner (`momVort3`) vorticity rather than
the centred `relative_vorticity` channel — different stencils by
design.


## Note on wind stress (`oceTAUX`/`oceTAUY`)

From the model metadata: in the Arakawa C-grid, wind stress acts on
horizontal velocities which are staggered relative to the tracer
cells, with indexing such that +oceTAUY(i, j_g) corresponds to +y
momentum fluxes at the 'v' edge of the tracer cell at (i, j, k=0)
(and +oceTAUX(i_g, j) to +x fluxes at the 'u' edge).  Also, the model
+y direction does not necessarily correspond to the geographical
north–south direction, because the x and y axes of the llc grid have
arbitrary orientations which vary within and across tiles.  This is
why `oceTAUX`/`oceTAUY` go through `rotate_vector_to_geographic` like
`U`/`V`, and are eastward/northward stress in the stores.

Two footnotes on sources:

- `oceTAUY`'s `comments_2` in the model metadata has an
  indexing typo (`+oceTAUY(i_g, j)`), contradicting its own dims
  `oceTAUY(time, j_g, i)` and `comments_1` ("centered over the 'v'
  side").  The dims are authoritative.
- The [ECCO gradient tutorial](
  https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Gradient_calc_on_native_grid.html)
  says wind stress lives at cell centres — that's ECCO's
  *atmospheric* forcing (`EXFuwind`/`EXFvwind`).  `oceTAUX`/`oceTAUY`
  are edge-located, so its closing "just swap in oceTAUX/oceTAUY"
  exercise is misleading.  Our `wind_stress_curl` is fine: it feeds
  the raw staggered τ into `calculate_jacobian`, which interpolates
  from the edges first.

*Generated by LH and Claude.*
