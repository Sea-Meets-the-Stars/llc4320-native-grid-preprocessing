"""
Depth-resolved derived diagnostics for LLC4320 data.

This module computes physical properties that require the full water column
(3D fields) and then reduces them to 2D outputs at specific depth definitions.

Design
------
Two orthogonal abstractions are combined:

1. **Field functions** — each computes a single 3D diagnostic from the merged
   dataset (e.g., buoyancy frequency squared, vertical shear magnitude).
   Signature: ``(ds_merge, grid=None, **intermediates) -> xr.DataArray``

2. **Depth-selection strategies** — each takes a 3D field and returns a 2D
   slice or reduction at a particular depth definition.
   Four strategies:

   - ``surface``      : z = 0  (k = 0)
   - ``fixed_depth``  : nearest model level to 25 m
   - ``at_mld``       : nearest model level to the mixed-layer depth
   - ``mld_mean``     : thickness-weighted mean over 0 ≤ z ≤ MLD

   Signature: ``(field3d, ds_merge, mld=None) -> xr.DataArray``

Fields marked with ``*`` in the project spec are evaluated at all four depth
definitions.  Fields that are inherently 2D (MLD, Burger number, wind stress
curl, Ekman pumping) bypass the depth dispatch entirely.

MITgcm / xmitgcm coordinate conventions
-----------------------------------------
Tracer fields (Theta, Salt, buoyancy, density) live on:
    ``('k', 'face', 'j', 'i')``  with vertical coordinate ``Z``

Zonal velocity (U) lives on:
    ``('k', 'face', 'j', 'i_g')``  with vertical coordinate ``Z``

Meridional velocity (V) lives on:
    ``('k', 'face', 'j_g', 'i')``  with vertical coordinate ``Z``

Vertical velocity (W) lives on:
    ``('k_l', 'face', 'j', 'i')``  with vertical coordinate ``Zl``

Vertical coordinates:
    Z   — depth of tracer cell centres (negative upward in MITgcm)
    Zl  — depth of lower cell interface (where W lives)
    Zu  — depth of upper cell interface
    Zp1 — depth at cell interfaces (nk+1 levels)

This module is explicit about which vertical coordinate is used for each
operation.  Vertical derivatives and interpolations always attach the
appropriate 1D depth coordinate before operating.
"""

import logging

import numpy as np
import xarray as xr
import dask

import dbof.utils.physical_calculations as physical_calculations
import dbof.utils.native_gradient as ng

# Re-use surface-level functions that are still in calculate_additional_fields
from dbof.preprocessing.calculate_additional_fields import (
    coriolis_parameter,
    relative_vorticity,
    rossby_number,
)
import dbof.preprocessing.calculate_additional_fields as caf

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO0 = 1000.0   # kg m⁻³  reference density
G    = 9.81      # m s⁻²  gravitational acceleration
CP   = 3994.0    # J kg⁻¹ K⁻¹  seawater specific heat capacity
OMEGA_EARTH = 7.292115e-5  # rad s⁻¹


# ---------------------------------------------------------------------------
# xgcm / dask materialisation helper
# ---------------------------------------------------------------------------
# xgcm's grid.interp / grid.diff use dask ``map_overlap`` with LLC face
# connections.  The ``adjust_chunks`` parameter that xgcm passes to dask's
# ``blockwise`` is hard-coded for a single face block, and dask validates
# this at **graph-construction time** — before any data is read.  A lazy
# ``da.chunk({"face": -1})`` rechunk is not sufficient because xgcm's
# internal bookkeeping still sees a mismatch.
#
# The reliable workaround (same as the surface pipeline in
# generate_global_depth_dask.py) is to eagerly materialise any array that
# will be passed to ``grid.interp``, ``grid.diff``, or native-gradient
# helpers that call them.  With the ~750 GB RAM available on the compute
# node this is feasible even for full 3D fields.
# ---------------------------------------------------------------------------

def _materialise_for_xgcm(da):
    """Eagerly load *da* into memory so xgcm grid ops work.

    xgcm's ``map_overlap`` with LLC face connections cannot build a valid
    dask task graph when the ``face`` dimension is chunked — even if
    rechunked to a single block.  Materialising to numpy removes dask
    chunking entirely, which is what xgcm expects.

    No-op for arrays that are already numpy-backed.
    """
    if hasattr(da, "chunks") and da.chunks is not None:
        return da.compute()
    return da


# ===========================================================================
#  DENSITY
# ===========================================================================
# Density is computed via physical_calculations.density_of_field, which uses
# xr.apply_ufunc(jmd95, ..., dask="parallelized").  This processes the JMD95
# polynomial one chunk at a time, confining float64 intermediates to ~2 GB
# per chunk rather than the full array.
#
# _density_lazy wraps the call to avoid .persist() so the result stays fully
# lazy for graph-building; the final dask.compute() in each subset function
# materialises everything at once.
# ===========================================================================


def _density_lazy(ds_merge):
    """Compute density lazily via apply_ufunc (no .persist()).

    Wraps physical_calculations.density_of_field but ensures the result
    stays dask-backed and lazy — no eager scheduling on workers.  This is
    important for building a single fused task graph that dask.compute()
    materialises all at once at the end of each subset function.

    For numpy-backed inputs (per-face eager path), this still works
    correctly — apply_ufunc falls back to direct numpy execution.
    """
    # Ensure sensible chunking for 3D data (k=-1 keeps full water column
    # in one chunk, matching the on-disk layout of the S3 timestep stores).
    chunk_spec = {'face': 1, 'j': 720, 'i': 720}
    for dim in ('k', 'k_l'):
        if dim in ds_merge.dims:
            chunk_spec[dim] = -1
    ds = ds_merge.chunk(chunk_spec)

    import dbof.utils.jmd95_xgcm_implementation as jmd95
    p = xr.zeros_like(ds.Theta)

    rho = xr.apply_ufunc(
        jmd95.jmd95,
        ds.Salt,
        ds.Theta,
        p,
        dask="parallelized",
        output_dtypes=[float],
    )
    return rho


# ===========================================================================
#  VERTICAL HELPERS
# ===========================================================================
# These helpers are careful about which vertical dimension and coordinate
# they use.  Functions that operate on tracer-point fields (Theta, Salt,
# density, buoyancy, N²) use ``Z`` / ``k``.  Functions that operate on
# W-point fields use ``Zl`` / ``k_l``.  Velocity fields (U, V) also live
# on ``Z`` / ``k`` vertically, but are horizontally staggered — the caller
# is responsible for interpolating to tracer points when needed.
# ===========================================================================

def _get_vertical_dim(da_in):
    """Return the name of the vertical dimension present in *da_in*."""
    for dim in ("k", "k_l", "k_u", "k_p1", "Z", "Zl", "Zu", "Zp1",
                "depth", "lev"):
        if dim in da_in.dims:
            return dim
    raise ValueError(
        f"Could not identify vertical dimension from dims={da_in.dims}"
    )


def _get_depth_coord(ds_merge, zdim=None):
    """
    Return a 1D depth coordinate in metres, **positive downward**.

    For tracer-level fields (zdim='k') this returns ``|Z|``.
    For W-level fields (zdim='k_l') this returns ``|Zl|``.

    LLC4320 specific — expects ``Z`` or ``Zl`` to exist in *ds_merge*.
    """
    if zdim is None:
        zdim = _get_vertical_dim(ds_merge.Theta)

    coord_name = {"k": "Z", "k_l": "Zl"}[zdim]
    z = ds_merge[coord_name]

    # xmitgcm stores Z with dim='Z' but data variables use dim='k'.
    # Rename so xarray aligns element-wise along the vertical.
    if z.dims[0] != zdim:
        z = z.rename({z.dims[0]: zdim})

    # MITgcm stores depth as negative-upward; convert to positive-downward.
    if np.nanmean(z.values) < 0:
        z = -z

    return z


def _get_vertical_spacing(ds_merge, zdim=None):
    """
    Layer thickness [m] appropriate for the vertical grid.

    - For tracer levels (zdim='k'):  returns ``drF`` — the thickness of
      each tracer cell.
    - For W levels (zdim='k_l'):  returns ``drC`` — the distance between
      adjacent tracer cell centres (i.e. the spacing of W levels).

    Falls back to finite-differencing the depth coordinate if the model
    spacing variable is unavailable.
    """
    if zdim is None:
        zdim = _get_vertical_dim(ds_merge.Theta)

    # Choose the correct MITgcm spacing variable for this vertical grid.
    _SPACING_MAP = {"k": "drF", "k_l": "drC"}
    spacing_var = _SPACING_MAP.get(zdim, "drF")

    if spacing_var in ds_merge:
        dz = ds_merge[spacing_var]
        if zdim not in dz.dims:
            dz = dz.rename({dz.dims[0]: zdim})
        return dz

    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz_vals = np.gradient(z.values.astype(float))
    return xr.DataArray(dz_vals, dims=(zdim,), coords={zdim: z})


def _nearest_k_to_depth(z_vals, target_depth):
    """Return the integer k-index whose depth is nearest to *target_depth*.

    Parameters
    ----------
    z_vals : np.ndarray
        1D depth coordinate, positive downward, in metres (same convention
        as returned by ``_get_depth_coord``).
    target_depth : float
        Target depth in metres, positive downward.

    Returns
    -------
    int
        Index into *z_vals* closest to *target_depth*.
    """
    return int(np.abs(z_vals - float(target_depth)).argmin())


def _vertical_derivative(field, ds_merge):
    """
    Compute d(field)/dz on the field's own vertical grid.

    Positive z is downward (so dθ/dz < 0 means θ decreases with depth).

    Uses centered finite differences at interior levels and one-sided
    differences at boundaries, matching MITgcm's discretisation style.
    The spacing between levels comes from the positive-downward depth
    coordinate (``_get_depth_coord``).

    Implemented via ``apply_ufunc`` with ``dask="parallelized"`` so the
    derivative is computed per-chunk without materialising the full array.
    """
    zdim = _get_vertical_dim(field)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)  # 1D, always small
    nk = len(z_vals)
    k_axis = field.dims.index(zdim)

    # Pre-compute spacing arrays for the 1D depth coordinate.
    # Interior:  dz_centered[k] = z[k+1] - z[k-1]
    # Forward:   dz_fwd        = z[1]    - z[0]
    # Backward:  dz_bwd        = z[-1]   - z[-2]
    dz_centered = z_vals[2:] - z_vals[:-2]                   # length nk-2
    dz_fwd = z_vals[1] - z_vals[0]
    dz_bwd = z_vals[-1] - z_vals[-2]

    def _deriv_along_k(f_chunk):
        f = f_chunk.astype(np.float64)

        # Slicing helper for arbitrary axis position
        def _sl(start, stop):
            s = [slice(None)] * f.ndim
            s[k_axis] = slice(start, stop)
            return tuple(s)

        out = np.empty_like(f)

        # Forward difference at k=0
        out[_sl(0, 1)] = (
            (f[_sl(1, 2)] - f[_sl(0, 1)]) / dz_fwd
        )
        # Centered differences at interior levels
        # Broadcast dz_centered to the right shape for the k_axis
        dz_shape = [1] * f.ndim
        dz_shape[k_axis] = nk - 2
        dz_bc = dz_centered.reshape(dz_shape)
        out[_sl(1, -1)] = (
            (f[_sl(2, None)] - f[_sl(None, -2)]) / dz_bc
        )
        # Backward difference at k=N-1
        out[_sl(-1, None)] = (
            (f[_sl(-1, None)] - f[_sl(-2, -1)]) / dz_bwd
        )
        return out

    return xr.apply_ufunc(
        _deriv_along_k,
        field,
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _select_at_depth(field, target_depth, ds_merge):
    """Select the nearest model level to *target_depth* [m, positive downward].

    Uses ``_nearest_k_to_depth`` to find the closest k-index, then returns
    a 2D slice via ``.isel()``.  No interpolation — this returns the value
    at the discrete model level closest to the requested depth.
    """
    zdim = _get_vertical_dim(field)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)
    k_idx = _nearest_k_to_depth(z_vals, target_depth)
    return field.isel({zdim: k_idx})


def _extract_at_mld(field3d, mld, ds_merge):
    """Extract a 3D field at per-column MLD depths via nearest-k indexing.

    ``mld`` is a 2D array (face, j, i) giving a different depth per column.
    For each column, the nearest model level to that column's MLD is found
    and the field value at that level is returned.  No vertical interpolation
    is performed — MLD itself is defined as a discrete model-level depth
    (the deepest Z where the density criterion holds), so nearest-k
    recovers the exact level.

    Implementation note: ``apply_ufunc`` with ``input_core_dims=[[zdim], []]``
    moves the vertical axis to the **last** position of the chunk before
    calling the inner function.  The inner function therefore receives
    ``field_chunk`` with shape ``(..., nk)`` and ``mld_chunk`` with the
    spatial shape ``(...)``.  We work with the k-axis in last position
    throughout (no moveaxis needed).
    """
    zdim = _get_vertical_dim(field3d)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)  # 1D, small — captured in closure

    def _nearest_k_at_mld_chunk(field_chunk, mld_chunk):
        # field_chunk: (..., nk) — apply_ufunc moved zdim to last axis
        # mld_chunk:   (...)     — 2D spatial
        spatial_shape = mld_chunk.shape
        mld_flat = mld_chunk.ravel().astype(np.float64)

        # Nearest k-index for each column
        # z_vals is (nk,), mld_flat is (n_cols,) → broadcast to (n_cols, nk)
        k_idx = np.abs(z_vals[np.newaxis, :] - mld_flat[:, np.newaxis]).argmin(axis=1)

        # Flatten the spatial dims of field_chunk for advanced indexing
        f_flat = field_chunk.reshape(-1, field_chunk.shape[-1])  # (n_cols, nk)
        col_idx = np.arange(f_flat.shape[0])
        result_flat = f_flat[col_idx, k_idx]
        return result_flat.reshape(spatial_shape).astype(np.float32)

    return xr.apply_ufunc(
        _nearest_k_at_mld_chunk,
        field3d,
        mld,
        input_core_dims=[[zdim], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )


def _interp_w_to_tracer_levels(ds_merge):
    """
    Interpolate W from cell interfaces (k_p1 / Zp1) to tracer cell centres
    (k / Z) by averaging adjacent interface levels.

    On the MITgcm C-grid, ``grid.interp(W, 'Z')`` is equivalent to
    ``0.5 * (W[k] + W[k+1])`` along the vertical.  This avoids fragile
    coordinate-based ``xr.interp`` + rename gymnastics that break across
    xarray versions.
    """
    zdim_w = _get_vertical_dim(ds_merge.W)
    zdim_t = _get_vertical_dim(ds_merge.Theta)
    n_t = ds_merge.Theta.sizes[zdim_t]

    W = ds_merge.W
    W_upper = W.isel({zdim_w: slice(0, n_t)}).drop_vars(
        [c for c in W.coords if c == zdim_w or c == zdim_t],
        errors="ignore",
    )
    W_lower = W.isel({zdim_w: slice(1, n_t + 1)}).drop_vars(
        [c for c in W.coords if c == zdim_w or c == zdim_t],
        errors="ignore",
    )

    # Align dimensions: rename the sliced zdim_w → zdim_t so the arrays
    # can be added element-wise.
    W_upper = W_upper.rename({zdim_w: zdim_t})
    W_lower = W_lower.rename({zdim_w: zdim_t})

    return 0.5 * (W_upper + W_lower)


def _masked_ml_mean(field3d, mld, ds_merge):
    """
    Thickness-weighted mean of a 3D field over 0 ≤ z ≤ MLD.

    This is the standard depth-reduction for "mean over the mixed layer".
    """
    zdim = _get_vertical_dim(field3d)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz = _get_vertical_spacing(ds_merge, zdim=zdim)

    mask = z <= mld
    num = (field3d * dz).where(mask).sum(dim=zdim)
    den = dz.where(mask).sum(dim=zdim)
    return num / den


# ===========================================================================
#  DEPTH-SELECTION STRATEGIES
# ===========================================================================
# Each strategy takes a 3D field and returns a 2D reduction.
# They all share the signature:
#     (field3d, ds_merge, mld=None, fixed_depth_m=25.0) -> xr.DataArray
# ===========================================================================

FIXED_DEPTH_M = 25.0   # default fixed depth (≈ k=3 for LLC4320)


def select_surface(field3d, ds_merge, mld=None, fixed_depth_m=None):
    """Select z = 0 (surface).  Uses k=0 / k_l=0."""
    zdim = _get_vertical_dim(field3d)
    return field3d.isel({zdim: 0})


def select_fixed_depth(field3d, ds_merge, mld=None,
                       fixed_depth_m=FIXED_DEPTH_M):
    """Select the nearest model level to a fixed depth (default 25 m)."""
    return _select_at_depth(field3d, fixed_depth_m, ds_merge)


def select_at_mld(field3d, ds_merge, mld=None, fixed_depth_m=None):
    """Select the nearest model level to the mixed-layer depth."""
    if mld is None:
        raise ValueError("select_at_mld requires mld to be precomputed.")
    return _extract_at_mld(field3d, mld, ds_merge)


def select_mld_mean(field3d, ds_merge, mld=None, fixed_depth_m=None):
    """Thickness-weighted mean over 0 ≤ z ≤ MLD."""
    if mld is None:
        raise ValueError("select_mld_mean requires mld to be precomputed.")
    return _masked_ml_mean(field3d, mld, ds_merge)


# Ordered mapping: suffix → strategy function.
DEPTH_STRATEGIES = {
    "sfc":      select_surface,
    "z25m":     select_fixed_depth,
    "mld":      select_at_mld,
    "mld_mean": select_mld_mean,
}

# Partition into point-depth strategies (can be satisfied from a single
# k-level) and column strategies (need the full vertical column).
POINT_STRATEGIES = {
    "sfc":  select_surface,
    "z25m": select_fixed_depth,
}
COLUMN_STRATEGIES = {
    "mld":      select_at_mld,
    "mld_mean": select_mld_mean,
}

def _get_point_k_indices(ds_merge):
    """Return ``{suffix: k_index}`` for each point strategy."""
    zdim = _get_vertical_dim(ds_merge.Theta)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)
    return {
        "sfc":  0,
        "z25m": _nearest_k_to_depth(z_vals, FIXED_DEPTH_M),
    }


def apply_depth_strategies(field3d, field_base_name, ds_merge, mld=None,
                           field_at_k=None, requested=None):
    """
    Apply depth strategies to a 3D field and return a dict of named 2D
    outputs.

    Two modes of operation
    ----------------------
    **Eager mode** (``field_at_k is None``, the default):
        The caller supplies a complete 3D field.  All four depth strategies
        are applied directly — identical to the original behaviour.

    **Lazy / memory-optimised mode** (``field_at_k`` is a callable):
        ``field_at_k(k_index) -> xr.DataArray`` returns a 2D slice of the
        field at a single depth level.  This is the memory-saving path:

        * **Point strategies** (sfc, z25m) call ``field_at_k`` once each,
          materialising only the 2D slice they need.
        * **Column strategies** (mld, mld_mean) call ``field_at_k`` at every
          k-level, concatenate the results into a 3D array, and then apply
          the column reduction.  Only two 2D slices are live at any time
          during the loop; the full 3D is assembled in float32 at the end.

        If both a ``field3d`` and a ``field_at_k`` callable are provided,
        ``field_at_k`` is used for point strategies and ``field3d`` is used
        for column strategies.  This lets callers avoid assembling the 3D
        field when only point channels are requested.

    Parameters
    ----------
    field3d : xr.DataArray or None
        Complete 3D field.  Required for column strategies unless
        ``field_at_k`` is provided.
    field_base_name : str
        Base name for the output keys (e.g. ``"N2"``).
    ds_merge : xr.Dataset
    mld : xr.DataArray or None
    field_at_k : callable or None
        ``(k_index: int) -> xr.DataArray``  —  returns the field at a
        single depth level.  When provided, point strategies use this
        instead of indexing into ``field3d``.
    requested : set[str] or None
        If given, only channels in this set are computed.  This avoids
        building the 3D field for column strategies when no column channels
        are requested.  If *None*, all four channels are computed.

    Returns
    -------
    dict[str, xr.DataArray]
        Keys are ``{field_base_name}_{suffix}``.
    """
    if requested is None:
        requested_suffixes = set(DEPTH_STRATEGIES)
    else:
        requested_suffixes = {
            s for s in DEPTH_STRATEGIES
            if f"{field_base_name}_{s}" in requested
        }

    results = {}

    # ------------------------------------------------------------------
    # Point strategies
    # ------------------------------------------------------------------
    point_suffixes = requested_suffixes & set(POINT_STRATEGIES)
    if point_suffixes:
        if field_at_k is not None:
            k_map = _get_point_k_indices(ds_merge)
            for suffix in point_suffixes:
                key = f"{field_base_name}_{suffix}"
                results[key] = field_at_k(k_map[suffix])
        elif field3d is not None:
            for suffix in point_suffixes:
                key = f"{field_base_name}_{suffix}"
                results[key] = POINT_STRATEGIES[suffix](
                    field3d, ds_merge, mld=mld)
        # else: nothing to do (no source)

    # ------------------------------------------------------------------
    # Column strategies
    # ------------------------------------------------------------------
    col_suffixes = requested_suffixes & set(COLUMN_STRATEGIES)
    if col_suffixes:
        # Assemble field3d from field_at_k if not already provided.
        if field3d is None and field_at_k is not None:
            zdim = _get_vertical_dim(ds_merge.Theta)
            nk = ds_merge.Theta.sizes[zdim]
            logging.info(
                f"  {field_base_name}: assembling 3D from per-k callable "
                f"({nk} levels) for column strategies"
            )
            levels = [field_at_k(k).astype(np.float32) for k in range(nk)]
            field3d = xr.concat(levels, dim=zdim)
            del levels

        if field3d is not None:
            for suffix in col_suffixes:
                key = f"{field_base_name}_{suffix}"
                results[key] = COLUMN_STRATEGIES[suffix](
                    field3d, ds_merge, mld=mld)

    return results


# ===========================================================================
#  GROUP 1: STRATIFICATION & HEAT
# ===========================================================================

MLD_REFERENCE_DEPTH_M = 10.0  # metres — Bodner et al. reference depth (≈ 9.66 m)


def mixed_layer_depth(ds_merge, density_threshold=0.03,
                      ref_depth_m=MLD_REFERENCE_DEPTH_M):
    """
    Mixed-layer depth following Bodner et al.:

        σ₀ = ρ(S, Θ, p=0) − 1000
        Δσ = σ₀ − σ₀(k_ref)             # k_ref = nearest level to ref_depth_m
        MLD = deepest Z where Δσ ≤ threshold

    The reference level is determined by finding the nearest model k-index
    to ``ref_depth_m`` (default 10 m ≈ k=6 on the LLC4320 grid), rather
    than hard-coding a k-index.

    Uses pure xarray operations so the computation stays fully lazy
    when inputs are dask-backed.

    Returns a 2D field (positive metres, no depth dispatch needed).
    """
    rho = _density_lazy(ds_merge)
    zdim = _get_vertical_dim(rho)

    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)
    ref_k = _nearest_k_to_depth(z_vals, ref_depth_m)

    sigma0 = rho - 1000.0
    sigma0_ref = sigma0.isel({zdim: ref_k})
    delta_sigma = sigma0 - sigma0_ref

    # z is already positive downward (from _get_depth_coord above).
    # xarray broadcasts the 1D z against the 4D delta_sigma automatically.

    # Mask out levels outside the mixed layer, then take the max
    # (deepest = largest positive-downward Z) along the vertical axis.
    z_masked = z.where(delta_sigma <= density_threshold)
    mld = z_masked.max(dim=zdim, skipna=True)

    mld.name = "MLD"
    mld.attrs["long_name"] = "Mixed layer depth"
    mld.attrs["units"] = "m"
    return mld


def buoyancy_frequency_squared_3d(ds_merge):
    """
    Full 3D profile of N² = (g/ρ₀) dρ/dz.

    With the positive-downward z convention used throughout this module,
    density increases with depth in a stably stratified ocean, so
    dρ/dz > 0 and N² > 0 without a leading negative sign.

    (In the traditional negative-upward convention N² = -(g/ρ₀) dρ/dz,
    but the sign flip in z absorbs the minus sign.)

    Computed on tracer levels (dim='k', coord='Z').
    """
    rho = _density_lazy(ds_merge)
    drho_dz = _vertical_derivative(rho, ds_merge)

    n2 = (G / RHO0) * drho_dz
    n2.name = "N2"
    n2.attrs["long_name"] = "Buoyancy frequency squared"
    n2.attrs["units"] = "s^-2"
    return n2


def mixed_layer_heat_content(ds_merge, mld=None):
    """
    Mixed-layer heat content integrated over the mixed layer.

    Q_ml = ∫₀ᴹᴸᴰ cp ρ₀ Θ dz   [J m⁻²]

    This is a 2D field (no depth dispatch — it is inherently an MLD integral).
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)

    theta = ds_merge.Theta
    integrand = CP * RHO0 * theta

    # True vertical integral: sum of (field × dz) over 0 ≤ z ≤ MLD
    zdim = _get_vertical_dim(integrand)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz = _get_vertical_spacing(ds_merge, zdim=zdim)
    mask = z <= mld

    out = (integrand * dz).where(mask).sum(dim=zdim)
    out.name = "ml_heat_content"
    out.attrs["long_name"] = "Mixed-layer heat content (integrated)"
    out.attrs["units"] = "J m^-2"
    return out


# ===========================================================================
#  GROUP 2: SHEAR & DIMENSIONLESS NUMBERS
# ===========================================================================

def vertical_shear_components_3d(ds_merge, grid, k_index=None):
    """
    du/dz and dv/dz on tracer levels in geographic coordinates.

    The vertical derivative is computed **lazily** on the staggered grids
    (via ``_vertical_derivative`` / ``apply_ufunc``), which does *not*
    require materialisation.  Materialisation for ``grid.interp`` is then
    scoped to the minimum needed:

    * ``k_index is None`` (default) — full 3D.  Both staggered derivatives
      are materialised, interpolated to tracer points, and rotated.  This
      is the path used by ``ertel_pv_terms_3d`` which needs the actual
      geographic components at every depth.

    * ``k_index = int`` — only a single depth level is materialised as a
      2D slice (~1 GB), interpolated, and rotated.  Used by
      ``compute_vertical_shear`` for point-depth strategies (sfc, z25m)
      where materialising the full 3D field is unnecessary.

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    k_index : int or None
        If given, only this k-level is materialised and returned (2D).
        If *None*, the full 3D field is materialised and returned.

    Returns
    -------
    uz, vz : xr.DataArray
        Geographic zonal and meridional vertical shear on tracer points [s⁻¹].
        Shape is 2D ``(face, j, i)`` when *k_index* is set, otherwise
        3D ``(k, face, j, i)``.
    """
    zdim = _get_vertical_dim(ds_merge.U)  # 'k'

    # 1. Vertical derivatives — lazy on the staggered grids.
    dUdz_stag = _vertical_derivative(ds_merge.U, ds_merge)   # (k, face, j, i_g)
    dVdz_stag = _vertical_derivative(ds_merge.V, ds_merge)   # (k, face, j_g, i)

    # 2. Select depth scope, materialise, interpolate to tracer points.
    if k_index is not None:
        dUdz_2d = _materialise_for_xgcm(dUdz_stag.isel({zdim: k_index}))
        dVdz_2d = _materialise_for_xgcm(dVdz_stag.isel({zdim: k_index}))
        uz_model = grid.interp(dUdz_2d, 'X', boundary='fill')
        vz_model = grid.interp(dVdz_2d, 'Y', boundary='fill')
    else:
        # Lazy path — no materialise.  calculate_jacobian in
        # native_gradient.py proves xgcm can operate on dask arrays.
        # Callers that need numpy should use k_index or .compute()
        # on the returned arrays.
        uz_model = grid.interp(dUdz_stag, 'X', boundary='fill')
        vz_model = grid.interp(dVdz_stag, 'Y', boundary='fill')

    # 3. Rotate model basis → geographic (zonal/meridional).
    uz = uz_model * ds_merge['CS'] - vz_model * ds_merge['SN']
    vz = uz_model * ds_merge['SN'] + vz_model * ds_merge['CS']

    uz.name = "u_z"
    vz.name = "v_z"
    uz.attrs["units"] = "s^-1"
    vz.attrs["units"] = "s^-1"
    return uz, vz


def vertical_shear_magnitude_3d(ds_merge, grid):
    """
    3D profile of |S| = sqrt(u_z² + v_z²) [s⁻¹].

    All quantities on tracer points.
    """
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear = np.sqrt(uz**2 + vz**2)
    shear.name = "vertical_shear"
    shear.attrs["units"] = "s^-1"
    return shear


def richardson_number_3d(ds_merge, grid):
    """
    Local Richardson number Ri(z) = N² / (u_z² + v_z²).

    Computed on tracer levels.  Where shear² = 0, Ri is set to NaN.
    """
    n2 = buoyancy_frequency_squared_3d(ds_merge)
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear2 = uz**2 + vz**2
    ri = xr.where(shear2 > 0, n2 / shear2, np.nan)
    ri.name = "Ri"
    ri.attrs["units"] = "1"
    return ri


def froude_number_3d(ds_merge, grid, mld=None, k_index=None):
    """
    Froude number Fr(z) = speed(z) / (N(z) * H_ml).

    U and V are interpolated to tracer points before computing speed.
    N is taken as sqrt(|N²|) at each level.  H_ml is the mixed-layer
    depth (a constant per water column).

    Parameters
    ----------
    k_index : int or None
        If given, only this k-level is materialised (2D output).
        If *None*, the result stays lazy (3D dask-backed).
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)

    n2 = buoyancy_frequency_squared_3d(ds_merge)
    zdim = _get_vertical_dim(ds_merge.U)

    if k_index is not None:
        # Per-k path: materialise only one depth level (~1 GB each).
        n2_zdim = _get_vertical_dim(n2)
        n2_k = n2.isel({n2_zdim: k_index}).compute()
        n_abs = np.sqrt(np.abs(n2_k))
        U_k = _materialise_for_xgcm(ds_merge.U.isel({zdim: k_index}))
        V_k = _materialise_for_xgcm(ds_merge.V.isel({zdim: k_index}))
        U_c = grid.interp(U_k, 'X', boundary='fill')
        V_c = grid.interp(V_k, 'Y', boundary='fill')
    else:
        # Lazy path — no materialise.
        n_abs = np.sqrt(np.abs(n2))
        U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
        V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')

    speed = np.sqrt(U_c**2 + V_c**2)

    denom = n_abs * mld
    fr = xr.where(denom > 0, speed / denom, np.nan)
    fr.name = "Fr"
    fr.attrs["units"] = "1"
    return fr


def rossby_number_3d(ds_merge, grid, k_index=None):
    """
    3D Rossby number profile: ζ(z) / f.

    Relative vorticity is computed from horizontal velocity at each depth
    level using the native-grid Jacobian.  Coriolis f is depth-independent.

    NOTE: ``calculate_jacobian`` expects staggered U/V on the LLC grid at
    each k-level.  The output vorticity is at tracer points per level.

    Parameters
    ----------
    k_index : int or None
        If given, only this k-level is materialised (2D output).
        If *None*, the result stays lazy (3D dask-backed).
        ``calculate_jacobian`` does not materialise internally, so the
        lazy path works with dask arrays.
    """
    zdim = _get_vertical_dim(ds_merge.U)

    if k_index is not None:
        u_x = _materialise_for_xgcm(ds_merge.U.isel({zdim: k_index}))
        v_y = _materialise_for_xgcm(ds_merge.V.isel({zdim: k_index}))
    else:
        # Lazy — calculate_jacobian handles dask arrays natively.
        u_x = ds_merge.U
        v_y = ds_merge.V

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(u_x, v_y, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)

    ro = zeta / f
    ro.name = "Ro"
    ro.attrs["units"] = "1"
    return ro

def burger_number(ds_merge, grid, mld=None):
    """
    Burger number Bu = (Ro / Fr)².

    Both Ro and Fr are evaluated at the MLD (nearest model level to the
    mixed-layer depth in each water column).  This is inherently a 2D
    quantity (no depth dispatch).

    Implementation: the MLD varies per water column, so we cannot use a
    single ``k_index``.  Instead we compute Ro and Fr on the full 3D
    column and then extract at the MLD via ``_extract_at_mld``.
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)

    # Rossby number at MLD
    ro_3d = rossby_number_3d(ds_merge, grid, k_index=None)
    ro_at_mld = _extract_at_mld(ro_3d, mld, ds_merge)

    # Froude number at MLD
    fr_3d = froude_number_3d(ds_merge, grid, mld=mld, k_index=None)
    fr_at_mld = _extract_at_mld(fr_3d, mld, ds_merge)

    bu = xr.where(fr_at_mld != 0, (ro_at_mld / fr_at_mld)**2, np.nan)
    bu.name = "Burger_number"
    bu.attrs["units"] = "1"
    return bu

# ===========================================================================
#  GROUP 3: SURFACE WIND PROPERTIES (inherently 2D — no depth dispatch)
# ===========================================================================

def wind_stress_curl(ds_merge, grid):
    """
    Wind stress curl  curl(τ) = ∂τ_y_geo/∂λ − ∂τ_x_geo/∂φ   [N m⁻³].

    Uses ``calculate_jacobian`` because oceTAUX/oceTAUY are vector
    components in model coordinates on staggered grids (same layout as
    U/V).  The Jacobian handles: stagger → tracer interpolation,
    model → geographic rotation, and geographic derivatives.
    """
    taux = _materialise_for_xgcm(ds_merge.oceTAUX.copy(deep=True))
    tauy = _materialise_for_xgcm(ds_merge.oceTAUY.copy(deep=True))

    dtaux_dlambda, dtaux_dphi, dtauy_dlambda, dtauy_dphi = (
        ng.calculate_jacobian(taux, tauy, ds_merge, grid))

    curl_tau = dtauy_dlambda - dtaux_dphi
    curl_tau.name = "wind_stress_curl"
    curl_tau.attrs["units"] = "N m^-3"
    return curl_tau


def ekman_pumping(ds_merge, grid, rho0=RHO0):
    """
    Ekman pumping velocity  w_E = curl(τ) / (ρ₀ f)   [m s⁻¹].

    Singular near the equator (f → 0); masked to NaN there.
    """
    curl_tau = wind_stress_curl(ds_merge, grid)
    f = coriolis_parameter(ds_merge, grid)

    wE = xr.where(np.abs(f) > 0, curl_tau / (rho0 * f), np.nan)
    wE.name = "ekman_pumping"
    wE.attrs["units"] = "m s^-1"
    return wE


def _wind_stress_geographic(ds_merge, grid):
    """
    Interpolate wind stress to tracer points and rotate to geographic.

    oceTAUX/oceTAUY are model-basis vector components on staggered grids.
    Returns (τ_λ, τ_φ) on the tracer grid — geographic zonal and
    meridional wind stress.
    """
    taux_c = grid.interp(_materialise_for_xgcm(ds_merge.oceTAUX), 'X', boundary='fill')
    tauy_c = grid.interp(_materialise_for_xgcm(ds_merge.oceTAUY), 'Y', boundary='fill')

    # Rotate model basis → geographic basis using CS/SN
    tau_lambda = taux_c * ds_merge['CS'] - tauy_c * ds_merge['SN']
    tau_phi    = taux_c * ds_merge['SN'] + tauy_c * ds_merge['CS']
    return tau_lambda, tau_phi


def ekman_transport_velocity(ds_merge, grid, rho0=RHO0):
    """
    Ekman transport velocity components from k × τ / (ρ₀ f):

        u_E =  τ_φ / (ρ₀ f)
        v_E = -τ_λ / (ρ₀ f)

    Wind stress is first interpolated to tracer points and rotated to
    geographic coordinates before dividing by f.

    Returns
    -------
    dict with keys ``"u_ekman"``, ``"v_ekman"``.
    """
    f = coriolis_parameter(ds_merge, grid)
    denom = rho0 * f
    safe_denom = xr.where(np.abs(f) > 0, denom, np.nan)

    tau_lambda, tau_phi = _wind_stress_geographic(ds_merge, grid)

    u_E =  tau_phi   / safe_denom
    v_E = -tau_lambda / safe_denom

    u_E.name = "u_ekman"
    v_E.name = "v_ekman"
    u_E.attrs["units"] = "m s^-1"
    v_E.attrs["units"] = "m s^-1"

    return {"u_ekman": u_E, "v_ekman": v_E}


# ===========================================================================
#  GROUP 4: ADVECTIVE BUOYANCY FLUXES & ERTEL PV
# ===========================================================================

def buoyancy_field_3d(ds_merge):
    """3D buoyancy field b = g ρ / ρ_ref, scaled ×1e3 to match project convention.

    This is a **fully lazy** version that avoids the ``.persist()`` calls in
    ``physical_calculations.buoyancy_of_field``.  Keeping the graph lazy is
    critical so that downstream code can select individual k-levels (for
    point-depth strategies) or build one flux component at a time (for column
    strategies) without inflating the task graph.
    """
    import dbof.utils.jmd95_xgcm_implementation as jmd95

    g = 0.0098     # km/s^2  (same constant as physical_calculations)
    ref_rho = 1025.0

    p = xr.zeros_like(ds_merge.Theta)

    rho = xr.apply_ufunc(
        jmd95.jmd95,
        ds_merge.Salt,
        ds_merge.Theta,
        p,
        dask="parallelized",
        output_dtypes=[float],
    )

    b = (g * rho / ref_rho) * 1e3   # includes the project ×1e3 scaling
    b.name = "buoyancy"
    b.attrs["units"] = "m^2 s^-3"
    return b


def advective_buoyancy_fluxes_3d(ds_merge, grid, k_index=None):
    """
    Advective buoyancy flux components: uB, vB, wB.

    Horizontal velocity is interpolated to tracer points and rotated to
    geographic coordinates (zonal/meridional) before multiplying with
    buoyancy, so uB and vB represent geographic flux components.
    W is interpolated from cell interfaces to cell centres.

    Parameters
    ----------
    k_index : int or None
        If given, only this k-level is materialised (2D output).
        If *None*, the result stays lazy (3D dask-backed).

    Returns
    -------
    uB, vB, wB : xr.DataArray
        Zonal, meridional, and vertical advective buoyancy fluxes.
    """
    b = buoyancy_field_3d(ds_merge)   # lazy
    zdim = _get_vertical_dim(ds_merge.U)
    b_zdim = _get_vertical_dim(b)

    if k_index is not None:
        U_k = _materialise_for_xgcm(ds_merge.U.isel({zdim: k_index}))
        V_k = _materialise_for_xgcm(ds_merge.V.isel({zdim: k_index}))
        U_c = grid.interp(U_k, 'X', boundary='fill')
        V_c = grid.interp(V_k, 'Y', boundary='fill')
        b_k = b.isel({b_zdim: k_index}).compute()
    else:
        # Lazy path — no materialise.
        U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
        V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')
        b_k = b

    u_geo = U_c * ds_merge['CS'] - V_c * ds_merge['SN']
    v_geo = U_c * ds_merge['SN'] + V_c * ds_merge['CS']

    uB = u_geo * b_k
    vB = v_geo * b_k

    # Interpolate W from cell interfaces (k_l / Zl) to tracer cell
    # centres (k / Z), then multiply with buoyancy.
    W_c = _interp_w_to_tracer_levels(ds_merge)   # lazy
    if k_index is not None:
        w_zdim = _get_vertical_dim(W_c)
        W_c = W_c.isel({w_zdim: k_index}).compute()
    wB = W_c * b_k

    uB.name = "uB"
    vB.name = "vB"
    wB.name = "wB"
    for v in (uB, vB, wB):
        v.attrs["units"] = "m^2 s^-3"
    return uB, vB, wB


def ertel_pv_terms_3d(ds_merge, grid, k_index=None):
    """
    Ertel potential vorticity and its decomposition.

    q = (ζ + f) b_z  +  (w_y - v_z) b_x  +  (u_z - w_x) b_y
        ─────────────    ──────────────────────────────────────
          q_vert                    q_tilt

    The two tilting sub-terms are combined into a single ``q_tilt`` field
    as requested.

    Parameters
    ----------
    k_index : int or None
        If given, all intermediate fields are materialised only at this
        k-level, producing 2D output.  This is the memory-saving path
        used by ``compute_ertel_pv`` for point-depth strategies and the
        per-k column assembly loop.

        If *None*, all intermediates stay lazy (dask-backed) and the
        output is 3D.  ``calculate_jacobian`` and
        ``calculate_native_gradient_tracer`` both operate on dask
        arrays natively.

    Coordinate notes
    ----------------
    All terms are in geographic (zonal/meridional) coordinates on the
    tracer grid:

    - ζ is computed via the native-grid Jacobian (geographic output).
    - u_z, v_z are geographic via ``vertical_shear_components_3d``.
    - b_x, b_y and w_x, w_y are geographic via ``calculate_native_gradient_tracer``.
    - b_z is on tracer levels (scalar — no rotation needed).
    - W is interpolated from k_l/Zl to k/Z before horizontal gradients.

    Returns
    -------
    dict of str → xr.DataArray
        Keys: ``ertel_pv``, ``ertel_pv_vertical``, ``ertel_pv_tilt``
    """
    if "W" not in ds_merge:
        raise ValueError("Ertel PV tilting terms require ds_merge['W'].")

    zdim = _get_vertical_dim(ds_merge.U)

    # Helper: select + materialise at k if k_index is set; else stay lazy.
    def _at_k(da, da_zdim=None):
        if da_zdim is None:
            da_zdim = zdim
        if k_index is not None:
            return _materialise_for_xgcm(da.isel({da_zdim: k_index}))
        return da

    # Horizontal vorticity — Jacobian handles dask natively.
    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
        _at_k(ds_merge.U), _at_k(ds_merge.V), ds_merge, grid)
    zeta = dv_dx - du_dy

    f = coriolis_parameter(ds_merge, grid)

    # Buoyancy and its derivatives (all on tracer points).
    b = buoyancy_field_3d(ds_merge)                       # lazy
    b_zdim = _get_vertical_dim(b)
    b_x, b_y = ng.calculate_native_gradient_tracer(
        _at_k(b, b_zdim), ds_merge, grid=grid)
    b_z = _vertical_derivative(b, ds_merge)               # lazy
    if k_index is not None:
        b_z = b_z.isel({b_zdim: k_index}).compute()

    # Geographic vertical shear — k_index flows through.
    u_z, v_z = vertical_shear_components_3d(ds_merge, grid, k_index=k_index)

    # W horizontal gradients.  Interpolate W from cell interfaces
    # (k_l / Zl) to tracer cell centres (k / Z) first.
    W_on_tracer = _interp_w_to_tracer_levels(ds_merge)    # lazy
    w_zdim = _get_vertical_dim(W_on_tracer)
    w_x, w_y = ng.calculate_native_gradient_tracer(
        _at_k(W_on_tracer, w_zdim), ds_merge, grid=grid)

    # PV terms
    q_vert = (zeta + f) * b_z
    q_tilt_x = (w_y - v_z) * b_x
    q_tilt_y = (u_z - w_x) * b_y
    q_tilt = q_tilt_x + q_tilt_y
    q_total = q_vert + q_tilt

    q_total.name = "ertel_pv"
    q_vert.name = "ertel_pv_vertical"
    q_tilt.name = "ertel_pv_tilt"

    for v in (q_total, q_vert, q_tilt):
        v.attrs["units"] = "s^-3"

    return {
        "ertel_pv": q_total,
        "ertel_pv_vertical": q_vert,
        "ertel_pv_tilt": q_tilt,
    }


# ===========================================================================
#  DISPATCH TABLE: field-name → (compute_fn, is_3d)
# ===========================================================================
# For 3D fields, depth strategies are applied automatically.
# For 2D fields (is_3d=False), the result is used as-is.
#
# Functions that return dicts (ertel_pv_terms, ekman_transport_velocity,
# advective_buoyancy_fluxes) are handled specially in the subset callbacks.
# ===========================================================================


# ===========================================================================
#  SUBSET COMPUTE FUNCTIONS
#  (called by generate_global_depth.py — same callback pattern as the
#   surface pipeline)
# ===========================================================================

def compute_stratification(ds_merge, grid, computed_feature_channels):
    """
    Subset: stratification

    Computes MLD, N² at four depths, and mixed-layer heat content.

    Channel names produced:
        mixed_layer_depth,
        N2_sfc, N2_z25m, N2_mld, N2_mld_mean,
        ml_heat_content
    """
    results = {}
    requested = set(computed_feature_channels)

    # MLD — shared dependency
    mld = mixed_layer_depth(ds_merge)
    if "mixed_layer_depth" in requested:
        results["mixed_layer_depth"] = mld

    # N² at four depths
    n2_channels = {c for c in requested if c.startswith("N2_")}
    if n2_channels:
        n2_3d = buoyancy_frequency_squared_3d(ds_merge)
        results.update(apply_depth_strategies(
            n2_3d, "N2", ds_merge, mld=mld, requested=requested))

    # Mixed-layer heat content
    if "ml_heat_content" in requested:
        results["ml_heat_content"] = mixed_layer_heat_content(
            ds_merge, mld=mld)

    if results:
        keys = list(results.keys())
        materialised = dask.compute(*[results[k] for k in keys], retries=10)
        results = dict(zip(keys, materialised))

    return results


def compute_vertical_shear(ds_merge, grid, computed_feature_channels):
    """
    Subset: vertical_shear

    Computes vertical shear magnitude and Richardson number at four depth
    definitions.  Uses ``vertical_shear_components_3d(k_index=…)`` via the
    ``field_at_k`` callable interface of ``apply_depth_strategies`` so that
    point strategies materialise only a 2D slice and column strategies
    assemble the 3D field level-by-level.

    Rotation preserves the magnitude (``|S|² = uz² + vz²``), so
    ``vertical_shear`` and ``Ri`` do not depend on the coordinate basis.

    Channel names produced:
        vertical_shear_sfc, vertical_shear_z25m,
            vertical_shear_mld, vertical_shear_mld_mean,
        Ri_sfc, Ri_z25m, Ri_mld, Ri_mld_mean,
    """
    results = {}
    requested = set(computed_feature_channels)

    shear_requested = {c for c in requested if c.startswith("vertical_shear_")}
    ri_requested    = {c for c in requested if c.startswith("Ri_")}
    if not shear_requested and not ri_requested:
        return results

    mld = mixed_layer_depth(ds_merge).compute()

    # Per-k callable: materialise + interp + rotate at one depth level,
    # then compute shear² = uz² + vz² (rotation-invariant).
    def _shear2_at_k(k):
        uz, vz = vertical_shear_components_3d(ds_merge, grid, k_index=k)
        return uz**2 + vz**2

    # Vertical shear magnitude at four depths
    if shear_requested:
        def _shear_at_k(k):
            return np.sqrt(_shear2_at_k(k))
        results.update(apply_depth_strategies(
            None, "vertical_shear", ds_merge, mld=mld,
            field_at_k=_shear_at_k, requested=requested))

    # Richardson number at four depths
    if ri_requested:
        n2_3d = buoyancy_frequency_squared_3d(ds_merge)
        n2_zdim = _get_vertical_dim(n2_3d)

        def _ri_at_k(k):
            shear2 = _shear2_at_k(k)
            n2_k = n2_3d.isel({n2_zdim: k}).compute()
            return xr.where(shear2 > 0, n2_k / shear2, np.nan)

        results.update(apply_depth_strategies(
            None, "Ri", ds_merge, mld=mld,
            field_at_k=_ri_at_k, requested=requested))

    return results


def compute_mixing_parameters(ds_merge, grid, computed_feature_channels):
    """
    Subset: mixing_parameters

    Computes Froude number, Rossby number, and Burger number at the
    appropriate depth definitions.

    Uses ``froude_number_3d(k_index=…)`` and ``rossby_number_3d(k_index=…)``
    via the ``field_at_k`` callable interface of ``apply_depth_strategies``
    so that point strategies materialise only a 2D slice and column
    strategies assemble the 3D field level-by-level.

    Channel names produced:
        Fr_sfc, Fr_z25m, Fr_mld, Fr_mld_mean,
        Ro_sfc, Ro_z25m, Ro_mld, Ro_mld_mean,
        Burger_number
    """
    results = {}
    requested = set(computed_feature_channels)
    mld = mixed_layer_depth(ds_merge).compute()

    # Froude number at four depths via per-k callable
    if any(c.startswith("Fr_") for c in requested):
        def _fr_at_k(k):
            return froude_number_3d(ds_merge, grid, mld=mld, k_index=k)
        results.update(apply_depth_strategies(
            None, "Fr", ds_merge, mld=mld,
            field_at_k=_fr_at_k, requested=requested))

    # Rossby number at four depths via per-k callable
    if any(c.startswith("Ro_") for c in requested):
        def _ro_at_k(k):
            return rossby_number_3d(ds_merge, grid, k_index=k)
        results.update(apply_depth_strategies(
            None, "Ro", ds_merge, mld=mld,
            field_at_k=_ro_at_k, requested=requested))

    # Burger number (inherently 2D — uses k_index=0 internally)
    if "Burger_number" in requested:
        results["Burger_number"] = burger_number(ds_merge, grid, mld=mld)

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results


def compute_ertel_pv(ds_merge, grid, computed_feature_channels):
    """
    Subset: ertel_pv

    Computes Ertel PV terms at four depth definitions.

    Uses ``ertel_pv_terms_3d(k_index=…)`` for per-k materialisation.
    Since that function returns a dict of 3 fields, this compute function
    handles the depth-strategy dispatch directly rather than going
    through ``apply_depth_strategies`` (which expects a single-field
    callable).

    * **Point strategies** (sfc, z25m): call ``ertel_pv_terms_3d``
      at the needed k-index(es) — only 2D slices live in memory.
    * **Column strategies** (mld, mld_mean): loop over all k, accumulate
      per-base 2D slices in float32, concatenate into 3D, then apply
      column reductions.

    Channel names produced:
        ertel_pv_sfc, ertel_pv_z25m, ertel_pv_mld, ertel_pv_mld_mean,
        ertel_pv_vertical_sfc, ..., ertel_pv_vertical_mld_mean,
        ertel_pv_tilt_sfc, ..., ertel_pv_tilt_mld_mean,
    """
    results = {}
    requested = set(computed_feature_channels)

    pv_bases = ("ertel_pv", "ertel_pv_vertical", "ertel_pv_tilt")
    # Which bases have at least one requested channel?
    active_bases = [b for b in pv_bases
                    if any(c.startswith(b + "_") for c in requested)]
    if not active_bases:
        return results

    mld = mixed_layer_depth(ds_merge).compute()

    # ------------------------------------------------------------------
    # Point strategies: compute ertel_pv_terms at specific k indices
    # ------------------------------------------------------------------
    k_map = _get_point_k_indices(ds_merge)
    point_k_needed = set()
    for base in active_bases:
        for suffix, k_idx in k_map.items():
            if f"{base}_{suffix}" in requested:
                point_k_needed.add((suffix, k_idx))

    for suffix, k_idx in point_k_needed:
        pv_dict = ertel_pv_terms_3d(ds_merge, grid, k_index=k_idx)
        for base in active_bases:
            key = f"{base}_{suffix}"
            if key in requested:
                results[key] = pv_dict[base]

    # ------------------------------------------------------------------
    # Column strategies: loop over k, assemble 3D per base, then reduce
    # ------------------------------------------------------------------
    col_suffixes_needed = set()
    for base in active_bases:
        for suffix in COLUMN_STRATEGIES:
            if f"{base}_{suffix}" in requested:
                col_suffixes_needed.add(suffix)

    if col_suffixes_needed:
        zdim = _get_vertical_dim(ds_merge.Theta)
        nk = ds_merge.Theta.sizes[zdim]
        pv_levels = {base: [] for base in active_bases}

        logging.info(
            f"  ertel_pv: assembling 3D from per-k callable "
            f"({nk} levels) for column strategies"
        )
        for k_idx in range(nk):
            pv_dict = ertel_pv_terms_3d(ds_merge, grid, k_index=k_idx)
            for base in active_bases:
                pv_levels[base].append(
                    pv_dict[base].astype(np.float32))

        for base in active_bases:
            field3d = xr.concat(pv_levels[base], dim=zdim)
            for suffix in col_suffixes_needed:
                key = f"{base}_{suffix}"
                if key in requested:
                    results[key] = COLUMN_STRATEGIES[suffix](
                        field3d, ds_merge, mld=mld)
            del field3d
        del pv_levels

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results


def compute_buoyancy_fluxes(ds_merge, grid, computed_feature_channels):
    """
    Subset: buoyancy_fluxes  (renamed from vertical_fluxes)

    Computes advective buoyancy flux components at four depth definitions.

    Uses per-k materialisation of U and V (via ``grid.interp`` on 2D
    slices) to avoid holding the full 3D velocity fields in memory.
    Buoyancy and W stay lazy and are materialised per-k inside the
    ``field_at_k`` closures.

    Channel names produced:
        uB_sfc, uB_z25m, uB_mld, uB_mld_mean,
        vB_sfc, vB_z25m, vB_mld, vB_mld_mean,
        wB_sfc, wB_z25m, wB_mld, wB_mld_mean,
    """
    results = {}
    requested = set(computed_feature_channels)

    flux_bases = ("uB", "vB", "wB")
    flux_requested = any(
        ch.startswith(base) for ch in requested for base in flux_bases
    )
    if not flux_requested:
        return results

    # -- shared lazy fields ------------------------------------------------
    mld = mixed_layer_depth(ds_merge).compute()
    b = buoyancy_field_3d(ds_merge)               # lazy 3D dask
    zdim_u = _get_vertical_dim(ds_merge.U)
    zdim_t = _get_vertical_dim(ds_merge.Theta)
    W_c = _interp_w_to_tracer_levels(ds_merge)    # lazy dask
    w_zdim = _get_vertical_dim(W_c)

    # Per-k helper: materialise U/V at one depth, interp, rotate.
    def _uv_geo_at_k(k):
        U_k = _materialise_for_xgcm(ds_merge.U.isel({zdim_u: k}))
        V_k = _materialise_for_xgcm(ds_merge.V.isel({zdim_u: k}))
        U_c = grid.interp(U_k, 'X', boundary='fill')
        V_c = grid.interp(V_k, 'Y', boundary='fill')
        u_geo = U_c * ds_merge['CS'] - V_c * ds_merge['SN']
        v_geo = U_c * ds_merge['SN'] + V_c * ds_merge['CS']
        return u_geo, v_geo

    # -- per-component depth dispatch --------------------------------------
    uB_requested = any(ch.startswith("uB_") for ch in requested)
    vB_requested = any(ch.startswith("vB_") for ch in requested)
    wB_requested = any(ch.startswith("wB_") for ch in requested)

    if uB_requested:
        def _uB_at_k(k):
            u_geo, _ = _uv_geo_at_k(k)
            b_k = b.isel({zdim_t: k}).compute()
            return u_geo * b_k
        results.update(apply_depth_strategies(
            None, "uB", ds_merge, mld=mld,
            field_at_k=_uB_at_k, requested=requested))

    if vB_requested:
        def _vB_at_k(k):
            _, v_geo = _uv_geo_at_k(k)
            b_k = b.isel({zdim_t: k}).compute()
            return v_geo * b_k
        results.update(apply_depth_strategies(
            None, "vB", ds_merge, mld=mld,
            field_at_k=_vB_at_k, requested=requested))

    if wB_requested:
        def _wB_at_k(k):
            w_k = W_c.isel({w_zdim: k}).compute()
            b_k = b.isel({zdim_t: k}).compute()
            return w_k * b_k
        results.update(apply_depth_strategies(
            None, "wB", ds_merge, mld=mld,
            field_at_k=_wB_at_k, requested=requested))

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results


def compute_surface_wind(ds_merge, grid, computed_feature_channels):
    """
    Subset: surface_wind

    Computes wind-stress curl, Ekman pumping, and Ekman transport velocity.
    All fields are inherently 2D (no depth dispatch).

    Channel names produced:
        wind_stress_curl, ekman_pumping, u_ekman, v_ekman
    """
    results = {}

    if "wind_stress_curl" in computed_feature_channels:
        results["wind_stress_curl"] = wind_stress_curl(ds_merge, grid)

    if "ekman_pumping" in computed_feature_channels:
        results["ekman_pumping"] = ekman_pumping(ds_merge, grid)

    ekman_channels = {"u_ekman", "v_ekman"}
    if ekman_channels.intersection(computed_feature_channels):
        ek = ekman_transport_velocity(ds_merge, grid)
        for ch in ekman_channels:
            if ch in computed_feature_channels:
                results[ch] = ek[ch]

    # Materialise
    if results:
        keys = list(results.keys())
        materialised = dask.compute(*[results[k] for k in keys], retries=10)
        results = dict(zip(keys, materialised))

    return results


# ===========================================================================
#  GROUP 5: ENERGETICS
# ===========================================================================

def kinetic_energy_submesoscale_3d(ds_merge, grid, mld=None):
    """
    Submesoscale kinetic energy at each depth level:

        KE = 0.5 * (H * b_x / f)²

    where H = MLD, b_x is the horizontal buoyancy gradient magnitude,
    and f is the Coriolis parameter.

    Returns a 3D field on tracer levels (k, face, j, i).
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)

    b = buoyancy_field_3d(ds_merge)   # lazy 3D
    f = coriolis_parameter(ds_merge, grid)

    return b, f, mld


def compute_energetics(ds_merge, grid, computed_feature_channels):
    """
    Subset: energetics

    Computes submesoscale kinetic energy KE = 0.5 * (H * b_x / f)²
    at four depth definitions, where H = MLD, b_x is the horizontal
    buoyancy gradient magnitude, and f is the Coriolis parameter.

    Channel names produced:
        KE_sfc, KE_z25m, KE_mld, KE_mld_mean
    """
    results = {}
    requested = set(computed_feature_channels)

    ke_channels = {c for c in requested if c.startswith("KE_")}
    if not ke_channels:
        return results

    mld = mixed_layer_depth(ds_merge).compute()
    b = buoyancy_field_3d(ds_merge)       # lazy 3D
    f = coriolis_parameter(ds_merge, grid)
    zdim_t = _get_vertical_dim(ds_merge.Theta)

    def _ke_at_k(k):
        b_k = _materialise_for_xgcm(b.isel({zdim_t: k}))
        b_x, b_y = ng.calculate_native_gradient_tracer(b_k, ds_merge, grid=grid)
        grad_b_mag = np.sqrt(b_x**2 + b_y**2)
        ke = 0.5 * (mld * grad_b_mag / f)**2
        return ke

    results.update(apply_depth_strategies(
        None, "KE", ds_merge, mld=mld,
        field_at_k=_ke_at_k, requested=requested))

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results


# ===========================================================================
#  GROUP 6: FRONTAL STRUCTURE (ported from generate_global.py surface pipeline)
# ===========================================================================

def compute_frontal_structure(ds_merge, grid, computed_feature_channels):
    """
    Subset: frontal_structure

    Computes scalar gradient-magnitude fields and the Turner angle at
    four depth definitions.

    These properties were originally surface-only in generate_global.py.
    They are now evaluated at each depth option via the standard
    depth-strategy framework.

    Channel names produced (each with _sfc, _z25m, _mld, _mld_mean):
        gradb2_*, gradtheta2_*, gradsalt2_*, gradeta2_*, gradrho2_*
    Plus inherently 2D:
        turner_angle_sfc, turner_angle_z25m, turner_angle_mld, turner_angle_mld_mean
    """
    results = {}
    requested = set(computed_feature_channels)
    mld = None

    def _ensure_mld():
        nonlocal mld
        if mld is None:
            mld = mixed_layer_depth(ds_merge).compute()
        return mld

    zdim_t = _get_vertical_dim(ds_merge.Theta)

    # --- Squared gradient fields at depth ---
    # Each uses calculate_native_gradient_tracer on a scalar at a given k.
    _SCALAR_MAP = {
        "gradtheta2": lambda ds, k: ds.Theta.isel({zdim_t: k}),
        "gradsalt2":  lambda ds, k: ds.Salt.isel({zdim_t: k}),
    }

    for base_name, scalar_fn in _SCALAR_MAP.items():
        channels = {c for c in requested if c.startswith(base_name + "_")}
        if not channels:
            continue
        _ensure_mld()

        def _grad2_at_k(k, _fn=scalar_fn):
            field_k = _materialise_for_xgcm(_fn(ds_merge, k))
            gx, gy = ng.calculate_native_gradient_tracer(
                field_k, ds_merge, grid=grid)
            return gx**2 + gy**2

        results.update(apply_depth_strategies(
            None, base_name, ds_merge, mld=mld,
            field_at_k=_grad2_at_k, requested=requested))

    # gradb2: squared buoyancy gradient magnitude
    if any(c.startswith("gradb2_") for c in requested):
        _ensure_mld()
        b = buoyancy_field_3d(ds_merge)

        def _gradb2_at_k(k):
            b_k = _materialise_for_xgcm(b.isel({zdim_t: k}))
            gx, gy = ng.calculate_native_gradient_tracer(
                b_k, ds_merge, grid=grid)
            return gx**2 + gy**2

        results.update(apply_depth_strategies(
            None, "gradb2", ds_merge, mld=mld,
            field_at_k=_gradb2_at_k, requested=requested))

    # gradrho2: squared density gradient magnitude
    if any(c.startswith("gradrho2_") for c in requested):
        _ensure_mld()
        import dbof.utils.physical_calculations as pc

        def _gradrho2_at_k(k):
            # Compute density at this k-level via the lazy density helper,
            # then materialise just this level.
            rho_3d = _density_lazy(ds_merge)
            rho_zdim = _get_vertical_dim(rho_3d)
            rho_k = rho_3d.isel({rho_zdim: k}).compute()
            gx, gy = ng.calculate_native_gradient_tracer(
                rho_k, ds_merge, grid=grid)
            return gx**2 + gy**2

        results.update(apply_depth_strategies(
            None, "gradrho2", ds_merge, mld=mld,
            field_at_k=_gradrho2_at_k, requested=requested))

    # gradeta2: SSH gradient — inherently 2D (no depth dimension).
    # Evaluated only at the surface; other depth suffixes are not meaningful.
    if "gradeta2_sfc" in requested:
        eta = _materialise_for_xgcm(ds_merge.Eta)
        gx, gy = ng.calculate_native_gradient_tracer(eta, ds_merge, grid=grid)
        results["gradeta2_sfc"] = gx**2 + gy**2

    # Turner angle at depth — requires gradtheta2, gradsalt2, gradrho2 at
    # each depth level.  Computed per-k using the linear EOS formula.
    if any(c.startswith("turner_angle_") for c in requested):
        _ensure_mld()
        ALPHA = 2.0e-4
        BETA  = 7.4e-4
        RHO0_TU = 1025.0

        def _turner_at_k(k):
            theta_k = _materialise_for_xgcm(ds_merge.Theta.isel({zdim_t: k}))
            salt_k  = _materialise_for_xgcm(ds_merge.Salt.isel({zdim_t: k}))
            gx_t, gy_t = ng.calculate_native_gradient_tracer(theta_k, ds_merge, grid=grid)
            gx_s, gy_s = ng.calculate_native_gradient_tracer(salt_k, ds_merge, grid=grid)
            gt2 = gx_t**2 + gy_t**2
            gs2 = gx_s**2 + gy_s**2

            rho_3d = _density_lazy(ds_merge)
            rho_zdim = _get_vertical_dim(rho_3d)
            rho_k = rho_3d.isel({rho_zdim: k}).compute()
            gx_r, gy_r = ng.calculate_native_gradient_tracer(rho_k, ds_merge, grid=grid)
            gr2 = gx_r**2 + gy_r**2

            numer = RHO0_TU * (BETA**2 * gs2 - ALPHA**2 * gt2)
            denom = xr.where(gr2 > 0, -gr2 / RHO0_TU, np.nan)
            tu_rad = np.arctan(numer / denom)
            return np.degrees(tu_rad)

        results.update(apply_depth_strategies(
            None, "turner_angle", ds_merge, mld=mld,
            field_at_k=_turner_at_k, requested=requested))

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results


# ===========================================================================
#  GROUP 7: KINEMATIC (ported from generate_global.py surface pipeline)
# ===========================================================================

def compute_kinematic(ds_merge, grid, computed_feature_channels):
    """
    Subset: kinematic

    Computes velocity-derived properties (relative vorticity, strain,
    divergence, Rossby number, Okubo-Weiss) at four depth definitions.

    These properties were originally surface-only in generate_global.py
    (via all_velocity_properties).  They are now evaluated at each depth
    using per-k Jacobian computation.

    Channel names produced (each with _sfc, _z25m, _mld, _mld_mean):
        relative_vorticity_*, strain_n_*, strain_s_*, strain_mag_*,
        divergence_*, okubo_weiss_*
    """
    results = {}
    requested = set(computed_feature_channels)

    bases = ("relative_vorticity", "strain_n", "strain_s", "strain_mag",
             "divergence", "okubo_weiss")
    active = any(
        any(c.startswith(b + "_") for c in requested)
        for b in bases
    )
    if not active:
        return results

    mld = mixed_layer_depth(ds_merge).compute()
    zdim = _get_vertical_dim(ds_merge.U)

    def _velocity_props_at_k(k):
        """Compute all velocity properties at a single k-level."""
        u_k = _materialise_for_xgcm(ds_merge.U.isel({zdim: k}))
        v_k = _materialise_for_xgcm(ds_merge.V.isel({zdim: k}))

        du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
            u_k, v_k, ds_merge, grid)

        omega     = dv_dx - du_dy
        strain_n  = du_dx - dv_dy
        strain_s  = du_dy + dv_dx
        strain_m  = np.sqrt(strain_n**2 + strain_s**2)
        div       = du_dx + dv_dy
        ow        = strain_n**2 + strain_s**2 - omega**2

        return {
            "relative_vorticity": omega,
            "strain_n": strain_n,
            "strain_s": strain_s,
            "strain_mag": strain_m,
            "divergence": div,
            "okubo_weiss": ow,
        }

    # Per-base depth dispatch using field_at_k callables
    for base in bases:
        channels = {c for c in requested if c.startswith(base + "_")}
        if not channels:
            continue

        def _field_at_k(k, _base=base):
            props = _velocity_props_at_k(k)
            return props[_base]

        results.update(apply_depth_strategies(
            None, base, ds_merge, mld=mld,
            field_at_k=_field_at_k, requested=requested))

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results


# ===========================================================================
#  GROUP 8: FRONTOGENESIS (ported from generate_global.py surface pipeline)
# ===========================================================================

def compute_frontogenesis(ds_merge, grid, computed_feature_channels):
    """
    Subset: frontogenesis

    Computes frontogenesis tendency and geostrophic/ageostrophic
    decomposition at four depth definitions.

    Channel names produced (each with _sfc, _z25m, _mld, _mld_mean):
        frontogenesis_tendency_*, frontogenesis_geo_*, frontogenesis_ageo_*
    Plus inherently 2D geostrophic velocity (surface only):
        ug_sfc, vg_sfc
    """
    results = {}
    requested = set(computed_feature_channels)

    fronto_bases = ("frontogenesis_tendency", "frontogenesis_geo",
                    "frontogenesis_ageo")
    active = any(
        any(c.startswith(b + "_") for c in requested)
        for b in fronto_bases
    )
    geo_vel = any(c.startswith("ug_") or c.startswith("vg_") for c in requested)

    if not active and not geo_vel:
        return results

    mld = mixed_layer_depth(ds_merge).compute()
    zdim = _get_vertical_dim(ds_merge.U)
    zdim_t = _get_vertical_dim(ds_merge.Theta)

    f = coriolis_parameter(ds_merge, grid)

    def _frontogenesis_at_k(k):
        """Full and geostrophic frontogenesis at a single k-level."""
        u_k = _materialise_for_xgcm(ds_merge.U.isel({zdim: k}))
        v_k = _materialise_for_xgcm(ds_merge.V.isel({zdim: k}))

        du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
            u_k, v_k, ds_merge, grid)

        # Buoyancy gradient at this level
        b = buoyancy_field_3d(ds_merge)
        b_k = _materialise_for_xgcm(b.isel({zdim_t: k}))
        grad_bx, grad_by = ng.calculate_native_gradient_tracer(
            b_k, ds_merge, grid=grid)

        # Full frontogenesis
        F_full = -(du_dx * grad_bx**2 +
                   (du_dy + dv_dx) * grad_bx * grad_by +
                   dv_dy * grad_by**2)

        # Geostrophic velocity from Eta (surface only, constant w.r.t. depth
        # in a barotropic approximation; still used here for consistency).
        g_accel = 9.81
        eta_grad_x, eta_grad_y = ng.calculate_native_gradient_tracer(
            _materialise_for_xgcm(ds_merge['Eta']), ds_merge, grid=grid)
        ug = -(g_accel / f) * eta_grad_y
        vg =  (g_accel / f) * eta_grad_x

        # Geostrophic frontogenesis
        gx_ug, gy_ug = ng.calculate_native_gradient_tracer(ug, ds_merge, grid=grid)
        gx_vg, gy_vg = ng.calculate_native_gradient_tracer(vg, ds_merge, grid=grid)
        F_geo = -(gx_ug * grad_bx**2 +
                  (gy_ug + gx_vg) * grad_bx * grad_by +
                  gy_vg * grad_by**2)

        F_ageo = F_full - F_geo

        return {
            "frontogenesis_tendency": F_full,
            "frontogenesis_geo": F_geo,
            "frontogenesis_ageo": F_ageo,
            "ug": ug,
            "vg": vg,
        }

    for base in fronto_bases:
        channels = {c for c in requested if c.startswith(base + "_")}
        if not channels:
            continue

        def _field_at_k(k, _base=base):
            props = _frontogenesis_at_k(k)
            return props[_base]

        results.update(apply_depth_strategies(
            None, base, ds_merge, mld=mld,
            field_at_k=_field_at_k, requested=requested))

    # Geostrophic velocity — inherently surface, but allow _sfc suffix
    for vname in ("ug", "vg"):
        key = f"{vname}_sfc"
        if key in requested:
            props_sfc = _frontogenesis_at_k(0)
            results[key] = props_sfc[vname]

    # Materialise any remaining dask-backed results
    if results:
        lazy_keys = [k for k, v in results.items()
                     if hasattr(v, 'chunks') and v.chunks is not None]
        if lazy_keys:
            mat = dask.compute(
                *[results[k] for k in lazy_keys], retries=10)
            results.update(dict(zip(lazy_keys, mat)))

    return results
