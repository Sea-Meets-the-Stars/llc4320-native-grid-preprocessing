"""
Depth-resolved derived diagnostics for LLC4320 data (vertical structure).

This module holds ONLY fields that genuinely require the water column —
a vertical dimension (``k``), vertical velocity (``W``), or vertical
spacing (``drF``).  Everything dimension-agnostic (kinematics, horizontal
gradients, frontogenesis, wind, geographic rotations, density/buoyancy)
lives in ``calculate_fields.py`` — one implementation per field, shared
by the SURF/OSN and DEPTH pipelines.  This module imports from
``calculate_fields``; never the reverse.  See prompts/field_migration.md.

All field functions are **fully lazy** — they build dask task graphs
without calling ``.compute()``.  A single ``dask.compute()`` at the end
of each ``compute_*`` entry point (in ``depth_subsets.py``) materialises
only the final 2D outputs.  Chunking is enforced once at the pipeline
entry points (``depth_subsets._ensure_depth_chunking``).

MITgcm / xmitgcm coordinate conventions
-----------------------------------------
Tracer fields (Theta, Salt, buoyancy, density) live on:
    ``('k', 'face', 'j', 'i')``  with vertical coordinate ``Z``

Zonal velocity (U) lives on:
    ``('k', 'face', 'j', 'i_g')``;  meridional (V) on
    ``('k', 'face', 'j_g', 'i')``;  vertical (W) on
    ``('k_l', 'face', 'j', 'i')``  with vertical coordinate ``Zl``.
"""

import logging

import numpy as np
import xarray as xr

import dbof.utils.native_gradient as ng
from dbof.preprocessing.calculate_fields import (
    coriolis_parameter,
    potential_density,
    potential_density_anomaly,
    buoyancy_of_field,
    geographic_velocity,
    rossby_number,
    compute_buoyancy_gradients,
)

from dbof.preprocessing.vertical_helpers import (
    _get_vertical_dim,
    _get_depth_coord,
    _get_vertical_spacing,
    _nearest_k_to_depth,
    _vertical_derivative,
    _interp_w_to_tracer_levels,
    _extract_at_mld,
    _masked_ml_mean,
)

from dbof.preprocessing.physical_constants import (
    RHO0_REFERENCE,
    G,
    CP,
    OMEGA_EARTH,
    ALPHA,
    BETA,
    MLD_REFERENCE_DEPTH_M,
    MLD_INTEGRATION_DEPTH_M,
)


# ===========================================================================
#  STRATIFICATION & HEAT
# ===========================================================================


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

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and vertical coordinate
        ``Z``.
    density_threshold : float, optional
        Potential-density anomaly threshold [kg m⁻³] defining the base of the
        mixed layer (default 0.03).
    ref_depth_m : float, optional
        Reference depth [m] used to locate the near-surface reference level
        (default ``MLD_REFERENCE_DEPTH_M``, typically 10 m).

    Returns
    -------
    xr.DataArray
        2D mixed-layer depth [m], positive metres below the surface,
        with ``name="MLD"``.
    """
    # Reuse the single canonical σ₀ routine (ρ(p=0) − 1000) rather than
    # re-deriving the anomaly here -- keeps one function per property.
    sigma0 = potential_density_anomaly(ds_merge)
    zdim = _get_vertical_dim(sigma0)

    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)
    ref_k = _nearest_k_to_depth(z_vals, ref_depth_m)

    sigma0_ref = sigma0.isel({zdim: ref_k})
    delta_sigma = sigma0 - sigma0_ref

    z_masked = z.where(delta_sigma <= density_threshold)
    mld = z_masked.max(dim=zdim, skipna=True)

    mld.name = "MLD"
    mld.attrs["long_name"] = "Mixed layer depth"
    mld.attrs["units"] = "m"
    return mld


def buoyancy_frequency_squared(ds_merge):
    """
    Full 3D profile of N² = (g/ρ₀) dρ/dz.

    With the positive-downward z convention used throughout this module,
    density increases with depth in a stably stratified ocean, so
    dρ/dz > 0 and N² > 0 without a leading negative sign.

    Computed on tracer levels (dim='k', coord='Z').

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and vertical spacing
        variables.

    Returns
    -------
    xr.DataArray
        3D buoyancy frequency squared N² [s⁻²] on tracer levels
        (dims ``k``, ``face``, ``j``, ``i``), with ``name="N2"``.
    """
    rho = potential_density(ds_merge)
    drho_dz = _vertical_derivative(rho, ds_merge)

    n2 = (G / RHO0_REFERENCE) * drho_dz
    n2.name = "N2"
    n2.attrs["long_name"] = "Buoyancy frequency squared"
    n2.attrs["units"] = "s^-2"
    return n2


def mixed_layer_depth_DI(ds_merge,
                         integration_depth_m=MLD_INTEGRATION_DEPTH_M,
                         n2=None):
    """
    Mixed-layer depth via the Depth Integration (DI) Method.

    The DI estimator defines the MLD as the N²-weighted mean depth over the
    upper ``integration_depth_m`` metres of the water column:

        MLD_DI = ∫₀ᴴ z N²(z) dz / ∫₀ᴴ N²(z) dz      (H = integration_depth_m)

    where N²(z) is the buoyancy frequency squared.  Discretised on the model
    levels (positive-downward depth ``z`` and layer thickness ``dz``):

        MLD_DI = Σ_{z≤H} z N² dz / Σ_{z≤H} N² dz

    Stratification dominates the weighting, so the estimator returns a depth
    near the base of the pycnocline.  Unlike the threshold estimator
    (:func:`mixed_layer_depth`) it does not land exactly on a discrete model
    level; it is a continuous, profile-weighted depth.

    Statically unstable layers (N² < 0) are floored to N² = 0 before
    weighting — consistent with the rest of this module
    (cf. :func:`balanced_richardson_number`) — so convective columns
    contribute zero weight rather than a negative one.

    Uses pure xarray reductions so the computation stays fully lazy when the
    inputs are dask-backed.  Returns a 2D field (positive metres, no depth
    dispatch needed).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, the vertical
        coordinate ``Z``, and the layer-thickness variable ``drF`` (used to
        build N² when *n2* is not supplied, and always for the depth
        coordinate / spacing).
    integration_depth_m : float, optional
        Upper-ocean integration depth H [m], positive downward (default
        ``MLD_INTEGRATION_DEPTH_M``, typically 300 m).  Only model levels
        with ``z ≤ integration_depth_m`` enter the integrals.
    n2 : xr.DataArray, optional
        Pre-computed 3D buoyancy frequency squared N² [s⁻²] on tracer
        levels.  If ``None``, computed via
        :func:`buoyancy_frequency_squared`.  Supplying it lets callers
        that already have N² (or that only have density, not Theta/Salt)
        reuse it and avoid recomputation.

    Returns
    -------
    xr.DataArray
        2D mixed-layer depth [m], positive metres below the surface, with
        ``name="MLD_DI"``; ``NaN`` in columns where the N² weight integral
        is zero (e.g. fully unstable or land columns).

    Generated by JXP and Claude
    """
    # N² weighting field (reuse caller's if supplied to avoid recomputation).
    if n2 is None:
        n2 = buoyancy_frequency_squared(ds_merge)
    # Floor statically unstable layers (N² < 0) to 0 so they carry zero,
    # not negative, weight in the depth integral.
    n2 = n2.clip(min=0.0)

    zdim = _get_vertical_dim(n2)
    # Positive-downward depth coordinate and layer thicknesses (both 1D).
    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz = _get_vertical_spacing(ds_merge, zdim=zdim)

    # Restrict both integrals to the upper integration_depth_m metres.
    mask = z <= integration_depth_m
    numer = (z * n2 * dz).where(mask).sum(dim=zdim)   # ∫ z N² dz
    denom = (n2 * dz).where(mask).sum(dim=zdim)       # ∫ N² dz

    # Guard the (denom == 0) columns — the weighted mean is undefined there.
    mld = xr.where(denom > 0, numer / denom, np.nan)

    mld.name = "MLD_DI"
    mld.attrs["long_name"] = "Mixed layer depth (Depth Integration Method)"
    mld.attrs["units"] = "m"
    mld.attrs["description"] = (
        "The depth to the bottom of the mixed layer calculated using the "
        "Depth Integration Method"
    )
    return mld


def mixed_layer_heat_content(ds_merge, mld=None):
    """
    Mixed-layer heat content integrated over the mixed layer.

    Q_ml = ∫₀ᴹᴸᴰ cp ρ₀ Θ dz   [J m⁻²]

    This is a 2D field (no depth dispatch — it is inherently an MLD integral).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and vertical spacing
        variables.
    mld : xr.DataArray, optional
        Pre-computed 2D mixed-layer depth [m].  If ``None``, computed via
        :func:`mixed_layer_depth`.

    Returns
    -------
    xr.DataArray
        2D mixed-layer heat content [J m⁻²], with
        ``name="ml_heat_content"``.
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)

    theta = ds_merge.Theta
    integrand = CP * RHO0_REFERENCE * theta

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
#  SHEAR & DIMENSIONLESS NUMBERS
# ===========================================================================


def vertical_shear_components(ds_merge, grid):
    """Lazy 3D vertical shear (du/dz, dv/dz) in geographic coordinates.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, rotation coefficients
        ``CS`` and ``SN``, and vertical spacing variables.
    grid : xgcm.Grid
        Grid object used for interpolation between velocity and tracer points.

    Returns
    -------
    uz : xr.DataArray
        Zonal shear du/dz [s⁻¹] on tracer levels.
    vz : xr.DataArray
        Meridional shear dv/dz [s⁻¹] on tracer levels.
    """
    dUdz = _vertical_derivative(ds_merge.U, ds_merge)
    dVdz = _vertical_derivative(ds_merge.V, ds_merge)

    # Interpolate the staggered shear components to tracer points and rotate
    # into the geographic (east/north) basis.
    uz, vz = ng.rotate_vector_to_geographic(dUdz, dVdz, ds_merge, grid)
    return uz, vz


def vertical_shear_magnitude(ds_merge, grid):
    """Lazy 3D |S| = sqrt(uz² + vz²).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, rotation coefficients, and
        vertical spacing variables.
    grid : xgcm.Grid
        Grid object used for interpolation between velocity and tracer points.

    Returns
    -------
    xr.DataArray
        3D vertical shear magnitude |S| [s⁻¹] on tracer levels.
    """
    uz, vz = vertical_shear_components(ds_merge, grid)
    shear_mag = np.sqrt(uz**2 + vz**2)
    return shear_mag


def richardson_number(ds_merge, grid):
    """Lazy 3D Ri = N² / (uz² + vz²).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, ``U``, ``V``,
        rotation coefficients, and vertical spacing variables.
    grid : xgcm.Grid
        Grid object used for interpolation and differencing.

    Returns
    -------
    xr.DataArray
        3D Richardson number Ri [dimensionless] on tracer levels; ``NaN``
        where shear² ≤ 0.  Statically unstable columns (N² < 0) are floored
        to N² = 0, so Ri = 0 there rather than negative.
    """
    n2 = buoyancy_frequency_squared(ds_merge)
    # Floor statically unstable stratification (N² < 0) to 0 so unstable
    # columns yield Ri = 0 rather than a negative Richardson number.
    n2 = n2.clip(min=0.0)
    uz, vz = vertical_shear_components(ds_merge, grid)
    shear2 = uz**2 + vz**2
    ri = xr.where(shear2 > 0, n2 / shear2, np.nan)
    return ri


def froude_number(ds_merge, grid, mld=None):
    """Lazy 3D Fr = speed / (N * MLD).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, ``U``, ``V``,
        rotation coefficients, and vertical spacing variables.
    grid : xgcm.Grid
        Grid object used for interpolation and differencing.
    mld : xr.DataArray, optional
        Pre-computed 2D mixed-layer depth [m].  If ``None``, computed via
        :func:`mixed_layer_depth`.

    Returns
    -------
    xr.DataArray
        3D Froude number Fr [dimensionless] on tracer levels; ``NaN``
        where N * MLD ≤ 0.
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)
    n2 = buoyancy_frequency_squared(ds_merge)
    n_abs = np.sqrt(np.abs(n2))

    U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
    V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')
    speed = np.sqrt(U_c**2 + V_c**2)

    denom = n_abs * mld
    fr = xr.where(denom > 0, speed / denom, np.nan)
    return fr


def burger_number(ds_merge, grid, mld=None):
    """Lazy 3D Bu = (Ro / Fr)².

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, ``U``, ``V``,
        rotation coefficients, and vertical spacing variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    mld : xr.DataArray, optional
        Pre-computed 2D mixed-layer depth [m].  If ``None``, computed via
        :func:`mixed_layer_depth`.

    Returns
    -------
    xr.DataArray
        3D Burger number Bu [dimensionless] on tracer levels; ``NaN``
        where Fr = 0.
    """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)
    ro_3d = rossby_number(ds_merge, grid)
    fr_3d = froude_number(ds_merge, grid, mld=mld)
    bu_3d = xr.where(fr_3d != 0, (ro_3d / fr_3d) ** 2, np.nan)
    return bu_3d


def balanced_richardson_number(ds_merge, grid, *, n2=None, gradb2=None):
    """Lazy 3D balanced Richardson number R_ib = N² f² / |∇_h b|².

    The balanced Richardson number (Thomas, Tandon & Mahadevan 2013) is
    the gradient Richardson number of a flow in thermal-wind balance and a
    dimensionless measure of frontal stability.  N² and the horizontal
    buoyancy gradient are both derived from the *same* physically
    consistent, unscaled buoyancy (b = (g / ρ₀) ρ), so the ratio is truly
    dimensionless — the gradient comes from the single
    :func:`calculate_fields.compute_buoyancy_gradients`.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, latitude ``YC``,
        vertical spacing variables, and horizontal grid metrics.
    grid : xgcm.Grid
        Grid object used for horizontal differencing.
    n2 : xr.DataArray, optional
        Pre-computed 3D N² [s⁻²].  If ``None``, computed via
        :func:`buoyancy_frequency_squared`.
    gradb2 : xr.DataArray, optional
        Pre-computed 3D |∇_h b|² [s⁻⁴] on the project buoyancy
        definition (b = G·sigma0/RHO0_REFERENCE, the same used for N²).
        If ``None``, computed internally via
        :func:`calculate_fields.compute_buoyancy_gradients`.

    Returns
    -------
    xr.DataArray
        3D balanced Richardson number R_ib [dimensionless] on tracer
        levels, with ``name="R_ib"``; ``NaN`` where |∇_h b|² = 0 (the
        ratio is undefined).  At the equator f → 0 so R_ib → 0.  Statically
        unstable columns (N² < 0) are floored to N² = 0, so R_ib = 0 there
        rather than negative.

    Generated by JXP and Claude
    """
    # N² numerator term (reuse caller's if supplied to avoid recomputation).
    if n2 is None:
        n2 = buoyancy_frequency_squared(ds_merge)
    # Floor statically unstable stratification (N² < 0) to 0 so unstable
    # columns yield R_ib = 0 rather than a negative value.
    n2 = n2.clip(min=0.0)
    # |∇_h b|² denominator term, on the same buoyancy definition as N².
    if gradb2 is None:
        bg = compute_buoyancy_gradients(ds_merge, grid)
        gradb2 = ng.grad_squared(bg.zonal, bg.merid)

    # Coriolis parameter f(YC); f² enters the numerator.
    f = coriolis_parameter(ds_merge, grid)

    # R_ib = N² f² / |∇_h b|²; mask the undefined |∇_h b|² = 0 columns.
    r_ib = xr.where(gradb2 > 0, n2 * f**2 / gradb2, np.nan)
    r_ib.name = "R_ib"
    r_ib.attrs["long_name"] = "Balanced Richardson number"
    r_ib.attrs["units"] = "1"
    return r_ib


# ===========================================================================
#  ADVECTIVE BUOYANCY FLUXES
# ===========================================================================


def advective_buoyancy_fluxes(ds_merge, grid):
    """Lazy (uB, vB, wB) in geographic coordinates.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, ``U``, ``V``, ``W``,
        rotation coefficients ``CS`` and ``SN``, and standard grid variables.
    grid : xgcm.Grid
        Grid object used for interpolation to tracer points.

    Returns
    -------
    uB : xr.DataArray
        Zonal advective buoyancy flux u·b [m^2 s^-3] on tracer levels.
    vB : xr.DataArray
        Meridional advective buoyancy flux v·b [m^2 s^-3] on tracer levels.
    wB : xr.DataArray
        Vertical advective buoyancy flux w·b [m^2 s^-3] on tracer levels.
    """
    b = buoyancy_of_field(ds_merge)

    u_geog, v_geog = geographic_velocity(ds_merge, grid)

    W_c = _interp_w_to_tracer_levels(ds_merge)

    return u_geog * b, v_geog * b, W_c * b


# ===========================================================================
#  ERTEL PV
# ===========================================================================


def ertel_pv_terms(ds_merge, grid):
    """Lazy Ertel PV and its vertical/tilting decomposition.

    q = (ζ + f) b_z  +  (w_y - v_z) b_x  +  (u_z - w_x) b_y
        ─────────────    ──────────────────────────────────────
          q_vert                    q_tilt

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, ``U``, ``V``, ``W``,
        rotation coefficients, latitude ``YC``, and all grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    dict
        Dictionary with keys:

        ``"ertel_pv"`` : xr.DataArray
            Total Ertel PV q [s⁻³] on tracer levels.
        ``"ertel_pv_vertical"`` : xr.DataArray
            Vertical component q_vert = (ζ + f) b_z [s⁻³].
        ``"ertel_pv_tilt"`` : xr.DataArray
            Tilting component q_tilt [s⁻³].
    """
    U = ds_merge.U
    V = ds_merge.V

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(U, V, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)

    b = buoyancy_of_field(ds_merge)
    b_x, b_y = ng.calculate_native_gradient_tracer(
        b, ds_merge, grid=grid)
    b_z = _vertical_derivative(b, ds_merge)

    u_z, v_z = vertical_shear_components(ds_merge, grid)

    W_on_tracer = _interp_w_to_tracer_levels(ds_merge)
    w_x, w_y = ng.calculate_native_gradient_tracer(
        W_on_tracer, ds_merge, grid=grid)

    q_vert = (zeta + f) * b_z
    q_tilt = (w_y - v_z) * b_x + (u_z - w_x) * b_y
    q_total = q_vert + q_tilt

    return {
        "ertel_pv": q_total,
        "ertel_pv_vertical": q_vert,
        "ertel_pv_tilt": q_tilt,
    }
