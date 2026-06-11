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
   slice or reduction at a particular depth definition (see
   ``depth_strategies.py``).

All field functions are **fully lazy** — they build dask task graphs without
calling ``.compute()``.  A single ``dask.compute()`` at the end of each
``compute_*`` entry point (in ``depth_subsets.py``) materialises only the
final 2D outputs.

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
"""

import logging

import numpy as np
import xarray as xr

import dbof.utils.native_gradient as ng
from dbof.preprocessing.calculate_additional_fields import (
    coriolis_parameter,
    VelocityJacobian,
    BuoyancyGradients,
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
    RHO0_BOUSSINESQ,
    G,
    G_KM,
    CP,
    OMEGA_EARTH,
    ALPHA,
    BETA,
    RHO0_SEAWATER,
    MLD_REFERENCE_DEPTH_M,
)


# ===========================================================================
#  DENSITY
# ===========================================================================

def _density_lazy(ds_merge):
    """Compute density lazily via apply_ufunc (no .persist()).

    Wraps the JMD95 equation of state, ensuring the result stays
    dask-backed and lazy — no eager scheduling on workers.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing at minimum ``Theta`` and ``Salt`` on tracer
        levels, plus all standard grid-metric variables.

    Returns
    -------
    xr.DataArray
        In-situ density [kg m⁻³] on tracer levels (dims ``k``, ``face``,
        ``j``, ``i``), dask-backed.
    """
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
#  GROUP 1: STRATIFICATION & HEAT
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
    rho = _density_lazy(ds_merge)
    zdim = _get_vertical_dim(rho)

    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)
    ref_k = _nearest_k_to_depth(z_vals, ref_depth_m)

    sigma0 = rho - 1000.0
    sigma0_ref = sigma0.isel({zdim: ref_k})
    delta_sigma = sigma0 - sigma0_ref

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
    rho = _density_lazy(ds_merge)
    drho_dz = _vertical_derivative(rho, ds_merge)

    n2 = (G / RHO0_BOUSSINESQ) * drho_dz
    n2.name = "N2"
    n2.attrs["long_name"] = "Buoyancy frequency squared"
    n2.attrs["units"] = "s^-2"
    return n2


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
    integrand = CP * RHO0_BOUSSINESQ * theta

    zdim = _get_vertical_dim(integrand)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz = _get_vertical_spacing(ds_merge, zdim=zdim)
    mask = z <= mld

    out = (integrand * dz).where(mask).sum(dim=zdim)
    out.name = "ml_heat_content"
    out.attrs["long_name"] = "Mixed-layer heat content (integrated)"
    out.attrs["units"] = "J m^-2"
    return out


def buoyancy_field_3d(ds_merge):
    """3D buoyancy field b = g ρ / ρ_ref, scaled ×1e3 to match project convention.

    Fully lazy — avoids ``.persist()`` calls.  Keeping the graph lazy is
    critical so downstream code can build flux components one at a time.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and standard grid
        variables.

    Returns
    -------
    xr.DataArray
        3D buoyancy field [m² s⁻²] on tracer levels (dims ``k``, ``face``,
        ``j``, ``i``), with ``name="buoyancy"``.
    """
    import dbof.utils.jmd95_xgcm_implementation as jmd95

    p = xr.zeros_like(ds_merge.Theta)

    rho = xr.apply_ufunc(
        jmd95.jmd95,
        ds_merge.Salt,
        ds_merge.Theta,
        p,
        dask="parallelized",
        output_dtypes=[float],
    )

    b = (G_KM * rho / RHO0_SEAWATER) * 1e3
    b.name = "buoyancy"
    b.attrs["units"] = "m^2 s^-2"
    return b


# ===========================================================================
#  GROUP 2: SHEAR & DIMENSIONLESS NUMBERS
# ===========================================================================

def vertical_shear_components_3d(ds_merge, grid):
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

    uz_model = grid.interp(dUdz, 'X', boundary='fill')
    vz_model = grid.interp(dVdz, 'Y', boundary='fill')

    uz = uz_model * ds_merge['CS'] - vz_model * ds_merge['SN']
    vz = uz_model * ds_merge['SN'] + vz_model * ds_merge['CS']
    return uz, vz


def vertical_shear_magnitude_3d(ds_merge, grid):
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
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear_mag = np.sqrt(uz**2 + vz**2)
    return shear_mag


def richardson_number_3d(ds_merge, grid):
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
    n2 = buoyancy_frequency_squared_3d(ds_merge)
    # Floor statically unstable stratification (N² < 0) to 0 so unstable
    # columns yield Ri = 0 rather than a negative Richardson number.
    n2 = n2.clip(min=0.0)
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear2 = uz**2 + vz**2
    ri = xr.where(shear2 > 0, n2 / shear2, np.nan)
    return ri


def froude_number_3d(ds_merge, grid, mld=None):
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
    n2 = buoyancy_frequency_squared_3d(ds_merge)
    n_abs = np.sqrt(np.abs(n2))

    U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
    V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')
    speed = np.sqrt(U_c**2 + V_c**2)

    denom = n_abs * mld
    fr = xr.where(denom > 0, speed / denom, np.nan)
    return fr


def rossby_number_3d(ds_merge, grid):
    """Lazy 3D Ro = ζ / f.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, grid metrics, rotation
        coefficients, and latitude coordinate ``YC``.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    xr.DataArray
        3D Rossby number Ro [dimensionless] on tracer levels.
    """
    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
        ds_merge.U, ds_merge.V, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)
    ro = zeta / f
    return ro


def burger_number_3d(ds_merge, grid, mld=None):
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
    ro_3d = rossby_number_3d(ds_merge, grid)
    fr_3d = froude_number_3d(ds_merge, grid, mld=mld)
    bu_3d = xr.where(fr_3d != 0, (ro_3d / fr_3d) ** 2, np.nan)
    return bu_3d


def _buoyancy_gradient_squared_phys_3d(ds_merge, grid):
    """Lazy 3D |∇_h b|² built on a buoyancy consistent with N².

    The buoyancy used here is b = (g / ρ₀) ρ — the *same* physical
    definition implied by N² = (g / ρ₀) dρ/dz (Boussinesq reference
    density ``RHO0_BOUSSINESQ``, no ×1e3 scaling).  This differs
    deliberately from :func:`grad_b2_3d`, which is built on
    :func:`buoyancy_field_3d` and carries a ×1e3 scaling; using that
    scaled gradient would leave the balanced Richardson number off by a
    constant 1e6 factor.  Deriving the gradient straight from density
    keeps the N² numerator and the |∇_h b|² denominator on one common
    buoyancy definition, so R_ib comes out truly dimensionless.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and horizontal grid
        metrics.
    grid : xgcm.Grid
        Grid object used for horizontal differencing.

    Returns
    -------
    xr.DataArray
        3D horizontal buoyancy gradient magnitude squared |∇_h b|² [s⁻⁴]
        on tracer levels, physically consistent with N².

    Generated by JXP and Claude
    """
    # Density from the JMD95 EOS — the same source used to build N².
    rho = _density_lazy(ds_merge)
    # Physical buoyancy b = (g / ρ₀) ρ.  Both the additive constant and
    # the overall sign vanish once the horizontal gradient is squared.
    b_phys = (G / RHO0_BOUSSINESQ) * rho
    # |∇_h b|² = (∂b/∂x)² + (∂b/∂y)² on tracer points (lazy).
    return _grad_squared_3d(b_phys, ds_merge, grid)


def balanced_richardson_number_3d(ds_merge, grid, *, n2=None, gradb2=None):
    """Lazy 3D balanced Richardson number R_ib = N² f² / |∇_h b|².

    The balanced Richardson number (Thomas, Tandon & Mahadevan 2013) is
    the gradient Richardson number of a flow in thermal-wind balance and a
    dimensionless measure of frontal stability.  N² and the horizontal
    buoyancy gradient are both derived from the *same* physically
    consistent, unscaled buoyancy (b = (g / ρ₀) ρ), so the ratio is truly
    dimensionless — see :func:`_buoyancy_gradient_squared_phys_3d`.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, latitude ``YC``,
        vertical spacing variables, and horizontal grid metrics.
    grid : xgcm.Grid
        Grid object used for horizontal differencing.
    n2 : xr.DataArray, optional
        Pre-computed 3D N² [s⁻²].  If ``None``, computed via
        :func:`buoyancy_frequency_squared_3d`.
    gradb2 : xr.DataArray, optional
        Pre-computed 3D |∇_h b|² [s⁻⁴].  Must use the physically
        consistent (unscaled) buoyancy of
        :func:`_buoyancy_gradient_squared_phys_3d`; passing the ×1e3-scaled
        :func:`grad_b2_3d` output would bias R_ib by a constant factor.
        If ``None``, computed internally.

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
        n2 = buoyancy_frequency_squared_3d(ds_merge)
    # Floor statically unstable stratification (N² < 0) to 0 so unstable
    # columns yield R_ib = 0 rather than a negative value.
    n2 = n2.clip(min=0.0)
    # |∇_h b|² denominator term, on the same buoyancy definition as N².
    if gradb2 is None:
        gradb2 = _buoyancy_gradient_squared_phys_3d(ds_merge, grid)

    # Coriolis parameter f(YC); f² enters the numerator.
    f = coriolis_parameter(ds_merge, grid)

    # R_ib = N² f² / |∇_h b|²; mask the undefined |∇_h b|² = 0 columns.
    r_ib = xr.where(gradb2 > 0, n2 * f**2 / gradb2, np.nan)
    r_ib.name = "R_ib"
    r_ib.attrs["long_name"] = "Balanced Richardson number"
    r_ib.attrs["units"] = "1"
    return r_ib


# ===========================================================================
#  GROUP 3: WIND (inherently 2D)
# ===========================================================================

def wind_stress_curl(ds_merge, grid):
    """Lazy wind stress curl.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``oceTAUX`` and ``oceTAUY`` on
        staggered velocity points, plus grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    xr.DataArray
        2D wind stress curl ∂τy/∂x − ∂τx/∂y [N m⁻³] on tracer points.
    """
    taux = ds_merge.oceTAUX
    tauy = ds_merge.oceTAUY
    _, dtaux_dphi, dtauy_dlambda, _ = ng.calculate_jacobian(
        taux, tauy, ds_merge, grid)
    curl_tau = dtauy_dlambda - dtaux_dphi
    return curl_tau


def ekman_pumping(ds_merge, grid, rho0=RHO0_BOUSSINESQ):
    """Lazy Ekman pumping w_E = curl(τ) / (ρ₀ f).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``oceTAUX``, ``oceTAUY``, grid metrics,
        and latitude coordinate ``YC``.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    rho0 : float, optional
        Reference density [kg m⁻³] (default ``RHO0_BOUSSINESQ``).

    Returns
    -------
    xr.DataArray
        2D Ekman pumping velocity w_E [m s⁻¹] on tracer points; ``NaN``
        at the equator where f = 0.
    """
    curl_tau = wind_stress_curl(ds_merge, grid)
    f = coriolis_parameter(ds_merge, grid)
    w_e = xr.where(np.abs(f) > 0, curl_tau / (rho0 * f), np.nan)
    return w_e


def _wind_stress_geographic(ds_merge, grid):
    """Lazy geographic wind stress (τ_λ, τ_φ) on tracer points.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``oceTAUX``, ``oceTAUY``, and rotation
        coefficients ``CS`` and ``SN``.
    grid : xgcm.Grid
        Grid object used for interpolation to tracer points.

    Returns
    -------
    tau_lambda : xr.DataArray
        Zonal (eastward) wind stress component [N m⁻²] on tracer points.
    tau_phi : xr.DataArray
        Meridional (northward) wind stress component [N m⁻²] on tracer
        points.
    """
    taux_c = grid.interp(ds_merge.oceTAUX, 'X', boundary='fill')
    tauy_c = grid.interp(ds_merge.oceTAUY, 'Y', boundary='fill')
    tau_lambda = taux_c * ds_merge['CS'] - tauy_c * ds_merge['SN']
    tau_phi = taux_c * ds_merge['SN'] + tauy_c * ds_merge['CS']
    return tau_lambda, tau_phi


def ekman_transport_velocity(ds_merge, grid, rho0=RHO0_BOUSSINESQ):
    """Lazy Ekman transport (u_E, v_E).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``oceTAUX``, ``oceTAUY``, rotation
        coefficients ``CS`` and ``SN``, and latitude coordinate ``YC``.
    grid : xgcm.Grid
        Grid object used for interpolation to tracer points.
    rho0 : float, optional
        Reference density [kg m⁻³] (default ``RHO0_BOUSSINESQ``).

    Returns
    -------
    dict
        Dictionary with keys:

        ``"u_ekman"`` : xr.DataArray
            Zonal Ekman transport velocity [m s⁻¹] on tracer points.
        ``"v_ekman"`` : xr.DataArray
            Meridional Ekman transport velocity [m s⁻¹] on tracer points.
    """
    f = coriolis_parameter(ds_merge, grid)
    denom = rho0 * f
    safe_denom = xr.where(np.abs(f) > 0, denom, np.nan)
    tau_lambda, tau_phi = _wind_stress_geographic(ds_merge, grid)
    return {"u_ekman": tau_phi / safe_denom,
            "v_ekman": -tau_lambda / safe_denom}


# ===========================================================================
#  GROUP 4: ADVECTIVE BUOYANCY FLUXES (lazy 3D)
# ===========================================================================

def advective_buoyancy_fluxes_3d(ds_merge, grid):
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
        Zonal advective buoyancy flux u·b [m³ s⁻³] on tracer levels.
    vB : xr.DataArray
        Meridional advective buoyancy flux v·b [m³ s⁻³] on tracer levels.
    wB : xr.DataArray
        Vertical advective buoyancy flux w·b [m³ s⁻³] on tracer levels.
    """
    b = buoyancy_field_3d(ds_merge)

    U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
    V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')
    u_geo = U_c * ds_merge['CS'] - V_c * ds_merge['SN']
    v_geo = U_c * ds_merge['SN'] + V_c * ds_merge['CS']

    W_c = _interp_w_to_tracer_levels(ds_merge)

    return u_geo * b, v_geo * b, W_c * b


# ===========================================================================
#  GROUP 5: ERTEL PV (lazy 3D)
# ===========================================================================

def ertel_pv_terms_3d(ds_merge, grid):
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
            Total Ertel PV q [m⁻¹ s⁻¹] on tracer levels.
        ``"ertel_pv_vertical"`` : xr.DataArray
            Vertical component q_vert = (ζ + f) b_z [m⁻¹ s⁻¹].
        ``"ertel_pv_tilt"`` : xr.DataArray
            Tilting component q_tilt [m⁻¹ s⁻¹].
    """
    U = ds_merge.U
    V = ds_merge.V

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(U, V, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)

    b = buoyancy_field_3d(ds_merge)
    b_x, b_y = ng.calculate_native_gradient_tracer(
        b, ds_merge, grid=grid)
    b_z = _vertical_derivative(b, ds_merge)

    u_z, v_z = vertical_shear_components_3d(ds_merge, grid)

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


# ===========================================================================
#  GROUP 6: SCALAR GRADIENTS & FRONTAL STRUCTURE (lazy 3D)
# ===========================================================================

def _grad_squared_3d(scalar_3d, ds_merge, grid):
    """Lazy |∇s|² from a tracer-point 3D field.

    Parameters
    ----------
    scalar_3d : xr.DataArray
        3D scalar field on tracer points from which the horizontal gradient
        is computed.
    ds_merge : xr.Dataset
        Merged dataset providing grid metrics for gradient calculation.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    xr.DataArray
        3D horizontal gradient magnitude squared |∇s|² on tracer levels,
        in units of (input units / m)².
    """
    gx, gy = ng.calculate_native_gradient_tracer(
        scalar_3d, ds_merge, grid=grid)
    return gx**2 + gy**2


def grad_theta2_3d(ds_merge, grid):
    """Lazy 3D |∇θ|².  Units: (K/m)².

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta`` and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    xr.DataArray
        3D horizontal temperature gradient squared |∇θ|² [(K m⁻¹)²] on
        tracer levels.
    """
    return _grad_squared_3d(ds_merge.Theta, ds_merge, grid)


def grad_salt2_3d(ds_merge, grid):
    """Lazy 3D |∇S|².  Units: (PSU/m)².

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Salt`` and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    xr.DataArray
        3D horizontal salinity gradient squared |∇S|² [(PSU m⁻¹)²] on
        tracer levels.
    """
    return _grad_squared_3d(ds_merge.Salt, ds_merge, grid)


def grad_rho2_3d(ds_merge, grid):
    """Lazy 3D |∇ρ|².  Derives density via JMD95.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    xr.DataArray
        3D horizontal density gradient squared |∇ρ|² [(kg m⁻⁴)²] on
        tracer levels.
    """
    rho = _density_lazy(ds_merge)
    return _grad_squared_3d(rho, ds_merge, grid)


def grad_b2_3d(ds_merge, grid):
    """Lazy 3D |∇b|².  Units: s⁻⁴.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    xr.DataArray
        3D horizontal buoyancy gradient squared |∇b|² [s⁻⁴] on tracer
        levels.
    """
    b = buoyancy_field_3d(ds_merge)
    return _grad_squared_3d(b, ds_merge, grid)


def grad_eta2(ds_merge, grid):
    """Lazy |∇η|² — inherently 2D (surface only).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing sea-surface height ``Eta`` and grid
        metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    xr.DataArray
        2D sea-surface height gradient squared |∇η|² [m² m⁻²] on tracer
        points (dims ``face``, ``j``, ``i``).
    """
    gx, gy = ng.calculate_native_gradient_tracer(
        ds_merge.Eta, ds_merge, grid=grid,
    )
    return gx**2 + gy**2


def turner_angle_3d(ds_merge, grid, *, gradtheta2=None, gradsalt2=None,
                    gradrho2=None):
    """Lazy 3D horizontal Turner angle (degrees).

    Tu_h = arctan(ρ₀(β²|∇S|² − α²|∇θ|²) / (−|∇ρ|²/ρ₀))

    Optional kwargs accept pre-computed 3D gradient fields to avoid
    recomputation when the caller already has them (e.g. Turner angle
    shares gradtheta2/gradsalt2/gradrho2 with the frontal_structure subset).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.
    gradtheta2 : xr.DataArray, optional
        Pre-computed |∇θ|² [(K m⁻¹)²].  If ``None``, computed internally.
    gradsalt2 : xr.DataArray, optional
        Pre-computed |∇S|² [(PSU m⁻¹)²].  If ``None``, computed internally.
    gradrho2 : xr.DataArray, optional
        Pre-computed |∇ρ|² [(kg m⁻⁴)²].  If ``None``, computed internally.

    Returns
    -------
    xr.DataArray
        3D horizontal Turner angle [degrees] on tracer levels; ``NaN``
        where |∇ρ|² = 0.
    """
    if gradtheta2 is None:
        gradtheta2 = grad_theta2_3d(ds_merge, grid)
    if gradsalt2 is None:
        gradsalt2 = grad_salt2_3d(ds_merge, grid)
    if gradrho2 is None:
        gradrho2 = grad_rho2_3d(ds_merge, grid)

    numer = RHO0_SEAWATER * (BETA**2 * gradsalt2 - ALPHA**2 * gradtheta2)
    denom = xr.where(gradrho2 > 0, -gradrho2 / RHO0_SEAWATER, np.nan)
    return np.degrees(np.arctan(numer / denom))


# ===========================================================================
#  GROUP 6b: VELOCITY PROPERTIES (lazy 3D)
# ===========================================================================

def compute_velocity_jacobian_3d(ds_merge, grid):
    """3D velocity Jacobian → VelocityJacobian(du_dx, du_dy, dv_dx, dv_dy).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, grid metrics, and rotation
        coefficients ``CS`` and ``SN``.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    VelocityJacobian
        Named tuple with fields ``(du_dx, du_dy, dv_dx, dv_dy)``, each a
        dask-backed ``xr.DataArray`` on tracer levels [s⁻¹].
    """
    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
        ds_merge.U, ds_merge.V, ds_merge, grid,
    )
    return VelocityJacobian(du_dx, du_dy, dv_dx, dv_dy)


def relative_vorticity_3d(ds_merge, grid, *, jacobian=None):
    """Lazy 3D ζ = dv/dx − du/dy.  Accepts optional *jacobian*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, grid metrics, and rotation
        coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  If ``None``, computed internally.

    Returns
    -------
    xr.DataArray
        3D relative vorticity ζ [s⁻¹] on tracer levels.
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian_3d(ds_merge, grid)
    return jacobian.dv_dx - jacobian.du_dy


def strain_3d(ds_merge, grid, *, jacobian=None):
    """Lazy 3D strain → (strain_mag, strain_n, strain_s).  Accepts optional *jacobian*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, grid metrics, and rotation
        coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  If ``None``, computed internally.

    Returns
    -------
    sm : xr.DataArray
        3D strain magnitude |S| = sqrt(Sn² + Ss²) [s⁻¹] on tracer levels.
    sn : xr.DataArray
        3D normal strain Sn = du/dx − dv/dy [s⁻¹] on tracer levels.
    ss : xr.DataArray
        3D shear strain Ss = du/dy + dv/dx [s⁻¹] on tracer levels.
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian_3d(ds_merge, grid)
    sn = jacobian.du_dx - jacobian.dv_dy
    ss = jacobian.du_dy + jacobian.dv_dx
    sm = np.sqrt(sn**2 + ss**2)
    return sm, sn, ss


def divergence_3d(ds_merge, grid, *, jacobian=None):
    """Lazy 3D divergence = du/dx + dv/dy.  Accepts optional *jacobian*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, grid metrics, and rotation
        coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  If ``None``, computed internally.

    Returns
    -------
    xr.DataArray
        3D horizontal divergence δ = du/dx + dv/dy [s⁻¹] on tracer levels.
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian_3d(ds_merge, grid)
    divergence = jacobian.du_dx + jacobian.dv_dy
    return divergence


def okubo_weiss_3d(ds_merge, grid, *, jacobian=None):
    """Lazy 3D Okubo-Weiss = Sn² + Ss² − ζ².  Accepts optional *jacobian*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``U``, ``V``, grid metrics, and rotation
        coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  If ``None``, computed internally.

    Returns
    -------
    xr.DataArray
        3D Okubo-Weiss parameter W = Sn² + Ss² − ζ² [s⁻²] on tracer
        levels; positive values indicate strain-dominated regions.
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian_3d(ds_merge, grid)
    omega = relative_vorticity_3d(ds_merge, grid, jacobian=jacobian)
    _, sn, ss = strain_3d(ds_merge, grid, jacobian=jacobian)
    return sn**2 + ss**2 - omega**2



# ===========================================================================
#  GROUP 7: FRONTOGENESIS (lazy 3D)
# ===========================================================================

def compute_buoyancy_gradients_3d(ds_merge, grid):
    """3D buoyancy gradients → BuoyancyGradients(zonal, merid).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    BuoyancyGradients
        Named tuple with fields:

        ``zonal`` : xr.DataArray
            Zonal buoyancy gradient ∂b/∂x [m s⁻²  m⁻¹] on tracer levels.
        ``merid`` : xr.DataArray
            Meridional buoyancy gradient ∂b/∂y [m s⁻² m⁻¹] on tracer
            levels.
    """
    b = buoyancy_field_3d(ds_merge)
    zonal, merid = ng.calculate_native_gradient_tracer(
        b, ds_merge, grid=grid,
    )
    return BuoyancyGradients(zonal, merid)


def _frontogenesis_formula_3d(du_dx, du_dy, dv_dx, dv_dy, grad_bx, grad_by):
    """Kinematic frontogenesis tendency from velocity and buoyancy gradients.

    F = -(du/dx · bx² + (du/dy + dv/dx) · bx·by + dv/dy · by²)

    Internal helper shared by ``frontogenesis_tendency_3d`` and
    ``frontogenesis_geo_3d``.

    Parameters
    ----------
    du_dx : xr.DataArray
        Zonal velocity gradient ∂u/∂x [s⁻¹] on tracer levels.
    du_dy : xr.DataArray
        Cross-stream velocity gradient ∂u/∂y [s⁻¹] on tracer levels.
    dv_dx : xr.DataArray
        Along-stream velocity gradient ∂v/∂x [s⁻¹] on tracer levels.
    dv_dy : xr.DataArray
        Meridional velocity gradient ∂v/∂y [s⁻¹] on tracer levels.
    grad_bx : xr.DataArray
        Zonal buoyancy gradient ∂b/∂x [s⁻² m⁻¹] on tracer levels.
    grad_by : xr.DataArray
        Meridional buoyancy gradient ∂b/∂y [s⁻² m⁻¹] on tracer levels.

    Returns
    -------
    xr.DataArray
        3D frontogenesis tendency F [s⁻⁵] on tracer levels.
    """
    return -(du_dx * grad_bx**2
             + (du_dy + dv_dx) * grad_bx * grad_by
             + dv_dy * grad_by**2)


def frontogenesis_tendency_3d(ds_merge, grid, *, jacobian=None,
                              buoyancy_gradients=None):
    """Lazy 3D frontogenesis tendency F(u, v).  Accepts optional *jacobian* and *buoyancy_gradients*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Theta``, ``Salt``, ``U``, ``V``,
        grid metrics, and rotation coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  If ``None``, computed internally.
    buoyancy_gradients : BuoyancyGradients, optional
        Pre-computed buoyancy gradients.  If ``None``, computed internally.

    Returns
    -------
    xr.DataArray
        3D kinematic frontogenesis tendency F(u, v) [s⁻⁵] on tracer levels.
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian_3d(ds_merge, grid)
    if buoyancy_gradients is None:
        buoyancy_gradients = compute_buoyancy_gradients_3d(ds_merge, grid)

    return _frontogenesis_formula_3d(
        jacobian.du_dx, jacobian.du_dy,
        jacobian.dv_dx, jacobian.dv_dy,
        buoyancy_gradients.zonal, buoyancy_gradients.merid,
    )


def geostrophic_velocity_3d(ds_merge, grid):
    """Geostrophic velocity → (ug, vg) from ∂η/∂x, ∂η/∂y.

    Eta is inherently 2D so ug/vg are surface-only; named ``_3d`` for
    module consistency (they broadcast against 3D buoyancy gradients).

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing sea-surface height ``Eta``, latitude
        coordinate ``YC``, and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing.

    Returns
    -------
    ug : xr.DataArray
        Zonal geostrophic velocity ug = −(g/f) ∂η/∂y [m s⁻¹] on tracer
        points.
    vg : xr.DataArray
        Meridional geostrophic velocity vg = (g/f) ∂η/∂x [m s⁻¹] on tracer
        points.
    """
    f = coriolis_parameter(ds_merge, grid)
    eta_grad_x, eta_grad_y = ng.calculate_native_gradient_tracer(
        ds_merge['Eta'], ds_merge, grid=grid,
    )
    ug = -(G / f) * eta_grad_y
    vg =  (G / f) * eta_grad_x
    return ug, vg


def frontogenesis_geo_3d(ds_merge, grid, *, ug=None, vg=None,
                         buoyancy_gradients=None):
    """Lazy 3D geostrophic frontogenesis F(ug, vg).  Accepts optional *ug*, *vg*, *buoyancy_gradients*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``Eta``, ``Theta``, ``Salt``, latitude
        ``YC``, and grid metrics.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    ug : xr.DataArray, optional
        Pre-computed zonal geostrophic velocity [m s⁻¹].  If ``None``,
        computed internally together with ``vg``.
    vg : xr.DataArray, optional
        Pre-computed meridional geostrophic velocity [m s⁻¹].  If ``None``,
        computed internally together with ``ug``.
    buoyancy_gradients : BuoyancyGradients, optional
        Pre-computed buoyancy gradients.  If ``None``, computed internally.

    Returns
    -------
    xr.DataArray
        3D geostrophic frontogenesis tendency F(ug, vg) [s⁻⁵] on tracer
        levels.
    """
    if ug is None or vg is None:
        ug, vg = geostrophic_velocity_3d(ds_merge, grid)
    if buoyancy_gradients is None:
        buoyancy_gradients = compute_buoyancy_gradients_3d(ds_merge, grid)

    dug_dx, dug_dy = ng.calculate_native_gradient_tracer(
        ug, ds_merge, grid=grid,
    )
    dvg_dx, dvg_dy = ng.calculate_native_gradient_tracer(
        vg, ds_merge, grid=grid,
    )
    return _frontogenesis_formula_3d(
        dug_dx, dug_dy,
        dvg_dx, dvg_dy,
        buoyancy_gradients.zonal, buoyancy_gradients.merid,
    )

