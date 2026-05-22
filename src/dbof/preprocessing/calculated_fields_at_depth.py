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
from dbof.preprocessing.calculate_additional_fields import coriolis_parameter

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


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO0 = 1000.0   # kg m⁻³  reference density
G    = 9.81      # m s⁻²  gravitational acceleration
CP   = 3994.0    # J kg⁻¹ K⁻¹  seawater specific heat capacity
OMEGA_EARTH = 7.292115e-5  # rad s⁻¹

MLD_REFERENCE_DEPTH_M = 10.0  # metres — Bodner et al. reference depth (≈ 9.66 m)


# ===========================================================================
#  DENSITY
# ===========================================================================

def _density_lazy(ds_merge):
    """Compute density lazily via apply_ufunc (no .persist()).

    Wraps the JMD95 equation of state, ensuring the result stays
    dask-backed and lazy — no eager scheduling on workers.  
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

    b = (g * rho / ref_rho) * 1e3
    b.name = "buoyancy"
    b.attrs["units"] = "m^2 s^-2"
    return b


# ===========================================================================
#  GROUP 2: SHEAR & DIMENSIONLESS NUMBERS
# ===========================================================================

def vertical_shear_components_3d(ds_merge, grid):
    """Lazy 3D vertical shear (du/dz, dv/dz) in geographic coordinates."""
    dUdz = _vertical_derivative(ds_merge.U, ds_merge)
    dVdz = _vertical_derivative(ds_merge.V, ds_merge)

    uz_model = grid.interp(dUdz, 'X', boundary='fill')
    vz_model = grid.interp(dVdz, 'Y', boundary='fill')

    uz = uz_model * ds_merge['CS'] - vz_model * ds_merge['SN']
    vz = uz_model * ds_merge['SN'] + vz_model * ds_merge['CS']
    return uz, vz


def vertical_shear_magnitude_3d(ds_merge, grid):
    """Lazy 3D |S| = sqrt(uz² + vz²)."""
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear_mag = np.sqrt(uz**2 + vz**2)
    return shear_mag


def richardson_number_3d(ds_merge, grid):
    """Lazy 3D Ri = N² / (uz² + vz²)."""
    n2 = buoyancy_frequency_squared_3d(ds_merge)
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear2 = uz**2 + vz**2
    ri = xr.where(shear2 > 0, n2 / shear2, np.nan)
    return ri


def froude_number_3d(ds_merge, grid, mld=None):
    """Lazy 3D Fr = speed / (N * MLD)."""
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
    """Lazy 3D Ro = ζ / f."""
    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
        ds_merge.U, ds_merge.V, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)
    ro = zeta / f
    return ro


def burger_number_3d(ds_merge, grid, mld=None):
    """Lazy 3D Bu = (Ro / Fr)². """
    if mld is None:
        mld = mixed_layer_depth(ds_merge)
    ro_3d = rossby_number_3d(ds_merge, grid)
    fr_3d = froude_number_3d(ds_merge, grid, mld=mld)
    bu_3d = xr.where(fr_3d != 0, (ro_3d / fr_3d) ** 2, np.nan)
    return bu_3d


# ===========================================================================
#  GROUP 3: WIND (inherently 2D)
# ===========================================================================

def wind_stress_curl(ds_merge, grid):
    """Lazy wind stress curl."""
    taux = ds_merge.oceTAUX
    tauy = ds_merge.oceTAUY
    _, dtaux_dphi, dtauy_dlambda, _ = ng.calculate_jacobian(
        taux, tauy, ds_merge, grid)
    curl_tau = dtauy_dlambda - dtaux_dphi
    return curl_tau


def ekman_pumping(ds_merge, grid, rho0=RHO0):
    """Lazy Ekman pumping w_E = curl(τ) / (ρ₀ f)."""
    curl_tau = wind_stress_curl(ds_merge, grid)
    f = coriolis_parameter(ds_merge, grid)
    w_e = xr.where(np.abs(f) > 0, curl_tau / (rho0 * f), np.nan)
    return w_e


def _wind_stress_geographic(ds_merge, grid):
    """Lazy geographic wind stress (τ_λ, τ_φ) on tracer points."""
    taux_c = grid.interp(ds_merge.oceTAUX, 'X', boundary='fill')
    tauy_c = grid.interp(ds_merge.oceTAUY, 'Y', boundary='fill')
    tau_lambda = taux_c * ds_merge['CS'] - tauy_c * ds_merge['SN']
    tau_phi = taux_c * ds_merge['SN'] + tauy_c * ds_merge['CS']
    return tau_lambda, tau_phi


def ekman_transport_velocity(ds_merge, grid, rho0=RHO0):
    """Lazy Ekman transport (u_E, v_E)."""
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
    """Lazy (uB, vB, wB) in geographic coordinates."""
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

    Returns dict with keys: ertel_pv, ertel_pv_vertical, ertel_pv_tilt.
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
#  GROUP 6: SCALAR GRADIENT & VELOCITY PROPERTIES (lazy 3D)
# ===========================================================================

def _grad_squared_3d(scalar_3d, ds_merge, grid):
    """Lazy |∇s|² from a tracer-point 3D field."""
    gx, gy = ng.calculate_native_gradient_tracer(
        scalar_3d, ds_merge, grid=grid)
    return gx**2 + gy**2


def _velocity_properties_3d(ds_merge, grid):
    """Lazy 3D kinematic fields from the velocity Jacobian.

    Returns dict with keys: relative_vorticity, strain_n, strain_s,
    strain_mag, divergence, okubo_weiss.
    """
    U = ds_merge.U
    V = ds_merge.V

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(U, V, ds_merge, grid)

    omega = dv_dx - du_dy
    strain_n = du_dx - dv_dy
    strain_s = du_dy + dv_dx
    strain_mag = np.sqrt(strain_n**2 + strain_s**2)
    div = du_dx + dv_dy
    ow = strain_n**2 + strain_s**2 - omega**2

    return {
        "relative_vorticity": omega,
        "strain_n": strain_n,
        "strain_s": strain_s,
        "strain_mag": strain_mag,
        "divergence": div,
        "okubo_weiss": ow,
    }


# ===========================================================================
#  GROUP 7: FRONTOGENESIS (lazy 3D)
# ===========================================================================

def _frontogenesis_fields_3d(ds_merge, grid):
    """Lazy 3D frontogenesis tendency and geo/ageo decomposition.

    Returns dict with keys: frontogenesis_tendency, frontogenesis_geo,
    frontogenesis_ageo, ug, vg.
    """
    U = ds_merge.U
    V = ds_merge.V

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(U, V, ds_merge, grid)

    b = buoyancy_field_3d(ds_merge)
    grad_bx, grad_by = ng.calculate_native_gradient_tracer(
        b, ds_merge, grid=grid)

    F_full = -(du_dx * grad_bx**2
               + (du_dy + dv_dx) * grad_bx * grad_by
               + dv_dy * grad_by**2)

    f = coriolis_parameter(ds_merge, grid)
    g_accel = 9.81
    eta_grad_x, eta_grad_y = ng.calculate_native_gradient_tracer(
        ds_merge['Eta'], ds_merge, grid=grid)
    ug = -(g_accel / f) * eta_grad_y
    vg = (g_accel / f) * eta_grad_x

    gx_ug, gy_ug = ng.calculate_native_gradient_tracer(
        ug, ds_merge, grid=grid)
    gx_vg, gy_vg = ng.calculate_native_gradient_tracer(
        vg, ds_merge, grid=grid)
    F_geo = -(gx_ug * grad_bx**2
              + (gy_ug + gx_vg) * grad_bx * grad_by
              + gy_vg * grad_by**2)

    F_ageo = F_full - F_geo

    return {
        "frontogenesis_tendency": F_full,
        "frontogenesis_geo": F_geo,
        "frontogenesis_ageo": F_ageo,
        "ug": ug,
        "vg": vg,
    }
