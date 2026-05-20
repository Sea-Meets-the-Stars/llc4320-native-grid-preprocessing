"""
Fully-lazy dask pipeline for LLC4320 depth-resolved diagnostics.

Strategy
--------
Build the complete 3D computation graph lazily — no ``.compute()`` or
``_materialise_for_xgcm()`` calls on intermediate fields.  A single
``dask.compute()`` at the end of each ``compute_*`` function materialises
only the final 2D outputs.

Chunking contract
-----------------
Inputs are opened with **native zarr chunks** — no rechunking.

On-disk layout:  ``{face: 1, k: 51, j: 720, i: 720}``

Each chunk is ~1.5 GB (1 face × 51 k × 720 j × 720 i × 8 bytes),
giving ~468 chunks per variable.  This is the same pattern used by
``generate_global.py`` for 2D surface fields: dask + xgcm operate
directly on the spatially-chunked data.

The generate script (``generate_global_depth_dask.py``) is responsible
for opening the local zarr cache without overriding chunk layout.

Compared to other pipelines
----------------------------
* **original** (per-k materialise + xgcm map_overlap):
  ``.compute()`` at every k-level, map_overlap overhead repeated 51×
  per field → ~8.5 hrs per gradient field.
* **horiz** (per-face numpy stencils):
  No xgcm, but explicit face loop + halo extraction + numpy stencils.
* **THIS** (fully lazy dask):
  Native chunks, lazy graph, one ``.compute()`` per subset at the end.
"""

import logging

import numpy as np
import xarray as xr
import dask

import dbof.utils.native_gradient as ng
from dbof.preprocessing.calculate_additional_fields import coriolis_parameter

from dbof.preprocessing.calculated_fields_at_depth import (
    # -- vertical helpers --
    _get_vertical_dim,
    _get_depth_coord,
    _get_vertical_spacing,
    _nearest_k_to_depth,
    _vertical_derivative,
    _interp_w_to_tracer_levels,

    # -- density --
    _density_lazy,

    # -- depth-strategy helpers (used by lazy strategies below) --
    FIXED_DEPTH_M,
    _extract_at_mld,
    _masked_ml_mean,

    # -- 3D field builders that are already fully lazy --
    mixed_layer_depth,
    buoyancy_frequency_squared_3d,
    buoyancy_field_3d,
    mixed_layer_heat_content,

    # -- constants --
    RHO0,
    G,
    CP,
    OMEGA_EARTH,
)


# ===========================================================================
#  LAZY DEPTH STRATEGIES
# ===========================================================================
# All operations return dask-backed DataArrays.  No .compute() calls.

def apply_depth_strategies(field3d, field_base_name, ds_merge, mld=None,
                           requested=None):
    """Apply depth strategies to a lazy 3D field → dict of lazy 2D results.

    Unlike the original ``apply_depth_strategies`` this never uses
    ``field_at_k`` closures or per-k loops.  It operates on the full
    lazy 3D array directly.  All returned DataArrays stay dask-backed
    until the caller's final ``dask.compute()``.
    """
    from dbof.preprocessing.calculated_fields_at_depth import DEPTH_STRATEGIES

    if requested is None:
        requested_suffixes = set(DEPTH_STRATEGIES)
    else:
        requested_suffixes = {
            s for s in DEPTH_STRATEGIES
            if f"{field_base_name}_{s}" in requested
        }

    zdim = _get_vertical_dim(field3d)
    results = {}

    for suffix in requested_suffixes:
        key = f"{field_base_name}_{suffix}"
        if suffix == "sfc":
            results[key] = field3d.isel({zdim: 0})
        elif suffix == "z25m":
            z = _get_depth_coord(ds_merge, zdim=zdim)
            k25 = _nearest_k_to_depth(z.values.astype(np.float64),
                                      FIXED_DEPTH_M)
            results[key] = field3d.isel({zdim: k25})
        elif suffix == "mld":
            results[key] = _extract_at_mld(field3d, mld, ds_merge)
        elif suffix == "mld_mean":
            results[key] = _masked_ml_mean(field3d, mld, ds_merge)

    return results


# ---------------------------------------------------------------------------
#  Materialise helper — single .compute() for all results in a subset
# ---------------------------------------------------------------------------

def _materialise_results(results):
    """dask.compute() every lazy value in *results*, return numpy-backed dict."""
    if not results:
        return results
    keys = list(results.keys())
    lazy_vals = [results[k] for k in keys]
    materialised = dask.compute(*lazy_vals, retries=10)
    return dict(zip(keys, materialised))


# ===========================================================================
#  3D FIELD FUNCTIONS — FULLY LAZY (no _materialise_for_xgcm)
# ===========================================================================

# --- Vertical shear & dimensionless numbers --------------------------------

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
    return np.sqrt(uz**2 + vz**2)


def richardson_number_3d(ds_merge, grid):
    """Lazy 3D Ri = N² / (uz² + vz²)."""
    n2 = buoyancy_frequency_squared_3d(ds_merge)
    uz, vz = vertical_shear_components_3d(ds_merge, grid)
    shear2 = uz**2 + vz**2
    return xr.where(shear2 > 0, n2 / shear2, np.nan)


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
    return xr.where(denom > 0, speed / denom, np.nan)


def rossby_number_3d(ds_merge, grid):
    """Lazy 3D Ro = ζ / f."""
    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
        ds_merge.U, ds_merge.V, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)
    return zeta / f


def burger_number_2d(ds_merge, grid, mld=None):
    """Lazy 2D Bu = (Ro / Fr)² evaluated at MLD."""
    if mld is None:
        mld = mixed_layer_depth(ds_merge)
    ro_3d = rossby_number_3d(ds_merge, grid)
    fr_3d = froude_number_3d(ds_merge, grid, mld=mld)
    ro_at_mld = _extract_at_mld(ro_3d, mld, ds_merge)
    fr_at_mld = _extract_at_mld(fr_3d, mld, ds_merge)
    return xr.where(fr_at_mld != 0, (ro_at_mld / fr_at_mld)**2, np.nan)


# --- Wind (inherently 2D) --------------------------------------------------

def wind_stress_curl(ds_merge, grid):
    """Lazy wind stress curl."""
    taux = ds_merge.oceTAUX
    tauy = ds_merge.oceTAUY
    _, dtaux_dphi, dtauy_dlambda, _ = ng.calculate_jacobian(
        taux, tauy, ds_merge, grid)
    return dtauy_dlambda - dtaux_dphi


def ekman_pumping(ds_merge, grid, rho0=RHO0):
    """Lazy Ekman pumping w_E = curl(τ) / (ρ₀ f)."""
    curl_tau = wind_stress_curl(ds_merge, grid)
    f = coriolis_parameter(ds_merge, grid)
    return xr.where(np.abs(f) > 0, curl_tau / (rho0 * f), np.nan)


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


# --- Advective buoyancy fluxes (lazy 3D) -----------------------------------

def advective_buoyancy_fluxes_3d(ds_merge, grid):
    """Lazy (uB, vB, wB) in geographic coordinates."""
    b = buoyancy_field_3d(ds_merge)

    # DEBUG: test 2D slice vs 3D to isolate xgcm map_overlap failure
    import logging as _log
    U = ds_merge.U
    _log.info(f"  DEBUG U dims={U.dims} shape={U.shape} chunks={U.chunks}")
    _log.info(f"  DEBUG trying grid.interp on 2D slice U.isel(k=0)...")
    _test = grid.interp(U.isel(k=0), 'X', boundary='fill')
    _log.info(f"  DEBUG 2D interp succeeded! Result shape={_test.shape}")
    _log.info(f"  DEBUG now trying full 3D grid.interp...")

    U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
    V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')
    u_geo = U_c * ds_merge['CS'] - V_c * ds_merge['SN']
    v_geo = U_c * ds_merge['SN'] + V_c * ds_merge['CS']

    W_c = _interp_w_to_tracer_levels(ds_merge)

    return u_geo * b, v_geo * b, W_c * b


# --- Ertel PV (lazy 3D) ----------------------------------------------------

def ertel_pv_terms_3d(ds_merge, grid):
    """Lazy Ertel PV and its vertical/tilting decomposition.

    q = (ζ + f) b_z  +  (w_y - v_z) b_x  +  (u_z - w_x) b_y
        ─────────────    ──────────────────────────────────────
          q_vert                    q_tilt

    Returns dict with keys: ertel_pv, ertel_pv_vertical, ertel_pv_tilt.
    """
    U = ds_merge.U
    V = ds_merge.V

    # Horizontal vorticity via Jacobian
    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(U, V, ds_merge, grid)
    zeta = dv_dx - du_dy
    f = coriolis_parameter(ds_merge, grid)

    # Buoyancy and its derivatives
    b = buoyancy_field_3d(ds_merge)
    b_x, b_y = ng.calculate_native_gradient_tracer(
        b, ds_merge, grid=grid)
    b_z = _vertical_derivative(b, ds_merge)

    # Vertical shear (geographic)
    u_z, v_z = vertical_shear_components_3d(ds_merge, grid)

    # W horizontal gradients
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


# --- Scalar gradient squared (lazy 3D) -------------------------------------

def _grad_squared_3d(scalar_3d, ds_merge, grid):
    """Lazy |∇s|² from a tracer-point 3D field."""
    gx, gy = ng.calculate_native_gradient_tracer(
        scalar_3d, ds_merge, grid=grid)
    return gx**2 + gy**2


# --- Velocity Jacobian properties (lazy 3D) --------------------------------

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


# --- Frontogenesis (lazy 3D) -----------------------------------------------

def _frontogenesis_fields_3d(ds_merge, grid):
    """Lazy 3D frontogenesis tendency and geo/ageo decomposition.

    Returns dict with keys: frontogenesis_tendency, frontogenesis_geo,
    frontogenesis_ageo, ug, vg.
    """
    U = ds_merge.U
    V = ds_merge.V

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(U, V, ds_merge, grid)

    # Buoyancy gradient
    b = buoyancy_field_3d(ds_merge)
    grad_bx, grad_by = ng.calculate_native_gradient_tracer(
        b, ds_merge, grid=grid)

    # Full frontogenesis
    F_full = -(du_dx * grad_bx**2
               + (du_dy + dv_dx) * grad_bx * grad_by
               + dv_dy * grad_by**2)

    # Geostrophic velocity from SSH
    f = coriolis_parameter(ds_merge, grid)
    g_accel = 9.81
    eta_grad_x, eta_grad_y = ng.calculate_native_gradient_tracer(
        ds_merge['Eta'], ds_merge, grid=grid)
    ug = -(g_accel / f) * eta_grad_y
    vg = (g_accel / f) * eta_grad_x

    # Geostrophic frontogenesis
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


# ===========================================================================
#  COMPUTE FUNCTIONS (entry points called by the generate script)
# ===========================================================================
# Pattern: build lazy 3D fields → apply lazy depth strategies →
#          single dask.compute() → return dict of numpy-backed DataArrays.

def compute_stratification(ds_merge, grid, computed_feature_channels):
    """Subset: stratification — MLD, N², mixed-layer heat content."""
    results = {}
    requested = set(computed_feature_channels)

    mld = mixed_layer_depth(ds_merge)   # lazy
    if "mixed_layer_depth" in requested:
        results["mixed_layer_depth"] = mld

    if any(c.startswith("N2_") for c in requested):
        n2_3d = buoyancy_frequency_squared_3d(ds_merge)   # lazy
        results.update(apply_depth_strategies(
            n2_3d, "N2", ds_merge, mld=mld, requested=requested))

    if "ml_heat_content" in requested:
        results["ml_heat_content"] = mixed_layer_heat_content(
            ds_merge, mld=mld)

    return _materialise_results(results)


def compute_vertical_shear(ds_merge, grid, computed_feature_channels):
    """Subset: vertical_shear — |S| and Ri at four depths."""
    results = {}
    requested = set(computed_feature_channels)

    shear_requested = any(c.startswith("vertical_shear_") for c in requested)
    ri_requested = any(c.startswith("Ri_") for c in requested)
    if not shear_requested and not ri_requested:
        return results

    mld = mixed_layer_depth(ds_merge)   # lazy

    if shear_requested:
        shear_3d = vertical_shear_magnitude_3d(ds_merge, grid)   # lazy
        results.update(apply_depth_strategies(
            shear_3d, "vertical_shear", ds_merge, mld=mld,
            requested=requested))

    if ri_requested:
        ri_3d = richardson_number_3d(ds_merge, grid)   # lazy
        results.update(apply_depth_strategies(
            ri_3d, "Ri", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_mixing_parameters(ds_merge, grid, computed_feature_channels):
    """Subset: mixing_parameters — Fr, Ro, Burger at four depths."""
    results = {}
    requested = set(computed_feature_channels)

    mld = mixed_layer_depth(ds_merge)   # lazy

    if any(c.startswith("Fr_") for c in requested):
        fr_3d = froude_number_3d(ds_merge, grid, mld=mld)   # lazy
        results.update(apply_depth_strategies(
            fr_3d, "Fr", ds_merge, mld=mld, requested=requested))

    if any(c.startswith("Ro_") for c in requested):
        ro_3d = rossby_number_3d(ds_merge, grid)   # lazy
        results.update(apply_depth_strategies(
            ro_3d, "Ro", ds_merge, mld=mld, requested=requested))

    if "Burger_number" in requested:
        results["Burger_number"] = burger_number_2d(ds_merge, grid, mld=mld)

    return _materialise_results(results)


def compute_ertel_pv(ds_merge, grid, computed_feature_channels):
    """Subset: ertel_pv — Ertel PV terms at four depths."""
    results = {}
    requested = set(computed_feature_channels)

    pv_bases = ("ertel_pv", "ertel_pv_vertical", "ertel_pv_tilt")
    active_bases = [b for b in pv_bases
                    if any(c.startswith(b + "_") for c in requested)]
    if not active_bases:
        return results

    mld = mixed_layer_depth(ds_merge)   # lazy
    pv_dict = ertel_pv_terms_3d(ds_merge, grid)   # lazy 3D fields

    for base in active_bases:
        results.update(apply_depth_strategies(
            pv_dict[base], base, ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_buoyancy_fluxes(ds_merge, grid, computed_feature_channels):
    """Subset: buoyancy_fluxes — uB, vB, wB at four depths."""
    results = {}
    requested = set(computed_feature_channels)

    flux_bases = ("uB", "vB", "wB")
    if not any(ch.startswith(b) for ch in requested for b in flux_bases):
        return results

    mld = mixed_layer_depth(ds_merge)   # lazy
    uB, vB, wB = advective_buoyancy_fluxes_3d(ds_merge, grid)   # lazy

    if any(c.startswith("uB_") for c in requested):
        results.update(apply_depth_strategies(
            uB, "uB", ds_merge, mld=mld, requested=requested))
    if any(c.startswith("vB_") for c in requested):
        results.update(apply_depth_strategies(
            vB, "vB", ds_merge, mld=mld, requested=requested))
    if any(c.startswith("wB_") for c in requested):
        results.update(apply_depth_strategies(
            wB, "wB", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_energetics(ds_merge, grid, computed_feature_channels):
    """Subset: energetics — KE = 0.5 * (MLD * |∇b| / f)²."""
    results = {}
    requested = set(computed_feature_channels)

    if not any(c.startswith("KE_") for c in requested):
        return results

    mld = mixed_layer_depth(ds_merge)   # lazy
    b = buoyancy_field_3d(ds_merge)   # lazy
    f = coriolis_parameter(ds_merge, grid)

    grad_b2 = _grad_squared_3d(b, ds_merge, grid)   # lazy
    grad_b_mag = np.sqrt(grad_b2)
    ke_3d = 0.5 * (mld * grad_b_mag / f)**2   # lazy

    results.update(apply_depth_strategies(
        ke_3d, "KE", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_frontal_structure(ds_merge, grid, computed_feature_channels):
    """Subset: frontal_structure — scalar gradient magnitudes, Turner angle."""
    results = {}
    requested = set(computed_feature_channels)
    mld = None

    def _ensure_mld():
        nonlocal mld
        if mld is None:
            mld = mixed_layer_depth(ds_merge)   # lazy
        return mld

    zdim_t = _get_vertical_dim(ds_merge.Theta)

    # gradtheta2 — |∇Θ|²
    if any(c.startswith("gradtheta2_") for c in requested):
        _ensure_mld()
        gt2_3d = _grad_squared_3d(ds_merge.Theta, ds_merge, grid)
        results.update(apply_depth_strategies(
            gt2_3d, "gradtheta2", ds_merge, mld=mld, requested=requested))

    # gradsalt2 — |∇S|²
    if any(c.startswith("gradsalt2_") for c in requested):
        _ensure_mld()
        gs2_3d = _grad_squared_3d(ds_merge.Salt, ds_merge, grid)
        results.update(apply_depth_strategies(
            gs2_3d, "gradsalt2", ds_merge, mld=mld, requested=requested))

    # gradb2 — |∇b|²
    if any(c.startswith("gradb2_") for c in requested):
        _ensure_mld()
        b = buoyancy_field_3d(ds_merge)
        gb2_3d = _grad_squared_3d(b, ds_merge, grid)
        results.update(apply_depth_strategies(
            gb2_3d, "gradb2", ds_merge, mld=mld, requested=requested))

    # gradrho2 — |∇ρ|²
    if any(c.startswith("gradrho2_") for c in requested):
        _ensure_mld()
        rho = _density_lazy(ds_merge)
        gr2_3d = _grad_squared_3d(rho, ds_merge, grid)
        results.update(apply_depth_strategies(
            gr2_3d, "gradrho2", ds_merge, mld=mld, requested=requested))

    # gradeta2 — |∇η|² (inherently 2D, surface only)
    if "gradeta2_sfc" in requested:
        eta = ds_merge.Eta
        gx, gy = ng.calculate_native_gradient_tracer(eta, ds_merge, grid=grid)
        results["gradeta2_sfc"] = gx**2 + gy**2

    # Turner angle
    if any(c.startswith("turner_angle_") for c in requested):
        _ensure_mld()
        ALPHA = 2.0e-4
        BETA = 7.4e-4
        RHO0_TU = 1025.0

        gt2_3d = _grad_squared_3d(ds_merge.Theta, ds_merge, grid)
        gs2_3d = _grad_squared_3d(ds_merge.Salt, ds_merge, grid)
        rho = _density_lazy(ds_merge)
        gr2_3d = _grad_squared_3d(rho, ds_merge, grid)

        numer = RHO0_TU * (BETA**2 * gs2_3d - ALPHA**2 * gt2_3d)
        denom = xr.where(gr2_3d > 0, -gr2_3d / RHO0_TU, np.nan)
        turner_3d = np.degrees(np.arctan(numer / denom))

        results.update(apply_depth_strategies(
            turner_3d, "turner_angle", ds_merge, mld=mld,
            requested=requested))

    return _materialise_results(results)


def compute_kinematic(ds_merge, grid, computed_feature_channels):
    """Subset: kinematic — vorticity, strain, divergence, Okubo-Weiss."""
    results = {}
    requested = set(computed_feature_channels)

    bases = ("relative_vorticity", "strain_n", "strain_s", "strain_mag",
             "divergence", "okubo_weiss")
    if not any(any(c.startswith(b + "_") for c in requested) for b in bases):
        return results

    mld = mixed_layer_depth(ds_merge)   # lazy
    props = _velocity_properties_3d(ds_merge, grid)   # lazy 3D dict

    for base in bases:
        if any(c.startswith(base + "_") for c in requested):
            results.update(apply_depth_strategies(
                props[base], base, ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_frontogenesis(ds_merge, grid, computed_feature_channels):
    """Subset: frontogenesis — tendency, geo/ageo decomposition."""
    results = {}
    requested = set(computed_feature_channels)

    fronto_bases = ("frontogenesis_tendency", "frontogenesis_geo",
                    "frontogenesis_ageo")
    active = any(
        any(c.startswith(b + "_") for c in requested)
        for b in fronto_bases
    )
    geo_vel = any(c.startswith("ug_") or c.startswith("vg_")
                  for c in requested)

    if not active and not geo_vel:
        return results

    mld = mixed_layer_depth(ds_merge)   # lazy
    flds = _frontogenesis_fields_3d(ds_merge, grid)   # lazy

    for base in fronto_bases:
        if any(c.startswith(base + "_") for c in requested):
            results.update(apply_depth_strategies(
                flds[base], base, ds_merge, mld=mld, requested=requested))

    # Geostrophic velocity — inherently surface
    for vname in ("ug", "vg"):
        key = f"{vname}_sfc"
        if key in requested:
            results[key] = flds[vname]

    return _materialise_results(results)


def compute_surface_wind(ds_merge, grid, computed_feature_channels):
    """Subset: surface_wind — curl, Ekman pumping, Ekman transport."""
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

    return _materialise_results(results)


# ===========================================================================
#  DISPATCH TABLE
# ===========================================================================

SUBSET_COMPUTE_FNS_DASK = {
    "stratification":    compute_stratification,
    "surface_wind":      compute_surface_wind,
    "vertical_shear":    compute_vertical_shear,
    "mixing_parameters": compute_mixing_parameters,
    "ertel_pv":          compute_ertel_pv,
    "buoyancy_fluxes":   compute_buoyancy_fluxes,
    "energetics":        compute_energetics,
    "frontal_structure": compute_frontal_structure,
    "kinematic":         compute_kinematic,
    "frontogenesis":     compute_frontogenesis,
}
