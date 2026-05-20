"""
Compute-subset entry points for the depth pipeline.

Each ``compute_*`` function:

1. Builds lazy 3D fields via the functions in ``calculated_fields_at_depth``.
2. Applies ``apply_depth_strategies`` to reduce each 3D field to 2D at the
   requested depth channels (surface, z25m, mld, mld_mean).
3. Calls ``dask.compute()`` once to materialise all results for the subset.

The dispatch table ``SUBSET_COMPUTE_FNS`` maps subset names (as they appear
in the YAML config) to the corresponding entry-point function.
"""

import numpy as np
import xarray as xr
import dask

from dbof.preprocessing.vertical_helpers import (
    _get_vertical_dim,
)
from dbof.preprocessing.depth_strategies import (
    apply_depth_strategies,
)
from dbof.preprocessing.calculated_fields_at_depth import (
    # -- density / core 3D --
    _density_lazy,
    mixed_layer_depth,
    buoyancy_frequency_squared_3d,
    buoyancy_field_3d,
    mixed_layer_heat_content,
    # -- shear & dimensionless --
    vertical_shear_magnitude_3d,
    richardson_number_3d,
    froude_number_3d,
    rossby_number_3d,
    burger_number_2d,
    # -- wind --
    wind_stress_curl,
    ekman_pumping,
    ekman_transport_velocity,
    # -- fluxes --
    advective_buoyancy_fluxes_3d,
    # -- PV --
    ertel_pv_terms_3d,
    # -- gradient / kinematic / fronto --
    _grad_squared_3d,
    _velocity_properties_3d,
    _frontogenesis_fields_3d,
    # -- constants --
    RHO0,
)
from dbof.preprocessing.calculate_additional_fields import coriolis_parameter
import dbof.utils.native_gradient as ng


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
#  COMPUTE FUNCTIONS
# ===========================================================================

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

def compute_native_fields(ds_merge, grid, computed_feature_channels):
    """No-op callback for subsets that only output raw model variables."""
    return {}


SUBSET_COMPUTE_FNS = {
    # Surface-only subsets (no depth computation — raw model fields only).
    "native_fields":     compute_native_fields,
    "native_surface":    compute_native_fields,
    "eta":               compute_native_fields,
    "icearea":           compute_native_fields,
    "windstress":        compute_native_fields,
    # Depth-resolved diagnostic subsets.
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
