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
import dask

from dbof.preprocessing.depth_strategies import (
    apply_depth_strategies,
)
from dbof.preprocessing.calculated_fields_at_depth import (
    # -- density / core 3D --
    mixed_layer_depth,
    buoyancy_frequency_squared_3d,
    buoyancy_field_3d,
    mixed_layer_heat_content,
    # -- shear & dimensionless --
    vertical_shear_magnitude_3d,
    richardson_number_3d,
    froude_number_3d,
    rossby_number_3d,
    burger_number_3d,
    # -- wind --
    wind_stress_curl,
    ekman_pumping,
    ekman_transport,
    # -- fluxes --
    advective_buoyancy_fluxes_3d,
    # -- PV --
    ertel_pv_terms_3d,
    # -- gradient / frontal structure --
    _grad_squared_3d,
    grad_theta2_3d,
    grad_salt2_3d,
    grad_rho2_3d,
    grad_b2_3d,
    grad_eta2,
    turner_angle_3d,
    # -- kinematic (individual 3D functions) --
    compute_velocity_jacobian_3d,
    relative_vorticity_3d,
    strain_3d,
    divergence_3d,
    okubo_weiss_3d,
    # -- frontogenesis (individual 3D functions) --
    compute_buoyancy_gradients_3d,
    frontogenesis_tendency_3d,
    geostrophic_velocity_3d,
    frontogenesis_geo_3d,
)
from dbof.preprocessing.calculate_additional_fields import coriolis_parameter
from dbof.preprocessing.vertical_helpers import _interp_w_to_tracer_levels


# ---------------------------------------------------------------------------
#  Materialise helper — single .compute() for all results in a subset
# ---------------------------------------------------------------------------

def _materialise_results(results):
    """dask.compute() every lazy value in *results*, return numpy-backed dict.

    Parameters
    ----------
    results : dict[str, xr.DataArray]
        Mapping of channel name to lazy dask-backed DataArray.

    Returns
    -------
    dict[str, np.ndarray]
        Same keys as *results*, values materialised as NumPy arrays.
    """
    if not results:
        return results
    keys = list(results.keys())
    lazy_vals = [results[k] for k in keys]
    materialised = dask.compute(*lazy_vals, retries=10)
    return dict(zip(keys, materialised))


def _needs_mld(requested):
    """Return True if any requested channel uses a MLD-based depth strategy.

    Parameters
    ----------
    requested : iterable of str
        Channel names requested by the caller.

    Returns
    -------
    bool
        ``True`` if at least one channel ends with ``_mld`` or
        ``_mld_mean``, or is ``"mixed_layer_depth"`` / ``"ml_heat_content"``.
    """
    return any(c.endswith("_mld") or c.endswith("_mld_mean")
               or c == "mixed_layer_depth" or c == "ml_heat_content"
               for c in requested)


def _lazy_mld(ds_merge, requested):
    """Return lazy MLD if any requested channel needs it, else None.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates.
    requested : iterable of str
        Channel names requested by the caller.

    Returns
    -------
    xr.DataArray or None
        Lazy mixed-layer depth (m, positive downward), or ``None`` if no
        MLD-based channel was requested.
    """
    if _needs_mld(requested):
        return mixed_layer_depth(ds_merge)
    return None


# ===========================================================================
#  COMPUTE FUNCTIONS
# ===========================================================================

def compute_stratification(ds_merge, grid, computed_feature_channels):
    """Subset: stratification — MLD, N², mixed-layer heat content.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    mld = _lazy_mld(ds_merge, requested)
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
    """Subset: vertical_shear — |S| and Ri at four depths.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    shear_requested = any(c.startswith("vertical_shear_") for c in requested)
    ri_requested = any(c.startswith("Ri_") for c in requested)
    if not shear_requested and not ri_requested:
        return results

    mld = _lazy_mld(ds_merge, requested)

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
    """Subset: mixing_parameters — Fr, Ro, Burger at four depths.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    mld = _lazy_mld(ds_merge, requested)

    if any(c.startswith("Fr_") for c in requested):
        fr_3d = froude_number_3d(ds_merge, grid, mld=mld)   # lazy
        results.update(apply_depth_strategies(
            fr_3d, "Fr", ds_merge, mld=mld, requested=requested))

    if any(c.startswith("Ro_") for c in requested):
        ro_3d = rossby_number_3d(ds_merge, grid)   # lazy
        results.update(apply_depth_strategies(
            ro_3d, "Ro", ds_merge, mld=mld, requested=requested))

    if any(c.startswith("Bu_") for c in requested):
        bu_3d = burger_number_3d(ds_merge, grid, mld=mld)   # lazy
        results.update(apply_depth_strategies(
            bu_3d, "Bu", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_ertel_pv(ds_merge, grid, computed_feature_channels):
    """Subset: ertel_pv — Ertel PV terms at four depths.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    pv_bases = ("ertel_pv", "ertel_pv_vertical", "ertel_pv_tilt")
    active_bases = [b for b in pv_bases
                    if any(c.startswith(b + "_") for c in requested)]
    if not active_bases:
        return results

    mld = _lazy_mld(ds_merge, requested)
    pv_dict = ertel_pv_terms_3d(ds_merge, grid)   # lazy 3D fields

    for base in active_bases:
        results.update(apply_depth_strategies(
            pv_dict[base], base, ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_buoyancy_fluxes(ds_merge, grid, computed_feature_channels):
    """Subset: buoyancy_fluxes — uB, vB, wB at four depths.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    flux_bases = ("uB", "vB", "wB")
    if not any(ch.startswith(b) for ch in requested for b in flux_bases):
        return results

    mld = _lazy_mld(ds_merge, requested)
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
    """Subset: energetics — KE = 0.5 * (MLD * |∇b| / f)².

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    if not any(c.startswith("KE_") for c in requested):
        return results

    mld = _lazy_mld(ds_merge, requested)
    b = buoyancy_field_3d(ds_merge)   # lazy
    f = coriolis_parameter(ds_merge, grid)

    grad_b2 = _grad_squared_3d(b, ds_merge, grid)   # lazy
    grad_b_mag = np.sqrt(grad_b2)
    ke_3d = 0.5 * (mld * grad_b_mag / f)**2   # lazy

    results.update(apply_depth_strategies(
        ke_3d, "KE", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_frontal_structure(ds_merge, grid, computed_feature_channels):
    """Subset: frontal_structure — scalar gradient magnitudes, Turner angle.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)
    mld = None

    def _ensure_mld():
        nonlocal mld
        if mld is None and _needs_mld(requested):
            mld = mixed_layer_depth(ds_merge)   # lazy
        return mld

    # Turner angle depends on gradtheta2, gradsalt2, gradrho2.
    # Compute their 3D fields if either the gradient itself or
    # turner_angle is requested, so each gradient is built at most once.
    turner_requested = any(c.startswith("turner_angle_") for c in requested)
    turner_deps = {"gradtheta2", "gradsalt2", "gradrho2"}
    needed_bases = {
        base for base in turner_deps
        if any(c.startswith(base + "_") for c in requested)
    }
    if turner_requested:
        needed_bases |= turner_deps

    # Cache of 3D gradient fields: base_name → lazy xr.DataArray
    _GRAD_FNS = {
        "gradtheta2": grad_theta2_3d,
        "gradsalt2":  grad_salt2_3d,
        "gradrho2":   grad_rho2_3d,
    }
    grad_3d_cache = {}
    for base in needed_bases:
        _ensure_mld()
        grad_3d_cache[base] = _GRAD_FNS[base](ds_merge, grid)

    # Apply depth strategies for each requested gradient channel.
    for base in ("gradtheta2", "gradsalt2", "gradrho2"):
        if any(c.startswith(base + "_") for c in requested) and base in grad_3d_cache:
            results.update(apply_depth_strategies(
                grad_3d_cache[base], base, ds_merge, mld=mld,
                requested=requested))

    # gradb2 — |∇b|² (independent of Turner angle)
    if any(c.startswith("gradb2_") for c in requested):
        _ensure_mld()
        results.update(apply_depth_strategies(
            grad_b2_3d(ds_merge, grid),
            "gradb2", ds_merge, mld=mld, requested=requested))

    # gradeta2 — |∇η|² (inherently 2D, surface only)
    if "gradeta2_sfc" in requested:
        results["gradeta2_sfc"] = grad_eta2(ds_merge, grid)

    # Turner angle — reuses cached 3D gradient fields.
    if turner_requested:
        _ensure_mld()
        results.update(apply_depth_strategies(
            turner_angle_3d(
                ds_merge, grid,
                gradtheta2=grad_3d_cache["gradtheta2"],
                gradsalt2=grad_3d_cache["gradsalt2"],
                gradrho2=grad_3d_cache["gradrho2"],
            ),
            "turner_angle", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


def compute_kinematic(ds_merge, grid, computed_feature_channels):
    """Subset: kinematic — vorticity, strain, divergence, Okubo-Weiss.

    Computes the 3D velocity Jacobian once and passes it to each
    individual kinematic function.  Only channels listed in
    ``computed_feature_channels`` are returned.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    bases = ("relative_vorticity", "strain_n", "strain_s", "strain_mag",
             "divergence", "rossby_number", "okubo_weiss")
    has_depth_channels = any(
        any(c.startswith(b + "_") for c in requested) for b in bases)
    has_coriolis = "coriolis_f" in requested
    if not has_depth_channels and not has_coriolis:
        return results

    mld = _lazy_mld(ds_merge, requested)
    jac = compute_velocity_jacobian_3d(ds_merge, grid)   # lazy — shared

    if any(c.startswith("relative_vorticity_") for c in requested):
        results.update(apply_depth_strategies(
            relative_vorticity_3d(ds_merge, grid, jacobian=jac),
            "relative_vorticity", ds_merge, mld=mld, requested=requested))

    if any(c.startswith(b + "_") for c in requested
           for b in ("strain_n", "strain_s", "strain_mag")):
        sm, sn, ss = strain_3d(ds_merge, grid, jacobian=jac)
        for base, field in [("strain_n", sn), ("strain_s", ss),
                            ("strain_mag", sm)]:
            if any(c.startswith(base + "_") for c in requested):
                results.update(apply_depth_strategies(
                    field, base, ds_merge, mld=mld, requested=requested))

    if any(c.startswith("divergence_") for c in requested):
        results.update(apply_depth_strategies(
            divergence_3d(ds_merge, grid, jacobian=jac),
            "divergence", ds_merge, mld=mld, requested=requested))

    if any(c.startswith("rossby_number_") for c in requested):
        results.update(apply_depth_strategies(
            rossby_number_3d(ds_merge, grid),
            "rossby_number", ds_merge, mld=mld, requested=requested))

    if any(c.startswith("okubo_weiss_") for c in requested):
        results.update(apply_depth_strategies(
            okubo_weiss_3d(ds_merge, grid, jacobian=jac),
            "okubo_weiss", ds_merge, mld=mld, requested=requested))

    # coriolis_f has no depth dependence — no depth strategies needed.
    # Keep it as a DataArray (dims (face, j, i) on the native grid) so
    # _materialise_results preserves dimension names and coordinates.
    if "coriolis_f" in requested:
        results["coriolis_f"] = coriolis_parameter(ds_merge, grid)

    return _materialise_results(results)


def compute_frontogenesis(ds_merge, grid, computed_feature_channels):
    """Subset: frontogenesis — tendency, geo/ageo decomposition, ug, vg.

    Computes shared intermediates (velocity Jacobian, 3D buoyancy
    gradients, geostrophic velocity) once and passes them to individual
    property functions.  Only channels listed in
    ``computed_feature_channels`` are returned.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)

    fronto_bases = ("frontogenesis_tendency", "frontogenesis_geo",
                    "frontogenesis_ageo")

    need_tendency = any(
        any(c.startswith(b + "_") for c in requested)
        for b in ("frontogenesis_tendency", "frontogenesis_ageo")
    )
    need_geo = any(
        any(c.startswith(b + "_") for c in requested)
        for b in ("frontogenesis_geo", "frontogenesis_ageo")
    )
    need_ugvg = any(c.startswith("ug_") or c.startswith("vg_")
                    for c in requested)

    if not (need_tendency or need_geo or need_ugvg):
        return results

    mld = _lazy_mld(ds_merge, requested)

    # -- shared intermediates --
    bg = compute_buoyancy_gradients_3d(ds_merge, grid)

    # Full frontogenesis tendency F(u, v)
    tendency_3d = None
    if need_tendency:
        jac = compute_velocity_jacobian_3d(ds_merge, grid)
        tendency_3d = frontogenesis_tendency_3d(
            ds_merge, grid, jacobian=jac, buoyancy_gradients=bg)
        if any(c.startswith("frontogenesis_tendency_") for c in requested):
            results.update(apply_depth_strategies(
                tendency_3d, "frontogenesis_tendency", ds_merge,
                mld=mld, requested=requested))

    # Geostrophic velocity (needed for geo frontogenesis and/or ug/vg output)
    ug = vg = None
    if need_geo or need_ugvg:
        ug, vg = geostrophic_velocity_3d(ds_merge, grid)
        # Geostrophic velocity — inherently surface
        for vname, field in [("ug", ug), ("vg", vg)]:
            key = f"{vname}_sfc"
            if key in requested:
                results[key] = field

    # Geostrophic frontogenesis tendency F(ug, vg)
    geo_3d = None
    if need_geo:
        geo_3d = frontogenesis_geo_3d(
            ds_merge, grid, ug=ug, vg=vg, buoyancy_gradients=bg)
        if any(c.startswith("frontogenesis_geo_") for c in requested):
            results.update(apply_depth_strategies(
                geo_3d, "frontogenesis_geo", ds_merge,
                mld=mld, requested=requested))

    # Ageostrophic frontogenesis — residual of full minus geostrophic.
    if any(c.startswith("frontogenesis_ageo_") for c in requested):
        ageo_3d = tendency_3d - geo_3d
        results.update(apply_depth_strategies(
            ageo_3d, "frontogenesis_ageo", ds_merge,
            mld=mld, requested=requested))

    return _materialise_results(results)


def compute_surface_wind(ds_merge, grid, computed_feature_channels):
    """Subset: surface_wind — curl, Ekman pumping, Ekman transport.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}

    if "wind_stress_curl" in computed_feature_channels:
        results["wind_stress_curl"] = wind_stress_curl(ds_merge, grid)

    if "ekman_pumping" in computed_feature_channels:
        results["ekman_pumping"] = ekman_pumping(ds_merge, grid)

    ekman_channels = {"u_ekman", "v_ekman"}
    if ekman_channels.intersection(computed_feature_channels):
        ek = ekman_transport(ds_merge, grid)
        for ch in ekman_channels:
            if ch in computed_feature_channels:
                results[ch] = ek[ch]

    return _materialise_results(results)


# ===========================================================================
#  DISPATCH TABLE
# ===========================================================================

def _compute_no_op(ds_merge, grid, computed_feature_channels):
    """No-op callback for subsets that only output raw model variables.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset (unused).
    grid : xgcm.Grid
        Grid object (unused).
    computed_feature_channels : list of str
        Channel names (unused).

    Returns
    -------
    dict[str, np.ndarray]
        Always an empty dict.
    """
    return {}


def compute_native_fields(ds_merge, grid, computed_feature_channels):
    """Subset: native_fields — raw tracers & velocities at depth.

    Feeds the raw 3D model variables through ``apply_depth_strategies``
    so they can be extracted at _sfc, _z25m, _mld, and _mld_mean.

    U and V live on staggered grids (i_g / j_g) and are interpolated to
    tracer points before the depth reduction.  Eta is inherently 2D
    (surface only) and is handled as a special case.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset with grid coordinates and model variables
        (Theta, Salt, U, V, W, Eta).
    grid : xgcm.Grid
        Grid object used for staggered-to-tracer-point interpolation.
    computed_feature_channels : list of str
        Channel names to compute (as specified in the YAML config).

    Returns
    -------
    dict[str, np.ndarray]
        Materialised arrays keyed by channel name.
    """
    results = {}
    requested = set(computed_feature_channels)
    if not requested:
        return results

    # Lazy MLD — only computed when a _mld or _mld_mean channel is requested.
    mld = None
    need_mld = any(c.endswith("_mld") or c.endswith("_mld_mean")
                   for c in requested)

    def _ensure_mld():
        nonlocal mld
        if mld is None and need_mld:
            mld = mixed_layer_depth(ds_merge)
        return mld

    # -- Scalar tracers (already on tracer grid) --
    for base in ("Theta", "Salt"):
        if any(c.startswith(base + "_") for c in requested):
            _ensure_mld()
            results.update(apply_depth_strategies(
                ds_merge[base], base, ds_merge, mld=mld,
                requested=requested))

    # -- Eta (inherently 2D — no vertical dimension) --
    if "Eta_sfc" in requested:
        results["Eta_sfc"] = ds_merge["Eta"]

    # -- Velocity: interpolate staggered → tracer points, then apply depths --
    if any(c.startswith("U_") for c in requested):
        _ensure_mld()
        U_c = grid.interp(ds_merge.U, 'X', boundary='fill')
        results.update(apply_depth_strategies(
            U_c, "U", ds_merge, mld=mld, requested=requested))

    if any(c.startswith("V_") for c in requested):
        _ensure_mld()
        V_c = grid.interp(ds_merge.V, 'Y', boundary='fill')
        results.update(apply_depth_strategies(
            V_c, "V", ds_merge, mld=mld, requested=requested))

    # -- W (on the vertical face grid) -> interpolate to tracer centres
    #    (k / Z) so the depth strategies align with the tracer-centred MLD.
    if any(c.startswith("W_") for c in requested):
        _ensure_mld()
        W_c = _interp_w_to_tracer_levels(ds_merge)
        results.update(apply_depth_strategies(
            W_c, "W", ds_merge, mld=mld, requested=requested))

    return _materialise_results(results)


SUBSET_COMPUTE_FNS = {
    # Surface-only subsets (no depth computation — raw model fields only).
    "icearea":           _compute_no_op,
    # Depth-resolved diagnostic subsets.
    "native_fields":     compute_native_fields,
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
