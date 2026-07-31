"""
Compute-subset entry points for the surface (2D) pipeline.

Each ``compute_*`` function takes ``(ds_merge, grid, computed_feature_channels)``
and returns a ``dict[str, DataArray|ndarray]`` of derived fields.

The dispatch table ``SUBSET_COMPUTE_FNS`` maps subset names (as they appear
in the YAML config) to the corresponding entry-point function.

These callbacks operate on 2D surface fields only (no vertical axis).  For
depth-resolved diagnostics see ``depth_subsets.py``.
"""

import dask

import dbof.preprocessing.calculate_fields as calculate_fields
from dbof.preprocessing.calculate_fields import (
    ekman_pumping,
    ekman_transport,
    wind_stress_curl,
)


# ===========================================================================
#  COMPUTE FUNCTIONS
# ===========================================================================

def _compute_no_op(ds_merge, grid, computed_feature_channels):
    """No-op callback for subsets that only output raw model variables."""
    return {}


def compute_native_fields(ds_merge, grid, computed_feature_channels):
    """Subset: native_fields — geographic velocity components.

    ``U``/``V`` are interpolated to tracer points and rotated to
    geographic components, so the output ``U`` channel is eastward and
    ``V`` northward velocity.  See the vector-handling policy in
    ``dbof.utils.faces_to_latlon`` for why.
    """
    requested = set(computed_feature_channels)
    results = {}
    if {"U", "V"} & requested:
        u_east, v_north = calculate_fields.geographic_velocity(
            ds_merge, grid)
        if "U" in requested:
            results["U"] = u_east
        if "V" in requested:
            results["V"] = v_north
    return results


def compute_surface_wind(ds_merge, grid, computed_feature_channels):
    """Subset: surface_wind — geographic wind stress, curl, Ekman pumping/transport.

    Output ``oceTAUX`` is eastward and ``oceTAUY`` northward wind stress
    (see the vector-handling policy in ``dbof.utils.faces_to_latlon``).
    """
    requested = set(computed_feature_channels)
    results = {}

    if {"oceTAUX", "oceTAUY"} & requested:
        tau_east, tau_north = calculate_fields.geographic_wind_stress(
            ds_merge, grid)
        if "oceTAUX" in requested:
            results["oceTAUX"] = tau_east
        if "oceTAUY" in requested:
            results["oceTAUY"] = tau_north

    if "wind_stress_curl" in requested:
        results["wind_stress_curl"] = wind_stress_curl(ds_merge, grid)

    if "ekman_pumping" in requested:
        results["ekman_pumping"] = ekman_pumping(ds_merge, grid)

    if {"u_ekman", "v_ekman"} & requested:
        ek = ekman_transport(ds_merge, grid)
        for ch in ("u_ekman", "v_ekman"):
            if ch in requested:
                results[ch] = ek[ch]

    return results


def compute_frontal_structure(ds_merge, grid, computed_feature_channels):
    """Subset: frontal_structure — scalar gradient magnitudes and Turner angle.

    Gradient fields that the Turner angle depends on (gradtheta2, gradsalt2,
    gradrho2) are computed first and forwarded so each gradient is evaluated
    only once.
    """
    _GRAD_FNS = {
        "gradsalt2":  calculate_fields.grad_salt2,
        "gradtheta2": calculate_fields.grad_theta2,
        "gradeta2":   calculate_fields.grad_eta2,
        "gradb2":     calculate_fields.grad_b2,
        "gradrho2":   calculate_fields.grad_rho2,
    }

    # Turner angle depends on gradtheta2, gradsalt2, and gradrho2.
    # If turner_angle is requested, ensure its dependencies are computed
    # even if they are not individually requested as output channels.
    turner_requested = "turner_angle" in computed_feature_channels
    turner_deps = {"gradtheta2", "gradsalt2", "gradrho2"}

    needed = set(computed_feature_channels) | (turner_deps if turner_requested else set())

    results = {
        name: fn(ds_merge, grid)
        for name, fn in _GRAD_FNS.items()
        if name in needed
    }

    if turner_requested:
        results["turner_angle"] = calculate_fields.turner_angle(
            ds_merge,
            grid,
            gradtheta2=results["gradtheta2"],
            gradsalt2=results["gradsalt2"],
            gradrho2=results["gradrho2"],
        )

    # Surface potential density [kg m-3] and buoyancy [m s-2] from the
    # single lazy implementations (JMD95 at p=0; b = G rho / RHO0).
    # Channel keys are unchanged.
    if "density" in needed:
        results["density"] = calculate_fields.potential_density_anomaly(
            ds_merge)
    if "buoyancy" in needed:
        results["buoyancy"] = calculate_fields.buoyancy_of_field(
            ds_merge)

    # Only return channels that were actually requested.
    return {k: v for k, v in results.items() if k in computed_feature_channels}


def compute_kinematic(ds_merge, grid, computed_feature_channels):
    """Subset: kinematic — vorticity, strain, divergence, Okubo-Weiss, etc.

    Computes the velocity Jacobian once and passes it to each individual
    property function.  Only channels listed in ``computed_feature_channels``
    are returned.
    """
    requested = set(computed_feature_channels)
    jac = calculate_fields.compute_velocity_jacobian(ds_merge, grid)

    results = {}

    if 'relative_vorticity' in requested:
        results['relative_vorticity'] = (
            calculate_fields.relative_vorticity(ds_merge, grid, jacobian=jac))

    if {'strain_n', 'strain_s', 'strain_mag'} & requested:
        strain_mag, strain_n, strain_s = (
            calculate_fields.strain(ds_merge, grid, jacobian=jac))
        if 'strain_n' in requested:
            results['strain_n'] = strain_n
        if 'strain_s' in requested:
            results['strain_s'] = strain_s
        if 'strain_mag' in requested:
            results['strain_mag'] = strain_mag

    if 'divergence' in requested:
        results['divergence'] = (
            calculate_fields.divergence(ds_merge, grid, jacobian=jac))

    if 'coriolis_f' in requested:
        results['coriolis_f'] = (
            calculate_fields.coriolis_parameter(ds_merge, grid))

    if 'rossby_number' in requested:
        results['rossby_number'] = (
            calculate_fields.rossby_number(ds_merge, grid, jacobian=jac))

    if 'okubo_weiss' in requested:
        results['okubo_weiss'] = (
            calculate_fields.okubo_weiss_parameter(ds_merge, grid, jacobian=jac))

    return results


def compute_frontogenesis(ds_merge, grid, computed_feature_channels):
    """Subset: frontogenesis — tendency, geo/ageo decomposition, ug, vg.

    Computes shared intermediates (velocity Jacobian, buoyancy gradients,
    geostrophic velocity) once and passes them to the individual property
    functions.  Only channels listed in ``computed_feature_channels`` are
    returned.

    **CRITICAL**: all selected fields are materialised via a single
    ``dask.compute()`` call.  This fuses the shared graph (Jacobian +
    tracer gradients) into one scheduler submission, avoiding dangerous
    run_spec warnings that appear when multiple frontogenesis arrays
    sharing a lazy lineage are computed separately.  Do NOT replace this
    with per-field ``.compute()`` calls.
    """
    requested = set(computed_feature_channels)

    need_tendency = ('frontogenesis_tendency' in requested
                     or 'frontogenesis_ageo' in requested)
    need_geo = ('frontogenesis_geo' in requested
                or 'frontogenesis_ageo' in requested)
    need_ugvg = 'ug' in requested or 'vg' in requested

    if not (need_tendency or need_geo or need_ugvg):
        return {}

    # -- shared intermediates --------------------------------------------------
    bg = calculate_fields.compute_buoyancy_gradients(ds_merge, grid)

    results = {}

    # Full frontogenesis tendency F(u, v)
    tendency = None
    if need_tendency:
        jac = calculate_fields.compute_velocity_jacobian(ds_merge, grid)
        tendency = calculate_fields.frontogenesis_tendency(
            ds_merge, grid, jacobian=jac, buoyancy_gradients=bg)
        if 'frontogenesis_tendency' in requested:
            results['frontogenesis_tendency'] = tendency

    # Geostrophic velocity (needed for geo frontogenesis and/or ug/vg output)
    ug = vg = None
    if need_geo or need_ugvg:
        ug, vg = calculate_fields.geostrophic_velocity(ds_merge, grid)
        if 'ug' in requested:
            results['ug'] = ug
        if 'vg' in requested:
            results['vg'] = vg

    # Geostrophic frontogenesis tendency F(ug, vg)
    geo = None
    if need_geo:
        geo = calculate_fields.frontogenesis_geo(
            ds_merge, grid, ug=ug, vg=vg, buoyancy_gradients=bg)
        if 'frontogenesis_geo' in requested:
            results['frontogenesis_geo'] = geo

    # Ageostrophic frontogenesis — residual of full minus geostrophic.
    if 'frontogenesis_ageo' in requested:
        results['frontogenesis_ageo'] = tendency - geo

    if not results:
        return results

    # CRITICAL: single dask.compute() materialises all fields together.
    keys = list(results.keys())
    materialised = dask.compute(*[results[k] for k in keys])
    return dict(zip(keys, materialised))


# ===========================================================================
#  DISPATCH TABLE
# ===========================================================================

SUBSET_COMPUTE_FNS = {
    "native_fields":     compute_native_fields,
    "surface_wind":      compute_surface_wind,
    "icearea":           _compute_no_op,          # raw model vars only
    "frontal_structure": compute_frontal_structure,
    "kinematic":         compute_kinematic,
    "frontogenesis":     compute_frontogenesis,
}
