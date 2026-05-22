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

import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields


# ===========================================================================
#  COMPUTE FUNCTIONS
# ===========================================================================

def compute_native_fields(ds_merge, grid, computed_feature_channels):
    """No-op callback for subsets that only output raw model variables."""
    return {}


def compute_frontal_structure(ds_merge, grid, computed_feature_channels):
    """Subset: frontal_structure — scalar gradient magnitudes and Turner angle.

    Gradient fields that the Turner angle depends on (gradtheta2, gradsalt2,
    gradrho2) are computed first and forwarded so each gradient is evaluated
    only once.
    """
    _GRAD_FNS = {
        "gradsalt2":  calculate_additional_fields.grad_salt2,
        "gradtheta2": calculate_additional_fields.grad_theta2,
        "gradeta2":   calculate_additional_fields.grad_eta2,
        "gradb2":     calculate_additional_fields.grad_b2,
        "gradrho2":   calculate_additional_fields.grad_rho2,
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
        results["turner_angle"] = calculate_additional_fields.turner_angle(
            ds_merge,
            grid,
            gradtheta2=results["gradtheta2"],
            gradsalt2=results["gradsalt2"],
            gradrho2=results["gradrho2"],
        )

    # Only return channels that were actually requested.
    return {k: v for k, v in results.items() if k in computed_feature_channels}


def compute_kinematic(ds_merge, grid, computed_feature_channels):
    """Subset: kinematic — vorticity, strain, divergence, Okubo-Weiss, etc.

    All velocity-derived properties are obtained in a single Jacobian pass;
    only channels listed in ``computed_feature_channels`` are returned.
    """
    velocity_props = calculate_additional_fields.all_velocity_properties(ds_merge, grid)
    return {
        name: field
        for name, field in velocity_props.items()
        if name in computed_feature_channels
    }


def compute_frontogenesis(ds_merge, grid, computed_feature_channels):
    """Subset: frontogenesis — tendency, geo/ageo decomposition, ug, vg.

    Single ``dask.compute()`` call fuses the shared graph (Jacobian + tracer
    gradients) into one scheduler submission, avoiding run_spec warnings that
    appear when multiple frontogenesis arrays share a lazy lineage.
    """
    props = calculate_additional_fields.all_frontogenesis_properties(ds_merge, grid)
    selected = {
        name: field
        for name, field in props.items()
        if name in computed_feature_channels
    }

    if not selected:
        return selected

    keys = list(selected.keys())
    materialised = dask.compute(*[selected[k] for k in keys])
    return dict(zip(keys, materialised))


# ===========================================================================
#  DISPATCH TABLE
# ===========================================================================

SUBSET_COMPUTE_FNS = {
    "native_fields":     compute_native_fields,
    "frontal_structure": compute_frontal_structure,
    "kinematic":         compute_kinematic,
    "frontogenesis":     compute_frontogenesis,
}
