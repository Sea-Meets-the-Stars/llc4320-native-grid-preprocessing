"""
Canonical subset definitions for all global pipelines.

Variable lists, dataset names, and surface_only flags live here in code --
users cannot modify them.  The only user-configurable knob is
``depth_suffixes`` (overridable in the YAML).

Surface subsets (SURF / OSN pipelines)
--------------------------------------
Operate on 2D surface fields.  No ``depth_suffixes`` or ``extra_channels``.

Depth subsets (DEPTH pipeline)
------------------------------
Operate on 3D fields, reduced to 2D via depth strategies (sfc, z25m, mld,
mld_mean).  Base channel names in ``compute_features_channels`` are expanded
with the active ``depth_suffixes`` at runtime.

Usage
-----
::

    defn = get_subset_definition("DEPTH", "stratification")
    compute_fn = get_compute_fn("DEPTH", "stratification")
"""

from dbof.preprocessing import surface_subsets, depth_subsets


# ===========================================================================
#  SURFACE SUBSETS  (shared by SURF and OSN pipelines)
# ===========================================================================

SURFACE_SUBSETS = {

    "native_fields": {
        "dataset_name": "native_fields.zarr",
        "surface_only": True,
        "model_data_feature_channels": [
            "Theta", "Salt", "Eta", "W",
        ],
        "compute_features_channels": ["U", "V"],
    },

    # oceQnet exists only in the LLC_SURF S3 stores, not the OSN kerchunk
    # endpoints — added via pipeline_model_channels for SURF only.
    "surface_wind": {
        "dataset_name": "surface_wind.zarr",
        "surface_only": True,
        "model_data_feature_channels": [],
        "pipeline_model_channels": {"SURF": ["oceQnet"]},
        "compute_features_channels": [
            "oceTAUX", "oceTAUY",
            "wind_stress_curl", "ekman_pumping", "u_ekman", "v_ekman",
        ],
    },

    "icearea": {
        "dataset_name": "icearea.zarr",
        "surface_only": True,
        "model_data_feature_channels": ["SIarea"],
        "compute_features_channels": [],
    },

    "frontal_structure": {
        "dataset_name": "frontal_structure.zarr",
        "surface_only": True,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "gradb2", "gradsalt2", "gradtheta2",
            "gradeta2", "gradrho2", "turner_angle",
            "density", "buoyancy",
        ],
    },

    "kinematic": {
        "dataset_name": "kinematic.zarr",
        "surface_only": True,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "relative_vorticity", "strain_n", "strain_s", "strain_mag",
            "divergence", "coriolis_f", "rossby_number", "okubo_weiss",
        ],
    },

    "frontogenesis": {
        "dataset_name": "frontogenesis.zarr",
        "surface_only": True,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "frontogenesis_tendency", "ug", "vg",
            "frontogenesis_geo", "frontogenesis_ageo",
        ],
    },
}


# ===========================================================================
#  DEPTH SUBSETS  (DEPTH pipeline)
# ===========================================================================

#: Default depth suffixes applied when the YAML does not override.
DEFAULT_DEPTH_SUFFIXES = ["sfc", "z25m", "mld", "mld_mean"]

#: Surface-only (2D) bases: only ever emit ``_sfc``, never depth suffixes.
SURFACE_ONLY_BASES = frozenset({"Eta", "gradeta2", "ug", "vg"})

DEPTH_SUBSETS = {

    # -- Depth-resolved diagnostic subsets -----------------------------------

    "stratification": {
        "dataset_name": "stratification.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": ["N2"],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
        "extra_channels": ["mixed_layer_depth", "ml_heat_content"],
    },

    "vertical_shear": {
        "dataset_name": "vertical_shear.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": ["vertical_shear", "Ri"],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "mixing_parameters": {
        "dataset_name": "mixing_parameters.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": ["Fr", "Ro", "Bu", "R_ib"],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "ertel_pv": {
        "dataset_name": "ertel_pv.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": ["ertel_pv", "ertel_pv_vertical", "ertel_pv_tilt"],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "buoyancy_fluxes": {
        "dataset_name": "buoyancy_fluxes.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": ["uB", "vB", "wB"],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "surface_wind": {
        "dataset_name": "surface_wind.zarr",
        "surface_only": True,
        "model_data_feature_channels": ["oceQnet"],
        "compute_features_channels": [
            "oceTAUX", "oceTAUY",
            "wind_stress_curl", "ekman_pumping", "u_ekman", "v_ekman",
        ],
    },

    "energetics": {
        "dataset_name": "energetics.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": ["KE"],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "frontal_structure": {
        "dataset_name": "frontal_structure.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "gradb2", "gradtheta2", "gradsalt2",
            "gradrho2", "gradeta2", "turner_angle",
        ],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "kinematic": {
        "dataset_name": "kinematic.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "relative_vorticity", "strain_n", "strain_s", "strain_mag",
            "divergence", "rossby_number", "okubo_weiss",
        ],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
        "extra_channels": ["coriolis_f"],
    },

    "frontogenesis": {
        "dataset_name": "frontogenesis.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "frontogenesis_tendency", "frontogenesis_geo",
            "frontogenesis_ageo", "ug", "vg", "Wstar",
        ],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "native_fields": {
        "dataset_name": "native_fields.zarr",
        "surface_only": False,
        "model_data_feature_channels": [],
        "compute_features_channels": [
            "Theta", "Salt", "Eta", "U", "V", "W",
        ],
        "depth_suffixes": DEFAULT_DEPTH_SUFFIXES,
    },

    "icearea": {
        "dataset_name": "icearea.zarr",
        "surface_only": True,
        "model_data_feature_channels": ["SIarea"],
        "compute_features_channels": [],
    },
}


# ===========================================================================
#  Compute-function dispatch tables (re-exported from preprocessing modules)
# ===========================================================================

_SURFACE_COMPUTE_FNS = surface_subsets.SUBSET_COMPUTE_FNS
_DEPTH_COMPUTE_FNS = depth_subsets.SUBSET_COMPUTE_FNS


# ===========================================================================
#  Public API
# ===========================================================================

def get_subset_definition(pipeline: str, subset_name: str) -> dict:
    """
    Return the canonical definition dict for *subset_name* under *pipeline*.

    Parameters
    ----------
    pipeline : str
        ``"SURF"``, ``"OSN"``, or ``"DEPTH"``.
    subset_name : str
        Subset name (e.g. ``"kinematic"``, ``"stratification"``).

    Returns
    -------
    dict
        Shallow copy of the definition.  Keys: ``dataset_name``,
        ``surface_only``, ``model_data_feature_channels``,
        ``compute_features_channels``, and optionally ``depth_suffixes``,
        ``extra_channels``.

    Raises
    ------
    ValueError
        If *subset_name* is not valid for *pipeline*.
    """
    if pipeline in ("SURF", "OSN"):
        table = SURFACE_SUBSETS
    elif pipeline == "DEPTH":
        table = DEPTH_SUBSETS
    else:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'.  Expected SURF, OSN, or DEPTH."
        )

    if subset_name not in table:
        raise ValueError(
            f"Subset '{subset_name}' is not defined for pipeline '{pipeline}'.  "
            f"Valid subsets: {list(table)}"
        )

    defn = dict(table[subset_name])
    # Optional per-pipeline extra model channels (e.g. oceQnet is available
    # to SURF but not OSN).
    extra = defn.pop("pipeline_model_channels", None)
    if extra and pipeline in extra:
        defn["model_data_feature_channels"] = (
            list(defn["model_data_feature_channels"]) + list(extra[pipeline]))
    return defn


def get_compute_fn(pipeline: str, subset_name: str):
    """
    Return the compute callback for *subset_name* under *pipeline*.

    The returned callable has signature
    ``(ds_merge, grid, computed_feature_channels) -> dict``.

    Raises
    ------
    ValueError
        If *subset_name* has no registered compute function for *pipeline*.
    """
    if pipeline in ("SURF", "OSN"):
        table = _SURFACE_COMPUTE_FNS
    elif pipeline == "DEPTH":
        table = _DEPTH_COMPUTE_FNS
    else:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'.  Expected SURF, OSN, or DEPTH."
        )

    if subset_name not in table:
        raise ValueError(
            f"No compute function registered for subset '{subset_name}' "
            f"in pipeline '{pipeline}'.  Available: {list(table)}"
        )

    return table[subset_name]


def valid_subsets(pipeline: str) -> list[str]:
    """Return the list of valid subset names for *pipeline*."""
    if pipeline in ("SURF", "OSN"):
        return list(SURFACE_SUBSETS)
    elif pipeline == "DEPTH":
        return list(DEPTH_SUBSETS)
    raise ValueError(
        f"Unknown pipeline '{pipeline}'.  Expected SURF, OSN, or DEPTH."
    )


def expand_channels_with_suffixes(
    channels: list[str],
    depth_suffixes: list[str] | None = None,
    extra_channels: list[str] | None = None,
) -> list[str]:
    """
    Expand channel base names with depth suffixes.

    If *depth_suffixes* is provided, each entry in *channels* is expanded
    to ``{base}_{suffix}`` for every suffix in the list.  Entries in
    *extra_channels* are appended unchanged (use for standalone diagnostics
    like ``mixed_layer_depth`` that have no depth variants).

    If *depth_suffixes* is ``None`` or empty, *channels* is returned as-is
    with any *extra_channels* appended.

    Surface-only bases (:data:`SURFACE_ONLY_BASES`, e.g. ``Eta``) are never
    expanded across the depth suffixes: they only ever get the ``_sfc``
    channel, matching what the depth compute functions actually produce.

    Parameters
    ----------
    channels : list[str]
        Base channel names (e.g. ``['N2', 'KE']``).
    depth_suffixes : list[str] or None
        Suffixes to append (e.g. ``['sfc', 'z25m', 'mld', 'mld_mean']``).
    extra_channels : list[str] or None
        Additional channels passed through without expansion.

    Returns
    -------
    list[str]
        Expanded channel list.
    """
    if not depth_suffixes:
        result = list(channels or [])
        if extra_channels:
            result.extend(extra_channels)
        return result

    result = []
    for base in (channels or []):
        if base in SURFACE_ONLY_BASES:
            # Inherently 2D field -- only the surface channel is produced.
            result.append(f"{base}_sfc")
        else:
            for suffix in depth_suffixes:
                result.append(f"{base}_{suffix}")
    if extra_channels:
        result.extend(extra_channels)
    return result
