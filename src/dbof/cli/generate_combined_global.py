"""
Combined global LLC4320 dataset generation.

This script unifies all global property generation into a single entry point.
The subset of properties to compute is selected at runtime via the ``--subset``
flag (or the ``active_subset`` key in the YAML config).

Available subsets
-----------------
native_fields
    Raw model state variables only (Theta, Salt, Eta, U, V, W) plus the
    always-automatic ``gradb2``.  No derived quantities are computed.

frontal_structure
    Scalar gradient magnitudes that characterise front intensity and water-mass
    structure: ``gradsalt2``, ``gradtheta2``, ``gradeta2``

kinematic
    Velocity-derived scalar fields computed from a single Jacobian pass:
    ``relative_vorticity``, ``strain_n``, ``strain_s``, ``strain_mag``,
    ``divergence``, ``coriolis_f``, ``rossby_number``, ``okubo_weiss``,

frontogenesis
    Kinematic frontogenesis tendency and geostrophic components
    ``frontogenesis_tendency``,``ug``, ``vg``, ``frontogenesis_geo``,
    ``frontogenesis_ageo``.

    *** Dask graph note ***
    This subset internally calls ``all_frontogenesis_properties``, which
    merges two large lazy lineages (velocity Jacobian gradients + tracer
    gradients for buoyancy and Eta).  To avoid the large-graph and
    run_spec scheduler warnings that appear when multiple frontogenesis
    arrays share the same lineage and are written together as lazy arrays,
    this callback materialises all selected fields with a *single*
    ``dask.compute()`` call before returning them.  This fuses the shared
    subgraph in one scheduler round and returns NumPy arrays, so downstream
    zarr writes are decoupled from the Dask graph entirely.

CLI usage
---------
    generate-global-combined-dataset \\
        --config configs/combined_global.yaml \\
        --subset kinematic \\
        [--run_id my_run]

Config design
-------------
The combined YAML adds two top-level keys that are ignored by ``load_config``
but consumed by this script before the config object is constructed:

    active_subset: kinematic       # default; overridden by --subset

    subsets:
      native_fields:
        dataset_name: "native_fields.zarr"
        model_data_feature_channels: [Theta, Salt, Eta, U, V, W]
        compute_features_channels: []
      frontal_structure:
        ...
      kinematic:
        ...
      frontogenesis:
        ...

Date format
-----------
All ``date_iterations`` entries in the YAML must use ISO format:
    'YYYY-MM-DD HH:MM:SS'  e.g. '2012-09-11 12:00:00'
"""

import argparse

import dask
import yaml

import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
import dbof.dataset_creation.config as config
from dbof.cli._generate_global_base import run_global_pipeline


# ---------------------------------------------------------------------------
# Per-subset compute callbacks
# ---------------------------------------------------------------------------

def _compute_native_fields(ds_merge, grid, computed_feature_channels: list) -> dict:
    """
    Compute callback for the ``native_fields`` subset.

    No derived quantities are computed here — all requested channels are raw
    model state variables specified in ``model_data_feature_channels`` in the
    config.  

    Returns
    -------
    dict
        Always empty; present for interface consistency.
    """
    return {}


def _compute_frontal_structure_fields(
    ds_merge, grid, computed_feature_channels: list
) -> dict:
    """
    Compute callback for the ``frontal_structure`` subset.

    Computes scalar gradient-magnitude fields that characterise front intensity
    and water-mass structure. 

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    computed_feature_channels : list of str

    Returns
    -------
    dict
        Mapping of ``{channel_name: DataArray}`` for each requested channel.
    """
    _FIELD_FNS = {
        "gradsalt2":   calculate_additional_fields.grad_salt2,
        "gradtheta2":  calculate_additional_fields.grad_theta2,
        "gradeta2":    calculate_additional_fields.grad_eta2,
        "gradb2":      calculate_additional_fields.grad_b2,
    }
    return {
        name: fn(ds_merge, grid)
        for name, fn in _FIELD_FNS.items()
        if name in computed_feature_channels
    }


def _compute_kinematic_fields(
    ds_merge, grid, computed_feature_channels: list
) -> dict:
    """
    Compute callback for the ``kinematic`` subset.

    All velocity-derived properties are obtained in a single Jacobian pass
    via ``all_velocity_properties``; only channels listed in
    ``computed_feature_channels`` are returned.

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    computed_feature_channels : list of str

    Returns
    -------
    dict
    """
    velocity_props = calculate_additional_fields.all_velocity_properties(ds_merge, grid)
    return {
        name: field
        for name, field in velocity_props.items()
        if name in computed_feature_channels
    }


def _compute_frontogenesis_fields(
    ds_merge, grid, computed_feature_channels: list
) -> dict:
    """
    Compute callback for the ``geostrophic`` subset.

    Computes geostrophic velocities and geostrophic/ageostrophic frontogenesis
    via a single pass through ``all_frontogenesis_properties``.

    Dask graph mitigation
    ---------------------
    This subset merges two large lazy lineages: the velocity Jacobian (shared
    with ``dynamical``) and tracer gradients for buoyancy *and* Eta (needed for
    ug/vg).  When these lazy arrays are assembled into a single Dataset and
    written together, the combined task graph can trigger both the large-graph
    warning and the more serious run_spec scheduler warnings.

    To avoid this, all selected fields are materialised with a *single*
    ``dask.compute()`` call here.  This allows the Dask scheduler to fuse and
    optimise the shared subgraph in one round, after which NumPy arrays are
    returned.  Downstream zarr writes are then completely decoupled from the
    Dask graph.

    Parameters
    ----------
    ds_merge : xr.Dataset
    grid : xgcm.Grid
    computed_feature_channels : list of str

    Returns
    -------
    dict of str -> numpy.ndarray
        Materialised (not lazy) arrays for each requested channel.
    """
    props = calculate_additional_fields.all_frontogenesis_properties(ds_merge, grid)
    selected = {
        name: field
        for name, field in props.items()
        if name in computed_feature_channels
    }

    if not selected:
        return selected

    # Single dask.compute() call fuses the shared graph (Jacobian + tracer
    # gradients) into one scheduler submission, avoiding the run_spec warnings
    # that appear when multiple frontogenesis arrays are computed lazily later.
    keys = list(selected.keys())
    materialised = dask.compute(*[selected[k] for k in keys])
    return dict(zip(keys, materialised))


# ---------------------------------------------------------------------------
# Subset registry: maps subset name → compute callback
# ---------------------------------------------------------------------------

SUBSET_COMPUTE_FNS = {
    "native_fields":     _compute_native_fields,
    "frontal_structure": _compute_frontal_structure_fields,
    "kinematic":         _compute_kinematic_fields,
    "frontogenesis":         _compute_frontogenesis_fields,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse --config, --run_id, and --subset from sys.argv."""
    parser = argparse.ArgumentParser(
        description=(
            "Combined global LLC4320 dataset generation. "
            "Select which property subset to compute with --subset."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the combined YAML config file.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Override the run_id defined in the config YAML.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        choices=list(SUBSET_COMPUTE_FNS),
        help=(
            "Property subset to compute. "
            f"One of: {', '.join(SUBSET_COMPUTE_FNS)}. "
            "If omitted, the value of 'active_subset' in the config YAML is used."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_file: str = None, run_id: str = None, subset: str = None) -> None:
    """
    Entry point for the combined global dataset generation script.

    Can be called from the CLI (no arguments; reads ``--config``, ``--run_id``,
    and ``--subset`` from ``sys.argv``) or directly from Python by passing the
    arguments explicitly.

    Parameters
    ----------
    config_file : str, optional
        Path to the combined YAML config.  If ``None``, ``--config`` is read
        from ``sys.argv``.
    run_id : str, optional
        Override for the run identifier.  If ``None`` and called from the CLI,
        ``--run_id`` is used if provided.
    subset : str, optional
        One of the keys in ``SUBSET_COMPUTE_FNS``.  If ``None``, falls back to
        ``--subset`` from the CLI, then to the ``active_subset`` key in the
        YAML config.
    """
    # --- Resolve arguments ---------------------------------------------------
    if config_file is None:
        cli = _parse_args()
        config_file = cli.config
        run_id = run_id or cli.run_id
        subset = subset or cli.subset

    # --- Load raw YAML -------------------------------------------------------
    with open(config_file, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # Determine active subset: CLI arg > YAML active_subset key > error
    if subset is None:
        subset = raw.get("active_subset")
    if subset is None:
        raise ValueError(
            "No subset specified.  Pass --subset on the command line "
            f"(one of: {', '.join(SUBSET_COMPUTE_FNS)}), "
            "or set 'active_subset' in the config YAML."
        )
    if subset not in SUBSET_COMPUTE_FNS:
        raise ValueError(
            f"Unknown subset '{subset}'.  "
            f"Valid options: {list(SUBSET_COMPUTE_FNS)}"
        )

    # --- Resolve subset entry ------------------------------------------------
    subsets_cfg = raw.get("subsets", {})
    subset_entry = subsets_cfg.get(subset, {})

    if not subset_entry:
        raise ValueError(
            f"No entry found for subset '{subset}' under the 'subsets' key in "
            f"{config_file}.  Please add a 'subsets.{subset}' block."
        )

    # --- Build JobConfig in memory -------------------------------------------
    # The 'subsets' and 'active_subset' keys are top-level YAML keys that
    # config.load_config does not know about.  The JobConfig is built directly.

    output_dict = {**raw.get("output", {})}
    if "dataset_name" in subset_entry:
        output_dict["dataset_name"] = subset_entry["dataset_name"]

    cfg = config.JobConfig(
        run=config.RunConfig(**raw.get("run", {})),
        data=config.DataConfig(**raw.get("data", {})),
        sampling=config.SamplingConfig(**raw.get("sampling", {})),
        output=config.OutputConfig(**output_dict),
        features=config.FeaturesConfig(
            model_data_feature_channels=subset_entry.get(
                "model_data_feature_channels", []
            ),
            compute_features_channels=subset_entry.get(
                "compute_features_channels", []
            ),
        ),
        runtime=config.RuntimeConfig(**raw.get("runtime", {})),
    )

    run_global_pipeline(
        run_id=run_id,
        compute_fields_fn=SUBSET_COMPUTE_FNS[subset],
        cfg=cfg,
    )


#if __name__ == "__main__":
#    main()
