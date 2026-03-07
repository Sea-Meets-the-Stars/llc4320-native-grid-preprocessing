"""
Global LLC4320 ocean-properties dataset generation.

This script produces a zarr store containing velocity-derived scalar fields
(vorticity, strain, divergence, Rossby number, Okubo-Weiss parameter, etc.)
plus ``gradb2``.

Computed-field mode: ``all_velocity_properties`` via a single Jacobian pass.

CLI usage
---------
    generate-global-properties-dataset --config configs/global_properties.yaml [--run_id my_run]

Date format
-----------
All ``date_iterations`` entries in the YAML must use ISO format:
    'YYYY-MM-DD HH:MM:SS'  e.g. '2012-09-11 12:00:00'
"""

import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
from dbof.cli._generate_global_base import run_global_pipeline


def _compute_properties_fields(ds_merge, grid, computed_feature_channels: list) -> dict:
    """
    Compute mode-specific derived fields for the properties pipeline.

    All velocity-derived properties are obtained in a single Jacobian pass via
    ``all_velocity_properties``; only channels listed in
    ``computed_feature_channels`` are returned.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged LLC4320 model data for one time snapshot.
    grid : xgcm.Grid
        xGCM grid object built from the LLC4320 grid dataset.
    computed_feature_channels : list of str
        Channel names requested in the YAML config.

    Returns
    -------
    dict
        Mapping of ``{channel_name: DataArray}`` for each channel in
        ``computed_feature_channels`` that this function provides.
    """
    velocity_props = calculate_additional_fields.all_velocity_properties(ds_merge, grid)
    return {
        name: field
        for name, field in velocity_props.items()
        if name in computed_feature_channels
    }


def main(config_file: str = None, run_id: str = None) -> None:
    """
    Entry point for native-grid LLC ocean-properties dataset generation.

    Can be called from the CLI (no arguments; reads ``--config`` and
    ``--run_id`` from ``sys.argv``) or directly from Python by passing
    ``config_file`` and optionally ``run_id``.
    """
    run_global_pipeline(config_file, run_id, _compute_properties_fields)


#if __name__ == "__main__":
#    main()
