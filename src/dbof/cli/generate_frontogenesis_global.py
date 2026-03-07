"""
Global LLC4320 frontogenesis dataset generation.

This script produces a zarr store containing frontogenesis-derived fields
(tendency, geostrophic and ageostrophic components, geostrophic velocities)
plus ``gradb2``.

Computed-field mode: ``all_frontogenesis_properties``.

CLI usage
---------
    generate-global-frontogenesis-dataset --config configs/frontogenesis_global.yaml [--run_id my_run]

Date format
-----------
All ``date_iterations`` entries in the YAML must use ISO format:
    'YYYY-MM-DD HH:MM:SS'  e.g. '2012-09-11 12:00:00'
"""

import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
from dbof.cli._generate_global_base import run_global_pipeline


def _compute_frontogenesis_fields(ds_merge, grid, computed_feature_channels: list) -> dict:
    """
    Compute mode-specific derived fields for the frontogenesis pipeline.

    All frontogenesis properties are obtained in a single pass via
    ``all_frontogenesis_properties``; only channels listed in
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
    frontogenesis_props = calculate_additional_fields.all_frontogenesis_properties(
        ds_merge, grid
    )
    return {
        name: field
        for name, field in frontogenesis_props.items()
        if name in computed_feature_channels
    }


def main(config_file: str = None, run_id: str = None) -> None:
    """
    Entry point for native-grid LLC frontogenesis dataset generation.

    Can be called from the CLI (no arguments; reads ``--config`` and
    ``--run_id`` from ``sys.argv``) or directly from Python by passing
    ``config_file`` and optionally ``run_id``.
    """
    run_global_pipeline(config_file, run_id, _compute_frontogenesis_fields)


#if __name__ == "__main__":
#    main()
