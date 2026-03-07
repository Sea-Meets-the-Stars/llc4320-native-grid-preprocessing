"""
Global LLC4320 front-training dataset generation.

This script produces a zarr store containing the raw model channels plus
``gradb2`` (buoyancy-gradient magnitude squared) — the key feature used for
front detection.

Computed-field mode: ``gradb2`` only (plus optional ``relative_vorticity``).

CLI usage
---------
    generate-global-llc-dataset --config configs/global.yaml [--run_id my_run]

Date format
-----------
All ``date_iterations`` entries in the YAML must use ISO format:
    'YYYY-MM-DD HH:MM:SS'  e.g. '2012-09-11 12:00:00'
"""

import dbof.preprocessing.calculate_additional_fields as calculate_additional_fields
from dbof.cli._generate_global_base import run_global_pipeline


def _compute_fronts_fields(ds_merge, grid, computed_feature_channels: list) -> dict:
    """
    Compute mode-specific derived fields for the fronts pipeline.

    Currently only ``relative_vorticity`` is optional here; ``gradb2`` is
    handled automatically by the base pipeline and must NOT be included.

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
    fields = {}
    if "relative_vorticity" in computed_feature_channels:
        fields["relative_vorticity"] = calculate_additional_fields.relative_vorticity(
            ds_merge, grid
        )
    return fields


def main(config_file: str = None, run_id: str = None) -> None:
    """
    Entry point for native-grid LLC front-training dataset generation.

    Can be called from the CLI (no arguments; reads ``--config`` and
    ``--run_id`` from ``sys.argv``) or directly from Python by passing
    ``config_file`` and optionally ``run_id``.
    """
    run_global_pipeline(config_file, run_id, _compute_fronts_fields)


#if __name__ == "__main__":
#    main()
