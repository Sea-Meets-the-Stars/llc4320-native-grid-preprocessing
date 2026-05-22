"""
Shared helpers for subset resolution, ``JobConfig`` construction, and
per-date pipeline dispatch.

All three ``generate_global_*`` entry points follow the same pattern:
load the YAML, resolve which subset to run, build a ``JobConfig`` from
the raw dict + subset entry, and optionally loop over dates.  This
module factors out that shared logic.
"""

import logging

import dbof.dataset_creation.config as config
from dbof.utils.iterations import date_to_run_id


# ---------------------------------------------------------------------------
# Subset resolution
# ---------------------------------------------------------------------------

def resolve_subset(raw: dict, subset: str | None, valid_subsets: dict) -> tuple[str, dict]:
    """
    Determine the active subset name and its YAML entry.

    Parameters
    ----------
    raw : dict
        The full raw YAML dict (``yaml.safe_load(...)``).
    subset : str or None
        Subset name from CLI / caller.  Falls back to ``raw["active_subset"]``.
    valid_subsets : dict
        ``SUBSET_COMPUTE_FNS`` dispatch table — used only for validation.

    Returns
    -------
    subset : str
        Validated subset name.
    subset_entry : dict
        The ``subsets.<name>`` block from the YAML.

    Raises
    ------
    ValueError
        If the subset is unknown or missing from the YAML ``subsets`` block.
    """
    if subset is None:
        subset = raw.get("active_subset")
    if subset is None:
        raise ValueError(
            "No subset specified.  Pass --subset on the command line "
            f"(one of: {', '.join(valid_subsets)}), "
            "or set 'active_subset' in the config YAML."
        )
    if subset not in valid_subsets:
        raise ValueError(
            f"Unknown subset '{subset}'.  "
            f"Valid options: {list(valid_subsets)}"
        )

    subsets_cfg = raw.get("subsets", {})
    subset_entry = subsets_cfg.get(subset, {})
    if not subset_entry:
        raise ValueError(
            f"No entry found for subset '{subset}' under the 'subsets' key.  "
            f"Please add a 'subsets.{subset}' block to the YAML."
        )

    return subset, subset_entry


# ---------------------------------------------------------------------------
# JobConfig construction
# ---------------------------------------------------------------------------

def build_job_config(raw: dict, subset_entry: dict) -> config.JobConfig:
    """
    Build a ``JobConfig`` from the raw YAML dict and the resolved subset entry.

    Used by ``generate_front_training_data.py`` and any pipeline that needs
    the full config (sampling, range-mode iteration fields, etc.).

    Parameters
    ----------
    raw : dict
        Full raw YAML dict.
    subset_entry : dict
        The specific subset block (e.g. ``raw["subsets"]["kinematic"]``).
    """
    output_dict = {**raw.get("output", {})}
    if "dataset_name" in subset_entry:
        output_dict["dataset_name"] = subset_entry["dataset_name"]

    return config.JobConfig(
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


def build_global_job_config(raw: dict, subset_entry: dict) -> config.GlobalJobConfig:
    """
    Build a ``GlobalJobConfig`` from the raw YAML dict and the resolved subset.

    This is the slim variant used by the three ``generate_global_*`` pipelines.
    It requires only the sections those scripts actually use: ``run``, ``data``
    (just ``date_iterations`` and ``endpoint_url``), ``output``, ``features``,
    and ``runtime``.  No ``sampling`` section is needed.

    Parameters
    ----------
    raw : dict
        Full raw YAML dict.
    subset_entry : dict
        The specific subset block (e.g. ``raw["subsets"]["kinematic"]``).
    """
    output_dict = {**raw.get("output", {})}
    if "dataset_name" in subset_entry:
        output_dict["dataset_name"] = subset_entry["dataset_name"]

    return config.GlobalJobConfig(
        run=config.RunConfig(**raw.get("run", {})),
        data=config.GlobalDataConfig(**raw.get("data", {})),
        output=config.GlobalOutputConfig(**output_dict),
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


# ---------------------------------------------------------------------------
# Per-date pipeline dispatch
# ---------------------------------------------------------------------------

def run_per_date(
    raw: dict,
    subset_entry: dict,
    date_iterations: list[str],
    pipeline_fn,
    compute_fields_fn,
    run_id: str | None = None,
    **pipeline_kwargs,
) -> None:
    """
    Run the pipeline once per date, each in its own date subdirectory.

    Output layout::

        s3://{bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}

    where *date_prefix* is derived from each date string via
    :func:`date_to_run_id` (e.g. ``'2012-11-09 12:00:00'`` →
    ``'20121109_120000'``).

    Parameters
    ----------
    raw : dict
        Full raw YAML dict.
    subset_entry : dict
        The resolved subset YAML block.
    date_iterations : list[str]
        Date strings to iterate over (from ``data.date_iterations``).
    pipeline_fn : callable
        The ``run_global_pipeline`` function to call for each date.
    compute_fields_fn : callable
        The subset compute callback.
    run_id : str or None, optional
        Explicit run_id override.  If ``None``, the value from the YAML
        config (``raw["run"]["run_id"]``) is used.
    **pipeline_kwargs
        Extra keyword arguments forwarded to *pipeline_fn* (e.g.
        ``apply_icemask``, ``s3_source``, ``surface_only``, ``config_dir``).
    """
    effective_run_id = run_id or raw.get("run", {}).get("run_id", "default")
    print(
        f"Will create a date subdirectory under run_id='{effective_run_id}' "
        f"for each of the {len(date_iterations)} date(s) in date_iterations."
    )
    for date_str in date_iterations:
        date_prefix = date_to_run_id(date_str)
        print(f"\n{'='*60}")
        print(f"Processing date: {date_str}  →  date_prefix: {date_prefix}")
        print(f"{'='*60}")

        # Build a single-date config so only this date is processed.
        single_date_raw = {**raw}
        single_date_raw["data"] = {
            **raw.get("data", {}),
            "date_iterations": [date_str],
        }
        cfg = build_global_job_config(single_date_raw, subset_entry)

        pipeline_fn(
            run_id=effective_run_id,
            compute_fields_fn=compute_fields_fn,
            cfg=cfg,
            date_prefix=date_prefix,
            **pipeline_kwargs,
        )

    print(f"\nAll {len(date_iterations)} date(s) processed.")
