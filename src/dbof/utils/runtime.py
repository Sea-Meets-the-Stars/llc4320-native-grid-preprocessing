"""
Shared runtime helpers for the ``generate_global_*`` pipeline scripts.

Centralises config resolution, feature-channel extraction, dask distributed
client setup, and zarr async-concurrency propagation so that each pipeline
script can replace ~30 lines of boilerplate with a couple of function calls.
"""

import logging

import dask
import zarr


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_config(cfg, config_file=None, run_id=None, *, config_module=None):
    """
    Parse / load config and apply an optional *run_id* override.

    Parameters
    ----------
    cfg : JobConfig or None
        Pre-built config.  When provided, *config_file* is ignored.
    config_file : str or None
        Path to the YAML config.  If both *cfg* and *config_file* are
        ``None``, ``config.parse_args()`` reads ``--config`` and
        ``--run_id`` from the command line.
    run_id : str or None
        Explicit run-id override; takes precedence over the YAML value.
    config_module : module
        The ``dbof.dataset_creation.config`` module (pass it in to avoid
        a hard import dependency).

    Returns
    -------
    cfg : JobConfig
        Resolved configuration object.
    """
    if config_module is None:
        import dbof.dataset_creation.config as config_module

    if cfg is None:
        if config_file is None:
            cli = config_module.parse_args()
            config_file = cli.config
            run_id = run_id or cli.run_id
        cfg = config_module.load_config(config_file)

    if run_id is not None:
        cfg = config_module.JobConfig(
            run=config_module.RunConfig(run_id=run_id, log_dir=cfg.run.log_dir),
            data=cfg.data,
            sampling=cfg.sampling,
            output=cfg.output,
            features=cfg.features,
            runtime=cfg.runtime,
        )

    return cfg


# ---------------------------------------------------------------------------
# Feature-channel extraction
# ---------------------------------------------------------------------------

def extract_feature_channels(cfg):
    """
    Return cleaned ``(model_feature_channels, computed_feature_channels)``
    lists from the config's ``features`` section.
    """
    model = [c.strip() for c in cfg.features.model_data_feature_channels if c.strip()]
    computed = [c.strip() for c in cfg.features.compute_features_channels if c.strip()]
    return model, computed


# ---------------------------------------------------------------------------
# Dask distributed client
# ---------------------------------------------------------------------------

def create_dask_client(runtime_cfg):
    """
    Set zarr async concurrency, create a dask distributed ``Client``,
    and propagate the zarr setting to all workers.

    Parameters
    ----------
    runtime_cfg : RuntimeConfig
        Must expose ``zarr_async_concurrency`` and optionally
        ``dask_n_workers``, ``dask_threads_per_worker``,
        ``dask_memory_limit``.

    Returns
    -------
    client : dask.distributed.Client
    """
    from dask.distributed import Client

    # Local zarr concurrency (scheduler process).
    zarr.config.set({'async.concurrency': runtime_cfg.zarr_async_concurrency})

    # Build Client kwargs from RuntimeConfig fields (bare Client() when all None).
    client_kwargs = {}
    if getattr(runtime_cfg, 'dask_n_workers', None) is not None:
        client_kwargs["n_workers"] = runtime_cfg.dask_n_workers
    if getattr(runtime_cfg, 'dask_threads_per_worker', None) is not None:
        client_kwargs["threads_per_worker"] = runtime_cfg.dask_threads_per_worker
    if getattr(runtime_cfg, 'dask_memory_limit', None) is not None:
        client_kwargs["memory_limit"] = runtime_cfg.dask_memory_limit

    client = Client(**client_kwargs)
    logging.info(f"Dask distributed client: {client}")
    dask.config.set({"distributed.scheduler.allowed-failures": 10})

    # Propagate zarr concurrency to every worker process.
    concurrency = runtime_cfg.zarr_async_concurrency

    def _set_zarr_concurrency(c):
        import zarr as _zarr
        _zarr.config.set({'async.concurrency': c})

    client.run(_set_zarr_concurrency, concurrency)
    logging.info(f"Zarr async concurrency set to {concurrency} on scheduler + all workers")

    return client
