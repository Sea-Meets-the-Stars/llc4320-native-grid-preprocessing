"""
Dask distributed client setup for global dataset generation pipelines.
"""

import logging

import dask
import zarr


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
