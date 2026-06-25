"""LLC4320 -> S3 transfer pipeline.

Two transfer modes share one set of zarr/source IO + writers (:mod:`dbof.transfer.zarr_io`):

* ``all``    -- :mod:`dbof.transfer.all_data`: transfer every spatial tile of a
  timestep (static grid once + time-varying fields per date).  This is the
  original ``transfer_llc4320`` behaviour.
* ``chunks`` -- :mod:`dbof.transfer.chunks`: transfer a single native
  720x720 spatial chunk (all depths) surrounding a lat/lon, for many
  timestamps.

The mode is selected by ``transfer.mode`` in the YAML config; the
``transfer-timestep`` CLI (:mod:`dbof.cli.transfer_llc4320`) dispatches.
"""
