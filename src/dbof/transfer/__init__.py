"""LLC4320 -> S3 transfer pipeline.

One unified pipeline (:mod:`dbof.transfer.pipeline`) transfers either the whole
native dataset or a single spatial chunk; the only difference is the spatial
extent, selected by ``transfer.mode`` in the YAML config:

* ``all``    -- the whole native dataset (all faces / 720x720 tiles).
* ``chunks`` -- a single native 720x720 chunk surrounding a lat/lon, resolved by
  :mod:`dbof.transfer.chunk_selection`.

Both write the static grid once, then loop over ``data.date_iterations`` writing
one time-varying store per date.  All zarr/source IO and the per-store write
(:func:`~dbof.transfer.zarr_io.write_store`) live in :mod:`dbof.transfer.zarr_io`.
The ``transfer-timestep`` CLI (:mod:`dbof.cli.transfer_llc4320`) calls
:func:`dbof.transfer.pipeline.run`.
"""
