"""Chunk transfer mode.

Instead of transferring every spatial tile of a timestep, this mode transfers a
single native LLC4320 720x720 spatial chunk -- all depth levels -- surrounding
a user-supplied ``lat,lon``, for a list of timestamps.

Resolution strategy
-------------------
The source store is the *native* LLC4320 grid with dims ``(face, j, i)`` and 2D
longitude/latitude fields ``XC``/``YC``.  We find the grid cell nearest the
target ``lat,lon`` by minimising an equirectangular distance over ``XC``/``YC``,
then floor the resulting ``(j, i)`` to the enclosing 720x720 block.  Because the
native store is already in ``(face, j, i)`` layout, no rectangular-grid stitch
is needed -- the chunk is one ``ds.isel(face=[f], j=slice, i=slice)`` away.

This is the native-source analogue of the rect-grid tile resolver on the
``tiles`` branch (``dbof.tiles.tile_mapping.rect_ij_to_tile``): that one maps a
*stitched rect-grid* pixel to a face + face-local 720 slices (handling per-face
rotations) for reading the post-transfer 3D field; here we map a *lat/lon*
directly to the native chunk for the transfer itself.

Output layout
-------------
::

    s3://{bucket}/{chunks_prefix}/{chunk_name}/grid.zarr      # written once
    s3://{bucket}/{chunks_prefix}/{chunk_name}/{run_id}/      # one per timestamp

where ``{run_id}`` is the directory-safe form of the ISO timestamp
(``'2011-12-09 12:00:00'`` -> ``'20111209_120000'``, via
:func:`~dbof.llc4320_ingestion.date_iterations.date_to_run_id`).  The
time-invariant grid (XC, YC, Z, hFac, ...) is written once per chunk location;
each timestamp directory holds only the time-varying fields for that chunk.
"""

from dataclasses import dataclass
import logging

import numpy as np

from dbof.transfer import config as config
from dbof.transfer import zarr_io
from dbof.llc4320_ingestion.date_iterations import (
    date_to_run_id,
    mit_date_to_iteration,
    mit_date_to_time_idx,
)

TILE_SIZE = 720


# ---------------------------------------------------------------------------
# Chunk resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkSelection:
    """Resolved native-grid chunk for a requested lat/lon."""
    face_idx: int        # LLC face holding the chunk
    j0: int              # face-local j start (multiple of TILE_SIZE)
    i0: int              # face-local i start (multiple of TILE_SIZE)
    tile: int            # chunk edge length (720)
    nearest_j: int       # face-local j of the cell nearest the target
    nearest_i: int       # face-local i of the cell nearest the target
    nearest_lat: float   # YC at the nearest cell
    nearest_lon: float   # XC at the nearest cell

    @property
    def j_slice(self) -> slice:
        return slice(self.j0, self.j0 + self.tile)

    @property
    def i_slice(self) -> slice:
        return slice(self.i0, self.i0 + self.tile)


def resolve_chunk(ds, lat: float, lon: float, tile: int = TILE_SIZE) -> ChunkSelection:
    """Resolve a lat/lon to the enclosing native 720x720 chunk.

    Parameters
    ----------
    ds : xarray.Dataset
        Native LLC4320 store; must carry 2D ``XC`` (lon) and ``YC`` (lat) on
        dims ``(face, j, i)``.
    lat, lon : float
        Target latitude / longitude in degrees.  ``lon`` may be given in either
        [-180, 180] or [0, 360]; longitude differences are wrapped.
    tile : int
        Chunk edge length (default 720).

    Returns
    -------
    ChunkSelection
    """
    if "XC" not in ds or "YC" not in ds:
        raise ValueError("Source store must contain XC and YC to resolve lat/lon.")

    XC = ds["XC"]
    YC = ds["YC"]
    for needed in ("face", "j", "i"):
        if needed not in XC.dims:
            raise ValueError(f"XC must have dim '{needed}'; got dims {XC.dims}")

    # Equirectangular squared distance with longitude wrap; good enough for
    # nearest-neighbour selection away from the poles.
    dlon = ((XC - lon + 180.0) % 360.0) - 180.0
    dlat = YC - lat
    d2 = (dlon * np.cos(np.deg2rad(lat))) ** 2 + dlat ** 2

    # Flatten over (face, j, i) and take the argmin lazily.
    stacked = d2.stack(cell=("face", "j", "i"))
    pos = int(stacked.argmin("cell").compute())

    face_idx = int(stacked["face"].values[pos])
    nearest_j = int(stacked["j"].values[pos])
    nearest_i = int(stacked["i"].values[pos])

    nearest_lat = float(YC.isel(face=face_idx, j=nearest_j, i=nearest_i).values)
    nearest_lon = float(XC.isel(face=face_idx, j=nearest_j, i=nearest_i).values)

    j0 = (nearest_j // tile) * tile
    i0 = (nearest_i // tile) * tile

    return ChunkSelection(
        face_idx=face_idx, j0=j0, i0=i0, tile=tile,
        nearest_j=nearest_j, nearest_i=nearest_i,
        nearest_lat=nearest_lat, nearest_lon=nearest_lon,
    )


def slice_to_chunk(ds, sel: ChunkSelection):
    """Return *ds* sliced to one face and the chunk's 720x720 spatial block.

    Both the C-grid (``j``/``i``) and any staggered (``j_g``/``i_g``) horizontal
    dims are sliced with the same index range, so all variables in the chunk
    share the ``(face=1, 720, 720)`` extent.
    """
    isel = {"face": [sel.face_idx]}
    for d in ("j", "j_g"):
        if d in ds.dims:
            isel[d] = sel.j_slice
    for d in ("i", "i_g"):
        if d in ds.dims:
            isel[d] = sel.i_slice
    return ds.isel(**isel)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(cfg: config.JobConfig, init_store: bool = False,
        skip_existing: bool = False) -> None:
    """Execute a chunk transfer for all configured timestamps.

    Writes the grid once to ``{chunk_name}/grid.zarr`` and the time-varying
    fields to ``{chunk_name}/{timestamp}/`` for each timestamp.
    """
    t = cfg.transfer
    loc = t.location
    source = cfg.data.MIT_data_path

    static_variables = list(t.static_variables or [])
    time_variables = list(t.variables or [])
    if not time_variables:
        raise ValueError("transfer.variables is empty; nothing to transfer for chunks mode.")

    ds = zarr_io.open_source(source)
    zarr_io.validate_variables_present(ds, static_variables + time_variables)
    if "time" not in ds.dims:
        raise ValueError("Source dataset has no 'time' dimension.")

    # --- Resolve the chunk surrounding the requested lat/lon ----------------
    sel = resolve_chunk(ds, loc.lat, loc.lon, tile=t.tile_i)
    logging.info(
        f"Resolved ({loc.lat}, {loc.lon}) -> face={sel.face_idx}, "
        f"j={sel.j0}:{sel.j0 + sel.tile}, i={sel.i0}:{sel.i0 + sel.tile} "
        f"(nearest cell j={sel.nearest_j}, i={sel.nearest_i} at "
        f"lat={sel.nearest_lat:.4f}, lon={sel.nearest_lon:.4f})"
    )
    ds_chunk = slice_to_chunk(ds, sel)

    folder = f"{cfg.output.chunks_prefix.strip('/')}/{loc.chunk_name.strip('/')}"
    provenance = {
        "requested_lat": float(loc.lat),
        "requested_lon": float(loc.lon),
        "chunk_name": loc.chunk_name,
        "resolved_face": int(sel.face_idx),
        "j_start": int(sel.j0),
        "i_start": int(sel.i0),
        "tile_size": int(sel.tile),
        "nearest_cell_j": int(sel.nearest_j),
        "nearest_cell_i": int(sel.nearest_i),
        "nearest_cell_lat": float(sel.nearest_lat),
        "nearest_cell_lon": float(sel.nearest_lon),
        "source_path": source,
    }

    # --- Grid: written once per chunk location ------------------------------
    if static_variables:
        s3_url = zarr_io._build_s3_url(cfg.output.bucket, folder, t.static_dataset_name)
        logging.info(f"--- Chunk grid transfer: {len(static_variables)} variables -> {s3_url} ---")
        root = zarr_io.open_zarr_store(s3_url, cfg.output.s3_endpoint, init_store)
        zarr_io.safe_set_attrs(root, provenance)
        zarr_io.transfer_variables(ds_chunk, static_variables, root,
                                   tile_j=sel.tile, tile_i=sel.tile,
                                   time_idx=None, skip_existing=skip_existing)
        logging.info("Chunk grid transfer complete.")

    # --- Time-varying fields: one store per timestamp -----------------------
    # Timestamps are ISO 'YYYY-MM-DD HH:MM:SS' (matching data.date_iterations in
    # the all-data config).  The S3 directory uses the directory-safe run-id form.
    ntime = ds.sizes["time"]
    for date_str in t.timestamps:
        iteration = mit_date_to_iteration(date_str)
        time_idx = mit_date_to_time_idx(date_str, ntime)
        run_dir = date_to_run_id(date_str)

        s3_url = zarr_io._build_s3_url(cfg.output.bucket, folder, run_dir)
        logging.info(
            f"--- Chunk transfer {date_str}: {len(time_variables)} variables -> {s3_url} ---\n"
            f"    iteration={iteration}  time_idx={time_idx}"
        )
        root = zarr_io.open_zarr_store(s3_url, cfg.output.s3_endpoint, init_store)
        zarr_io.safe_set_attrs(root, {
            **provenance,
            "timestamp": date_str,
            "selected_iteration": int(iteration),
            "selected_date_utc": date_str,
        })
        zarr_io.transfer_variables(ds_chunk, time_variables, root,
                                   tile_j=sel.tile, tile_i=sel.tile,
                                   time_idx=time_idx, skip_existing=skip_existing)
        logging.info(f"Chunk transfer for {date_str} complete.")

    logging.info("All chunk transfers complete.")
