"""Resolve a lat/lon to a single native LLC4320 720x720 spatial chunk.

The unified transfer pipeline (:mod:`dbof.transfer.pipeline`) transfers either
the full native dataset or a single spatial chunk; this module provides the
chunk path's spatial selection.

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
"""

from dataclasses import dataclass

import numpy as np

TILE_SIZE = 720


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


def chunk_provenance(sel: ChunkSelection, lat: float, lon: float,
                     chunk_name: str, source: str) -> dict:
    """Build the provenance attrs recorded on every store for a chunk transfer."""
    return {
        "requested_lat": float(lat),
        "requested_lon": float(lon),
        "chunk_name": chunk_name,
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
