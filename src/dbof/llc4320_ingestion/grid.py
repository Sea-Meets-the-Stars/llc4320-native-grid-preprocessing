# This dictionary is used to define the face connections for the LLC4320 grid.
# It is used to create the xgcm Grid object.

import numpy as np
import xarray as xr
import xgcm

#: COMODO coordinate attributes for the LLC4320 horizontal dimensions.
#: See docs/Grid.md.

COMODO_COORD_META = {
    'j':   {'axis': 'Y'},
    'j_g': {'axis': 'Y', 'c_grid_axis_shift': -0.5},
    'i':   {'axis': 'X'},
    'i_g': {'axis': 'X', 'c_grid_axis_shift': -0.5},
}


def comodo_attrs(dim, existing_attrs=None):
    """COMODO attrs to ADD for *dim*, empty when it needs none.

    The repo's one policy for these annotations: fill only where the
    store is silent, so a store that declares its own ``axis`` is
    believed and a reader's literal never wins over the data.

    Parameters
    ----------
    dim : str
        Dimension name.  Names outside :data:`COMODO_COORD_META` get ``{}``.
    existing_attrs : mapping, optional
        Attrs already on the coordinate.

    Returns
    -------
    dict
        Attrs to merge in, or ``{}`` to leave *dim* alone.

    """
    if dim not in COMODO_COORD_META:
        return {}
    if existing_attrs and 'axis' in existing_attrs:
        return {}
    return dict(COMODO_COORD_META[dim])


def ensure_comodo_attrs(ds, *, strict=False, source=None):
    """Annotate a dataset's horizontal dims so xgcm can find X and Y.

    The single application point for :func:`comodo_attrs`.  Callers differ
    only in whether an absent dim is an error: a full grid store without
    ``i_g`` is broken (*strict*), while a tile subset or an OSN kerchunk
    grid legitimately carries only some of the four.

    Parameters
    ----------
    ds : xarray.Dataset
        Grid dataset, tile-extent or full.  Not modified in place.
    strict : bool, default False
        Raise when a dim of :data:`COMODO_COORD_META` is absent instead
        of skipping it.
    source : str, optional
        Store path, quoted in the strict error.

    Returns
    -------
    xarray.Dataset
        ``ds`` with ``axis`` (and ``c_grid_axis_shift`` on the staggered
        dims) present on whichever of ``i``/``i_g``/``j``/``j_g`` it has.

    Raises
    ------
    ValueError
        *strict* and a dim is missing.

    """
    updates = {}
    for dim in COMODO_COORD_META:
        if dim not in ds.dims:
            if strict:
                raise ValueError(
                    f"grid store {source} is missing dimension {dim!r}; "
                    f"found {sorted(ds.dims)}")
            continue
        existing = (ds.coords[dim] if dim in ds.coords
                    else xr.DataArray(range(ds.sizes[dim]), dims=dim))
        attrs = comodo_attrs(dim, existing.attrs)
        if attrs:
            updates[dim] = existing.assign_attrs(attrs)
    return ds.assign_coords(updates) if updates else ds


face_connections = {'face':  {
        0: {'X': ((12, 'Y', False), (3, 'X', False)),
            'Y': (None, (1, 'Y', False))},
        1: {'X': ((11, 'Y', False), (4, 'X', False)),
            'Y': ((0, 'Y', False), (2, 'Y', False))},
        2: {'X': ((10, 'Y', False), (5, 'X', False)),
            'Y': ((1, 'Y', False), (6, 'X', False))},
        3: {'X': ((0, 'X', False), (9, 'Y', False)),
            'Y': (None, (4, 'Y', False))},
        4: {'X': ((1, 'X', False), (8, 'Y', False)),
            'Y': ((3, 'Y', False), (5, 'Y', False))},
        5: {'X': ((2, 'X', False), (7, 'Y', False)),
            'Y': ((4, 'Y', False), (6, 'Y', False))},
        6: {'X': ((2, 'Y', False), (7, 'X', False)),
            'Y': ((5, 'Y', False), (10, 'X', False))},
        7: {'X': ((6, 'X', False), (8, 'X', False)),
            'Y': ((5, 'X', False), (10, 'Y', False))},
        8: {'X': ((7, 'X', False), (9, 'X', False)),
            'Y': ((4, 'X', False), (11, 'Y', False))},
        9: {'X': ((8, 'X', False), None),
            'Y': ((3, 'X', False), (12, 'Y', False))},
        10: {'X': ((6, 'Y', False), (11, 'X', False)),
                'Y': ((7, 'Y', False), (2, 'X', False))},
        11: {'X': ((10, 'X', False), (12, 'X', False)),
                'Y': ((8, 'Y', False), (1, 'X', False))},
        12: {'X': ((11, 'X', False), None),
                'Y': ((9, 'Y', False), (0, 'X', False))}
}}

def set_xgcm_grid(ds_grid, use_connections:bool=True):
    """ Set the xgcm Grid object for the LLC4320 grid 
    Args:
        ds_grid: xarray.Dataset
            The dataset containing the LLC4320 grid data 

    Returns:
        xgcm.Grid
            The xgcm Grid object for the LLC4320 grid
    """
    # Do it
    if use_connections:
        grid = xgcm.Grid(ds_grid, padding='fill',
                         face_connections=face_connections)
    else:
        grid = xgcm.Grid(ds_grid, padding='fill')
    return grid


# ---------------------------------------------------------------------------
# Face seams
# ---------------------------------------------------------------------------

def invalid_seam_edges(kind='all'):
    """Face edges whose values cannot simply be copied from the neighbour.

    Two kinds: ``rotated`` (neighbour reached along the other axis, so a
    staggered component's value from across the boundary has to come
    from its partner) and ``open``
    (``None`` -- no neighbour at all).  Rotated edges are handled by
    passing both components to xgcm; open edges cannot be.

    Parameters
    ----------
    kind : {'all', 'rotated', 'open'}, default 'all'

    Returns
    -------
    set[tuple[int, str, str]]
        ``(face, axis, side)``, side in ``{'lower', 'upper'}``.

    Generated by LH and Claude
    """
    edges = set()
    for face, axes in face_connections['face'].items():
        for axis, (lower, upper) in axes.items():
            for side, link in (('lower', lower), ('upper', upper)):
                is_open = link is None
                if is_open or link[1] != axis:
                    if kind == 'all' or (kind == 'open') == is_open:
                        edges.add((face, axis, side))
    return edges


def face_seam_mask(da, grid, width=1, kind='open'):
    """NaN the *width* cells adjacent to each OPEN domain edge.

    Rotated edges are not masked: passing both components to xgcm takes
    their values from across the boundary from the partner face.  Open
    edges have no neighbour, so there is nothing to take and those cells
    stay NaN.

    A no-op on a grid without face connections (a tile has no seams).

    Parameters
    ----------
    da : xarray.DataArray
        Field with a ``face`` dim plus one X and one Y horizontal dim.
    grid : xgcm.Grid
    width : int, default 1
        Cells to invalidate at each edge.
    kind : {'open', 'rotated', 'all'}, default 'open'
        Which edges to mask; see :func:`invalid_seam_edges`.  ``'all'``
        is the pre-exchange behaviour, kept for before/after comparison.

    Returns
    -------
    xarray.DataArray

    Generated by LH and Claude
    """
    # attribute renamed between xgcm 0.8 and 0.9
    if not any(getattr(grid, a, None) is not None
               for a in ('_face_connections', '_connections')):
        return da

    xdim = next(d for d in ('i_g', 'i') if d in da.dims)
    ydim = next(d for d in ('j_g', 'j') if d in da.dims)
    shape = (da.sizes['face'], da.sizes[ydim], da.sizes[xdim])
    bad = np.zeros(shape, dtype=bool)
    pos = {'Y': 1, 'X': 2}
    for face, axis, side in invalid_seam_edges(kind):
        sl = [slice(None)] * 3
        sl[0] = face
        sl[pos[axis]] = (slice(0, width) if side == 'lower'
                         else slice(-width, None))
        bad[tuple(sl)] = True
    return da.where(~xr.DataArray(bad, dims=('face', ydim, xdim)))
