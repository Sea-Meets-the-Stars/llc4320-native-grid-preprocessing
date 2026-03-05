# grid / face-to-latlon stitching
import xarray as xr
from xmitgcm.llcreader import llcmodel
from xmitgcm.llcreader.llcmodel import (
    _faces_coords_to_latlon,
    _faces_to_latlon_scalar,
    _faces_to_latlon_vector,
    _drop_facedim,
)

def faces_dataset_to_latlon(ds, metric_vector_pairs):
    """
    This function is based on xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon
    but has been updated to be compatible with all xarray versions.

    xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon() stitches the 13 LLC
    faces into a single coherent 2D rectangular image. This is NOT interpolation
    — values are pixel-shifted and some faces are rotated to tile correctly, but
    we remain on the native LLC grid.
    Input:  xr.Dataset with (face, j, i) dimensions, shape (13, 4320, 4320) per var
    Output: xr.Dataset with (lat, lon) dimensions, shape (12960, 17280) per var

    The upstream function contains:
        ds_new = ds_new.update(data_vars)      # reassigns to None in xarray < 0.17
        ds_new = ds_new.set_coords(...)        # AttributeError: 'NoneType'...
    because Dataset.update() was in-place (returning None) in xarray < 0.17.
    This version uses xr.merge() instead, which has stable semantics across all
    xarray versions.
    """
    coord_vars = list(ds.coords)
    ds_new = _faces_coords_to_latlon(ds)    # skeleton Dataset with expanded coords
    nfaces = len(ds['face'])

    # Classify variables as scalars or vector pairs (same logic as upstream).
    vector_pairs = []
    vnames = list(ds.reset_coords().variables)
    for vname in list(vnames):
        try:
            mate = ds[vname].attrs['mate']
        except KeyError:
            mate = None
        if mate is not None:
            vector_pairs.append((vname, mate))
            try:
                vnames.remove(mate)
            except ValueError:
                raise ValueError(
                    f"If '{vname}' in varnames, '{mate}' must also be in varnames"
                )

    all_vector_components = [
        inner for outer in (vector_pairs + metric_vector_pairs) for inner in outer
    ]
    scalars = [v for v in vnames if v not in all_vector_components]

    data_vars = {}

    for vname in scalars:
        if vname == 'face' or vname in ds_new:
            continue
        if 'face' in ds[vname].dims:
            data = _faces_to_latlon_scalar(ds[vname].data, nfaces=nfaces)
            dims = _drop_facedim(ds[vname].dims)
        else:
            data = ds[vname].data
            dims = ds[vname].dims
        data_vars[vname] = xr.Variable(dims, data, ds[vname].attrs)

    for vname_u, vname_v in vector_pairs:
        u_data, v_data = _faces_to_latlon_vector(
            ds[vname_u].data, ds[vname_v].data, nfaces=nfaces
        )
        data_vars[vname_u] = xr.Variable(
            _drop_facedim(ds[vname_u].dims), u_data, ds[vname_u].attrs
        )
        data_vars[vname_v] = xr.Variable(
            _drop_facedim(ds[vname_v].dims), v_data, ds[vname_v].attrs
        )

    for vname_u, vname_v in metric_vector_pairs:
        u_data, v_data = _faces_to_latlon_vector(
            ds[vname_u].data, ds[vname_v].data, nfaces=nfaces, metric=True
        )
        data_vars[vname_u] = xr.Variable(
            _drop_facedim(ds[vname_u].dims), u_data, ds[vname_u].attrs
        )
        data_vars[vname_v] = xr.Variable(
            _drop_facedim(ds[vname_v].dims), v_data, ds[vname_v].attrs
        )

    # Use xr.merge instead of ds_new.update() to avoid the xarray < 0.17 issue
    # where Dataset.update() returned None rather than the modified dataset.
    ds_out = xr.merge([ds_new, xr.Dataset(data_vars)])
    ds_out = ds_out.set_coords([c for c in coord_vars if c in ds_out])
    return ds_out