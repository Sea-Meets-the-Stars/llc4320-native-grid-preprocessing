"""
LLC face-grid utilities: staggered→tracer interpolation, vector-pair
metadata, face→lat-lon stitching, and masked ``(C, H, W)`` assembly.

The low-level stitch (``faces_dataset_to_latlon``) wraps xmitgcm with an
xarray-compatibility fix.  The higher-level helpers
(``interp_staggered_to_tracer``, ``set_vector_pair_attrs``,
``stitch_and_mask``) are used by the pipeline ``process_snapshot`` functions
to prepare fields before and after the stitch.
"""

import logging

import numpy as np
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
    https://xmitgcm.readthedocs.io/en/latest/_modules/xmitgcm/llcreader/llcmodel.html

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


# ---------------------------------------------------------------------------
# Staggered → tracer interpolation
# ---------------------------------------------------------------------------
# NOTE: interp_staggered_to_tracer and set_vector_pair_attrs are no longer
# used by the production processors (process_surface / process_depth).
# Vector channels are now interpolated AND rotated to geographic components
# by the subset compute functions (rotate_vector_to_geographic) and stitched
# as scalars.  These helpers are kept for tests and reference — the
# mate/vector stitch path they enable applies a staggered-grid pixel shift
# that misregisters tracer-point data on the rotated faces
# (see tests/test_vector_rotation_equivalence.py).

#: Maps staggered variable names to the xgcm axis along which they must be
#: interpolated to reach the tracer-point (C-grid center) location.
STAGGER_MAP = {
    'V':       'Y',
    'U':       'X',
    'oceTAUY': 'Y',
    'oceTAUX': 'X',
}


def interp_staggered_to_tracer(fields, grid, stagger_map=None):
    """
    Interpolate staggered-grid variables to tracer points (in-place update).

    Parameters
    ----------
    fields : dict-like
        Mapping of ``{var_name: DataArray}`` **or** an ``xr.Dataset``.
        Staggered variables that are present in both *fields* and *stagger_map*
        are replaced with their tracer-point interpolated versions.
        Variables not in *stagger_map* (or absent from *fields*) are untouched.
    grid : xgcm.Grid
        The xgcm Grid object used for interpolation.
    stagger_map : dict, optional
        ``{variable_name: xgcm_axis}`` override.  Defaults to
        ``STAGGER_MAP`` (U/V + oceTAUX/oceTAUY).
    """
    if stagger_map is None:
        stagger_map = STAGGER_MAP

    for var, axis in stagger_map.items():
        if var in fields:
            fields[var] = grid.interp(fields[var], axis, boundary='fill')


# ---------------------------------------------------------------------------
# Vector-pair metadata
# ---------------------------------------------------------------------------

#: Default vector pairs: ``(x_component, y_component)``.
#: ``faces_dataset_to_latlon`` reads ``attrs['mate']`` on the *x*-component
#: to know which variable is its meridional partner for sign-aware rotation.
DEFAULT_VECTOR_PAIRS = [
    ('U', 'V'),
    ('oceTAUX', 'oceTAUY'),
]


def set_vector_pair_attrs(ds, vector_pairs=None):
    """
    Set ``mate`` attributes on vector-pair variables in *ds*.

    For each ``(x_comp, y_comp)`` pair:

    * Clear any existing ``mate`` attr on *y_comp*.
    * Set ``mate = y_comp`` on *x_comp*.

    Only pairs where **both** components are present in *ds* are touched.

    Parameters
    ----------
    ds : xr.Dataset
        Modified in-place (attribute mutation only — no data copy).
    vector_pairs : list of (str, str), optional
        Defaults to ``DEFAULT_VECTOR_PAIRS``.
    """
    if vector_pairs is None:
        vector_pairs = DEFAULT_VECTOR_PAIRS

    for x_comp, y_comp in vector_pairs:
        if x_comp in ds.variables and y_comp in ds.variables:
            ds[y_comp].attrs.pop('mate', None)
            ds[x_comp].attrs['mate'] = y_comp


# ---------------------------------------------------------------------------
# Face stitch + mask + stack
# ---------------------------------------------------------------------------

def stitch_and_mask(ds_to_convert, channels, mask_dict, progress_bar=False):
    """
    Stitch LLC faces to lat-lon, materialise, and return a masked ``(C, H, W)`` array.

    Parameters
    ----------
    ds_to_convert : xr.Dataset
        Face-gridded dataset containing all *channels*, already on the tracer
        grid (vector fields rotated to geographic components upstream).  Any
        'mate' attrs are stripped so all channels use the scalar stitch path.
    channels : list[str]
        Ordered channel names to include in the output array.
    mask_dict : dict[str, DataArray]
        Boolean mask arrays (``True`` = masked / NaN) to attach before
        stitching.  All masks are OR-combined after materialisation.
        Common entries: ``{'_land_mask': ..., '_ice_mask': ...}``.
    progress_bar : bool, default False
        When ``True``, materialisation is wrapped in a
        ``dask.diagnostics.ProgressBar`` context.

    Returns
    -------
    data : np.ndarray, shape ``(C, H, W)``
        Stacked channel array with masked pixels set to ``NaN``.
    """
    # Attach masks to the dataset so they ride along through the face stitch.
    mask_names = list(mask_dict.keys())
    ds_to_convert = ds_to_convert.assign(mask_dict)
    all_vars = channels + mask_names

    # Defensive: strip any stray 'mate' attrs so every channel goes through
    # the SCALAR stitch path.  Vector channels are rotated to geographic
    # (east/north) components upstream (rotate_vector_to_geographic), and
    # the mate/vector stitch path must not run on top of that — it applies
    # a staggered-grid pixel shift that misregisters tracer-point data
    # (see tests/test_vector_rotation_equivalence.py).  Raw variables read
    # via xmitgcm's llcreader carry 'mate' attrs by default.
    for vname in ds_to_convert.variables:
        ds_to_convert[vname].attrs.pop('mate', None)

    # Face → lat-lon stitch.
    logging.info("Converting LLC faces -> rectangular lat/lon")
    ds_rect = faces_dataset_to_latlon(
        ds_to_convert[all_vars],
        metric_vector_pairs=[],
    )

    # Materialise.
    logging.info("Materialising stitched arrays")
    if progress_bar:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            mask_arrays = [ds_rect[m].values.astype(bool) for m in mask_names]
            channel_arrays = [ds_rect[ch].values for ch in channels]
    else:
        mask_arrays = [ds_rect[m].values.astype(bool) for m in mask_names]
        channel_arrays = [ds_rect[ch].values for ch in channels]

    # Combine all masks with OR.
    combined_mask = mask_arrays[0]
    for m in mask_arrays[1:]:
        combined_mask = combined_mask | m

    # Stack into (C, H, W) and apply mask.
    logging.info("Stacking into (C, H, W)")
    data = np.stack(channel_arrays, axis=0)
    data = np.where(combined_mask[np.newaxis], np.nan, data)

    return data