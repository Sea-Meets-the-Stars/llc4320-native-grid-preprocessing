"""
Vertical-axis utilities for LLC4320 depth-resolved diagnostics.

These helpers are careful about which vertical dimension and coordinate
they use.  Functions that operate on tracer-point fields (Theta, Salt,
density, buoyancy, N²) use ``Z`` / ``k``.  Functions that operate on
W-point fields use ``Zl`` / ``k_l``.  Velocity fields (U, V) also live
on ``Z`` / ``k`` vertically, but are horizontally staggered — the caller
is responsible for interpolating to tracer points when needed.
"""

import warnings

import numpy as np
import xarray as xr


# ===========================================================================
#  Vertical dimension / coordinate look-ups
# ===========================================================================

def _get_vertical_dim(da_in):
    """Return the name of the vertical dimension present in *da_in*.

    Parameters
    ----------
    da_in : xr.DataArray
        Array whose dimensions are searched for a known vertical axis name.

    Returns
    -------
    str
        Name of the vertical dimension (e.g. ``'k'``, ``'k_l'``, ``'Z'``).

    Raises
    ------
    ValueError
        If no recognised vertical dimension is found.
    """
    for dim in ("k", "k_l", "k_u", "k_p1", "Z", "Zl", "Zu", "Zp1",
                "depth", "lev"):
        if dim in da_in.dims:
            return dim
    raise ValueError(
        f"Could not identify vertical dimension from dims={da_in.dims}"
    )


def _get_depth_coord(ds_merge, zdim=None):
    """
    Return a 1D depth coordinate in metres, **positive downward**.

    The LLC4320 native convention stores depth as positive-upward
    (Z < 0 below the surface).  This function converts to positive-
    downward (0 at the surface, increasing with depth) by taking
    the absolute value.

    .. warning::
       This **inverts the native MITgcm/ECCO sign convention**: every
       depth used downstream of this helper (MLD comparisons,
       nearest-level selection, vertical derivatives) is positive-
       downward, opposite to the ``Z`` coordinate that other ECCO
       tooling exposes.  A ``UserWarning`` is emitted when the flip
       is applied so the conversion never happens silently.

    For tracer-level fields (zdim='k') this returns ``|Z|``.
    For W-level fields (zdim='k_l') this returns ``|Zl|``.

    LLC4320 specific — expects ``Z`` or ``Zl`` to exist in *ds_merge*.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing the ``Z`` and/or ``Zl`` coordinates.
    zdim : str or None, optional
        Vertical dimension name (``'k'`` or ``'k_l'``).  Inferred from
        ``ds_merge.Theta`` when *None*.

    Returns
    -------
    xr.DataArray
        1D depth coordinate (m, positive downward) with dimension *zdim*.
    """
    if zdim is None:
        zdim = _get_vertical_dim(ds_merge.Theta)

    coord_name = {"k": "Z", "k_l": "Zl"}[zdim]
    z = ds_merge[coord_name]

    # xmitgcm stores Z with dim='Z' but data variables use dim='k'.
    # Rename so xarray aligns element-wise along the vertical.
    if z.dims[0] != zdim:
        z = z.rename({z.dims[0]: zdim})

    # MITgcm stores depth as negative-upward; convert to positive-downward.
    # Warn loudly: this inverts the native sign convention, which matters
    # to anyone comparing against other ECCO/MITgcm tooling.
    if np.nanmean(z.values) < 0:
        warnings.warn(
            f"Flipping vertical coordinate '{coord_name}' from the native "
            "MITgcm negative-upward convention to POSITIVE-DOWNWARD depth "
            "(0 at the surface, increasing with depth). All depths "
            "downstream of dbof vertical helpers use this convention.",
            UserWarning, stacklevel=2)
        z = -z

    return z


def _get_vertical_spacing(ds_merge, zdim=None):
    """
    Layer thickness [m] appropriate for the vertical grid.

    - For tracer levels (zdim='k'):  returns ``drF`` — the thickness of
      each tracer cell.
    - For W levels (zdim='k_l'):  returns ``drC`` — the distance between
      adjacent tracer cell centres (i.e. the spacing of W levels).

    Falls back to finite-differencing the depth coordinate if the model
    spacing variable is unavailable.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing ``drF`` / ``drC`` spacing variables.
    zdim : str or None, optional
        Vertical dimension name (``'k'`` or ``'k_l'``).  Inferred from
        ``ds_merge.Theta`` when *None*.

    Returns
    -------
    xr.DataArray
        1D layer thickness (m) with dimension *zdim*.
    """
    if zdim is None:
        zdim = _get_vertical_dim(ds_merge.Theta)

    # Choose the correct MITgcm spacing variable for this vertical grid.
    _SPACING_MAP = {"k": "drF", "k_l": "drC"}
    spacing_var = _SPACING_MAP.get(zdim, "drF")

    if spacing_var in ds_merge:
        dz = ds_merge[spacing_var]
        if zdim not in dz.dims:
            dz = dz.rename({dz.dims[0]: zdim})
        return dz

    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz_vals = np.gradient(z.values.astype(float))
    return xr.DataArray(dz_vals, dims=(zdim,), coords={zdim: z})


def _nearest_k_to_depth(z_vals, target_depth):
    """Return the integer k-index whose depth is nearest to *target_depth*.

    Both *z_vals* and *target_depth* must use the **positive-downward**
    convention (0 at the surface, increasing with depth) — the same
    convention returned by ``_get_depth_coord``.

    Parameters
    ----------
    z_vals : np.ndarray
        1D depth coordinate, positive downward, in metres (as returned
        by ``_get_depth_coord``).
    target_depth : float
        Target depth in metres, positive downward.

    Returns
    -------
    int
        Index into *z_vals* closest to *target_depth*.
    """
    return int(np.abs(z_vals - float(target_depth)).argmin())


# ===========================================================================
#  Derivatives and interpolation
# ===========================================================================

def _vertical_derivative(field, ds_merge):
    """
    Compute d(field)/dz on the field's own vertical grid.

    Positive z is downward (so dθ/dz < 0 means θ decreases with depth).

    The stencil spans **two layers** at interior points (centered
    difference between k-1 and k+1), and **one layer** at the top
    and bottom boundaries (forward / backward difference respectively):

    - k = 0      : forward   (f[1] - f[0])   / (z[1] - z[0])
    - 1 ≤ k < N-1: centered  (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    - k = N-1    : backward  (f[N-1] - f[N-2]) / (z[N-1] - z[N-2])

    The spacing between levels comes from the positive-downward depth
    coordinate (``_get_depth_coord``).

    Implemented via ``apply_ufunc`` with ``dask="parallelized"`` so the
    derivative is computed per-chunk without materialising the full array.

    Parameters
    ----------
    field : xr.DataArray
        3D dask-backed field to differentiate vertically.
    ds_merge : xr.Dataset
        Merged dataset used to retrieve the depth coordinate.

    Returns
    -------
    xr.DataArray
        Vertical derivative df/dz (same shape as *field*), dask-backed.
    """
    zdim = _get_vertical_dim(field)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)  # 1D, always small
    nk = len(z_vals)
    k_axis = field.dims.index(zdim)

    # Pre-compute spacing arrays for the 1D depth coordinate.
    # Interior:  dz_centered[k] = z[k+1] - z[k-1]
    # Forward:   dz_fwd        = z[1]    - z[0]
    # Backward:  dz_bwd        = z[-1]   - z[-2]
    dz_centered = z_vals[2:] - z_vals[:-2]                   # length nk-2
    dz_fwd = z_vals[1] - z_vals[0]
    dz_bwd = z_vals[-1] - z_vals[-2]

    def _deriv_along_k(f_chunk):
        f = f_chunk.astype(np.float64)

        # Slicing helper for arbitrary axis position
        def _sl(start, stop):
            s = [slice(None)] * f.ndim
            s[k_axis] = slice(start, stop)
            return tuple(s)

        out = np.empty_like(f)

        # Forward difference at k=0
        out[_sl(0, 1)] = (
            (f[_sl(1, 2)] - f[_sl(0, 1)]) / dz_fwd
        )
        # Centered differences at interior levels
        dz_shape = [1] * f.ndim
        dz_shape[k_axis] = nk - 2
        dz_bc = dz_centered.reshape(dz_shape)
        out[_sl(1, -1)] = (
            (f[_sl(2, None)] - f[_sl(None, -2)]) / dz_bc
        )
        # Backward difference at k=N-1
        out[_sl(-1, None)] = (
            (f[_sl(-1, None)] - f[_sl(-2, -1)]) / dz_bwd
        )
        return out

    return xr.apply_ufunc(
        _deriv_along_k,
        field,
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _select_at_depth(field, target_depth, ds_merge):
    """Select the nearest model level to *target_depth* [m, positive downward].

    Uses ``_nearest_k_to_depth`` to find the closest k-index, then returns
    a 2D slice via ``.isel()``.  No interpolation — this returns the value
    at the discrete model level closest to the requested depth.

    Parameters
    ----------
    field : xr.DataArray
        3D dask-backed field to slice.
    target_depth : float
        Target depth in metres, positive downward.
    ds_merge : xr.Dataset
        Merged dataset used to retrieve the depth coordinate.

    Returns
    -------
    xr.DataArray
        2D field at the nearest model level to *target_depth*, dask-backed.
    """
    zdim = _get_vertical_dim(field)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)
    k_idx = _nearest_k_to_depth(z_vals, target_depth)
    return field.isel({zdim: k_idx})


def _extract_at_mld(field3d, mld, ds_merge):
    """Extract a 3D field at per-column MLD depths via nearest-k indexing.

    ``mld`` is a 2D array (face, j, i) giving a different depth per column.
    For each column, the nearest model level to that column's MLD is found
    and the field value at that level is returned.  No vertical interpolation
    is performed — MLD itself is defined as a discrete model-level depth
    (the deepest Z where the density criterion holds), so nearest-k
    recovers the exact level.

    Implementation note
    ~~~~~~~~~~~~~~~~~~~
    ``apply_ufunc`` with ``input_core_dims=[[zdim], []]`` moves the
    vertical axis to the **last** position of each chunk before calling
    the inner function.  As a result the inner function receives:

    - ``field_chunk`` with shape ``(..., nk)``  (k axis last)
    - ``mld_chunk``   with shape ``(...)``      (2D spatial)

    All indexing inside the inner function operates with k in the last
    position — no ``moveaxis`` is needed.

    Parameters
    ----------
    field3d : xr.DataArray
        3D dask-backed field to extract from.
    mld : xr.DataArray
        2D mixed-layer depth (m, positive downward), one value per column.
    ds_merge : xr.Dataset
        Merged dataset used to retrieve the depth coordinate.

    Returns
    -------
    xr.DataArray
        2D field at the nearest model level to the per-column MLD,
        dtype float32, dask-backed.
    """
    zdim = _get_vertical_dim(field3d)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    z_vals = z.values.astype(np.float64)  # 1D, small — captured in closure

    def _nearest_k_at_mld_chunk(field_chunk, mld_chunk):
        # field_chunk: (..., nk) — apply_ufunc moved zdim to last axis
        # mld_chunk:   (...)     — 2D spatial
        spatial_shape = mld_chunk.shape
        mld_flat = mld_chunk.ravel().astype(np.float64)

        # Nearest k-index for each column
        k_idx = np.abs(z_vals[np.newaxis, :] - mld_flat[:, np.newaxis]).argmin(axis=1)

        # Flatten the spatial dims of field_chunk for advanced indexing
        f_flat = field_chunk.reshape(-1, field_chunk.shape[-1])  # (n_cols, nk)
        col_idx = np.arange(f_flat.shape[0])
        result_flat = f_flat[col_idx, k_idx]
        return result_flat.reshape(spatial_shape).astype(np.float32)

    return xr.apply_ufunc(
        _nearest_k_at_mld_chunk,
        field3d,
        mld,
        input_core_dims=[[zdim], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )


def _interp_w_to_tracer_levels(ds_merge):
    """
    Interpolate W from cell interfaces (k_l / Zl) to tracer cell centres
    (k / Z) by averaging adjacent interface levels.

    On the MITgcm C-grid, ``grid.interp(W, 'Z')`` is equivalent to
    ``0.5 * (W[k] + W[k+1])`` along the vertical.  This avoids fragile
    coordinate-based ``xr.interp`` + rename gymnastics that break across
    xarray versions.

    Parameters
    ----------
    ds_merge : xr.Dataset
        Merged dataset containing the ``W`` variable on the ``k_l`` grid
        and ``Theta`` on the ``k`` grid (used to determine the target
        number of tracer levels).

    Returns
    -------
    xr.DataArray
        Vertical velocity W interpolated to tracer levels (k / Z),
        dask-backed.
    """
    zdim_w = _get_vertical_dim(ds_merge.W)
    zdim_t = _get_vertical_dim(ds_merge.Theta)
    n_t = ds_merge.Theta.sizes[zdim_t]

    W = ds_merge.W
    W_upper = W.isel({zdim_w: slice(0, n_t)}).drop_vars(
        [c for c in W.coords if c == zdim_w or c == zdim_t],
        errors="ignore",
    )
    W_lower = W.isel({zdim_w: slice(1, n_t + 1)}).drop_vars(
        [c for c in W.coords if c == zdim_w or c == zdim_t],
        errors="ignore",
    )

    # Align dimensions: rename the sliced zdim_w → zdim_t so the arrays
    # can be added element-wise.
    W_upper = W_upper.rename({zdim_w: zdim_t})
    W_lower = W_lower.rename({zdim_w: zdim_t})

    return 0.5 * (W_upper + W_lower)


def _masked_ml_mean(field3d, mld, ds_merge):
    """
    Thickness-weighted mean of a 3D field over 0 ≤ z ≤ MLD.

    This is the standard depth-reduction for "mean over the mixed layer".
    The vertical dimension is reduced by the weighted sum, so the output
    is **2D** (face, j, i).

    Parameters
    ----------
    field3d : xr.DataArray
        3D dask-backed field to average.
    mld : xr.DataArray
        2D mixed-layer depth (m, positive downward), one value per column.
    ds_merge : xr.Dataset
        Merged dataset used to retrieve the depth coordinate and layer
        thicknesses.

    Returns
    -------
    xr.DataArray
        2D thickness-weighted mean over the mixed layer, dask-backed.
    """
    zdim = _get_vertical_dim(field3d)
    z = _get_depth_coord(ds_merge, zdim=zdim)
    dz = _get_vertical_spacing(ds_merge, zdim=zdim)

    mask = z <= mld
    num = (field3d * dz).where(mask).sum(dim=zdim)
    den = dz.where(mask).sum(dim=zdim)
    return num / den
