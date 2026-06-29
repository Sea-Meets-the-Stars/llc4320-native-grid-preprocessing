import numpy as np
import torch
import torch.nn.functional as F

def create_image_patch(ds, channels, patch):
    """
    Construct a multi-channel image patch from an xarray Dataset.

    Extracts a rectangular spatial patch for each requested variable and
    stacks them into a channel-first image tensor.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the requested variables on LLC faces. 
        Note : Non tracer values must be shifted to tracer location (example : ds_merge["V"] = grid.interp(ds_merge["V"], 'Y', boundary='fill')
    channels : sequence of str
        Names of variables in `ds` to include as image channels.
    patch : dict
        Dictionary specifying the patch location with keys:
        ``face``, ``i_start``, ``i_end``, ``j_start``, ``j_end``.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(C, H, W)`` with dtype ``float32``,
        where ``C`` is the number of channels.
    """
    channels_array = []
    for channel in channels:
        feature = ds[channel].isel(
            face=patch["face"],
            j=slice(patch["j_start"], patch["j_end"] + 1),
            i=slice(patch["i_start"], patch["i_end"] + 1)).compute()

        channels_array.append(feature.values)

    img = np.stack(channels_array, axis=0)   # (C, H, W)

    return torch.from_numpy(img).float()

def downsample_image(img, target_dim=64):
    """
    Downsample a channel-first image tensor to a fixed square resolution.

    Uses area-based interpolation. Upsampling is not supported.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor of shape ``(C, H, W)``.
    target_dim : int, optional
        Target spatial dimension ``(target_dim, target_dim)``.

    Returns
    -------
    torch.Tensor
        Downsampled image tensor of shape ``(C, target_dim, target_dim)``.

    Raises
    ------
    ValueError
        If ``target_dim`` is larger than the input spatial dimensions.
    """
    
    C, H, W = img.shape

    if target_dim > H or target_dim > W:
        raise ValueError("Upsampling is not allowed yet")

    img = img.unsqueeze(0)  # (1, C, H, W)

    out = F.interpolate(
        img,
        size=(target_dim, target_dim),
        mode="area"
    )

    out = out.squeeze(0)

    return out


def extent_in_i(dxC, j0, i0, km_x):
    """
    Compute the index extent in the i-direction corresponding to a physical
    distance in kilometers, on the stitched (j, i) grid.

    Parameters
    ----------
    dxC : np.ndarray
        Grid spacing (meters) in the i-direction, shape ``(j, i)``.
    j0 : int
        Central j-index.
    i0 : int
        Central i-index.
    km_x : float
        Target half-width in kilometers.

    Returns
    -------
    L : int
        Number of grid cells to include to the left of ``i0``.
    R : int
        Number of grid cells to include to the right of ``i0``.
    real_km_w : float
        Actual physical width (km) spanned by the selected indices.
    """

    dx_row = dxC[j0]

    dx_row = 0.5 * (dx_row[:-1] + dx_row[1:])  # move from i_g to i, this is sort of interpolating. Average i_g value on left and right of cell center
    dx_row = dx_row.astype(np.float64) / 1000. # meters to km

    cum_left = np.cumsum(dx_row[i0::-1])
    cum_right = np.cumsum(dx_row[i0:])

    L = np.searchsorted(cum_left, km_x)
    R = np.searchsorted(cum_right, km_x)

    if L == len(cum_left): # we hit the grid edge
        L = R # just use the right side instead. They will almost always be equal
    elif R == len(cum_right):
        R = L

    return L, R, np.sum(dx_row[i0-L:i0]) + np.sum(dx_row[i0:i0+R])

def extent_in_j(dyC, j0, i0, km_y):
    """
    Compute the index extent in the j-direction corresponding to a physical
    distance in kilometers, on the stitched (j, i) grid.

    Parameters
    ----------
    dyC : np.ndarray
        Grid spacing (meters) in the j-direction, shape ``(j, i)``.
    j0 : int
        Central j-index.
    i0 : int
        Central i-index.
    km_y : float
        Target half-height in kilometers.

    Returns
    -------
    D : int
        Number of grid cells to include downward from ``j0``.
    U : int
        Number of grid cells to include upward from ``j0``.
    real_km_h : float
        Actual physical height (km) spanned by the selected indices.
    """

    dy_col = dyC[:, i0]

    dy_col = 0.5 * (dy_col[:-1] + dy_col[1:])         # j_g → j
    dy_col = dy_col.astype(np.float64) / 1000.

    cum_dn = np.cumsum(dy_col[j0::-1])
    cum_up = np.cumsum(dy_col[j0:])

    D = np.searchsorted(cum_dn, km_y)
    U = np.searchsorted(cum_up, km_y)

    if D == len(cum_dn): # we hit the grid edge
        D = U
    elif U == len(cum_up):
        U = D

    return D, U, np.sum(dy_col[j0-D:j0])+np.sum(dy_col[j0:j0+U])

def get_lat_lon_extents_of_patch(index, dxC, dyC, grid_shape, km_size):
    """
    Determine index bounds for a square spatial patch of a given physical size
    on the stitched (j, i) grid.

    Given a central grid index, this function computes the i- and j-index
    extents required to approximate a square patch of size ``km_size`` using
    local grid spacing. Patches that would run off the global grid edge are
    rejected.

    Parameters
    ----------
    index : tuple of int
        Central index ``(j, i)``.
    dxC, dyC : np.ndarray
        Grid spacing (meters), shape ``(j, i)``.
    grid_shape : tuple of int
        ``(n_j, n_i)`` shape of the global grid.
    km_size : float
        Target physical size of the patch in kilometers.

    Returns
    -------
    dict or None
        Dictionary with patch bounds and realized physical dimensions:
        ``i_start``, ``i_end``, ``j_start``, ``j_end``,
        ``real_km_w``, ``real_km_h``.
        Returns ``None`` if the patch would extend beyond the grid edge.
    """

    half_km = km_size / 2

    j, i = index
    n_j, n_i = grid_shape

    L, R, real_km_w = extent_in_i(dxC, j, i, half_km)

    D, U, real_km_h = extent_in_j(dyC, j, i, half_km)

    if ((i - L) < 0):  # runs off the global grid edge
        print("i_start < 0 -- patch runs off the grid edge")
        return None
    i_start = i - L

    if (n_i - 1) < (i + R):
        print("i_end past grid edge")
        return None
    i_end = i + R

    if ((j - D) < 0):  # runs off the global grid edge
        print("j_start < 0 -- patch runs off the grid edge")
        return None
    j_start = j - D

    if (n_j - 1) < (j + U):
        print("j_end past grid edge")
        return None
    j_end   = j + U

    return dict(i_start=i_start, i_end=i_end, j_start=j_start, j_end=j_end, real_km_w = real_km_w, real_km_h = real_km_h)
