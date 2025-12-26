import numpy as np
import torch
import torch.nn.functional as F

# Take in the patch of coordinates and construct a multi-channel image
def create_image_patch(ds, channels, patch):
    # Move data to tracer point

    channels_array = []
    for channel in channels:
        feature = ds[channel].isel(
            face=patch["face"],
            j=slice(patch["j_start"], patch["j_end"] + 1),
            i=slice(patch["i_start"], patch["i_end"] + 1))
        channels_array.append(feature.values)

    img = np.stack(channels_array, axis=0)   # (C, H, W)

    return torch.from_numpy(img).float()

#img = tensor
def downsample_image(img, target_dim=64):
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


# This will take in our indexes and generate spatial patches of fixed km x and y
def extent_in_i(ds, face, j0, i0, km_x):
    dx_row = ds.dxC.sel(face=face).isel(j=j0).values

    dx_row = 0.5 * (dx_row[:-1] + dx_row[1:])  # move from i_g to i, this is sort of interpolating. Average i_g value on left and right of cell center
    dx_row = dx_row.astype(np.float64) / 1000. # meters to km

    cum_left = np.cumsum(dx_row[i0::-1])
    cum_right = np.cumsum(dx_row[i0:])

    #print(cum_left.shape, cum_right.shape)

    L = np.searchsorted(cum_left, km_x)
    R = np.searchsorted(cum_right, km_x)

    if L == len(cum_left): # we hit the face boundary
        print("HIT FACE")
        L = R # just use the right side instead. They will almost always be equal
    elif R == len(cum_right):
        print("HIT FACE")
        R = L

    return L, R, np.sum(dx_row[i0-L:i0]) + np.sum(dx_row[i0:i0+R])

def extent_in_j(ds, face, j0, i0, km_y):
    dy_col = ds.dyC.sel(face=face).isel(i=i0).values  # shape (j_g)

    dy_col = 0.5 * (dy_col[:-1] + dy_col[1:])         # j_g → j
    dy_col = dy_col.astype(np.float64) / 1000.

    cum_dn = np.cumsum(dy_col[j0::-1])
    cum_up = np.cumsum(dy_col[j0:])

    D = np.searchsorted(cum_dn, km_y)
    U = np.searchsorted(cum_up, km_y)

    #print(cum_dn.shape, cum_up.shape)

    if D == len(cum_dn): # we hit the face boundary
        D = U
        print(f"HIT FACE{len(cum_dn)}")
    elif U == len(cum_up):
        U = D
        print("HIT FACE")

    return D, U, np.sum(dy_col[j0-D:j0])+np.sum(dy_col[j0:j0+U])

def get_lat_lon_extents_of_patch(index, ds_merge, km_size):
    half_km = km_size / 2

    f,j,i = index

    L,R,real_km_w = extent_in_i(ds_merge, f, j, i, half_km)

    D,U,real_km_h = extent_in_j(ds_merge, f, j, i, half_km)

    #print(L,R,D,U)

    # todo we could stitch these pataches instead of throwing them out or we could mask these points out at sampling time

    if ((i - L) < 0):  # this would extend accross face lines
        print("STARTING i INDICES IS LESS THAN ZERO AND THEREFORE OFF THE FACE OR DATA")
        return None
    i_start = i - L

    if (ds_merge.sizes['i'] - 1) < (i + R):
        print("ENDING j INDICES IS GREATER THAN END OF FACE AND THEREFORE OFF THE FACE OR DATA")
        return None
    i_end = i + R

    if ((j - D) < 0):  # this would extend accross face lines
        print("STARTING j INDEX IS LESS THAN ZERO AND THEREFORE OFF THE FACE OR DATA")
        return None
    j_start = j - D

    if (ds_merge.sizes['j'] - 1) < (j + U):
        print("ENDING j INDICES IS GREATER THAN END OF FACE AND THEREFORE OFF THE FACE OR DATA")
        return None
    j_end   = j + U

    return dict(face=f, i_start=i_start, i_end=i_end, j_start=j_start, j_end=j_end, real_km_w = real_km_w, real_km_h = real_km_h)
