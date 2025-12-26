import skfmm
from scipy.ndimage import binary_erosion
import numpy as np


def llc_halo_mask(mask, dxC, dyC, halo_km):

    nface = mask.shape[0]
    halo_mask = np.zeros_like(mask, dtype=bool)

    for face in range(nface):

        mask_f = np.asarray(mask[face])
        mask_eroded = binary_erosion(mask_f)
        boundary = mask_f & ~mask_eroded

        # todo document this logic
        phi = np.ones_like(mask_f, dtype=float)
        phi[mask_f] = -1.0
        phi[boundary] = 0.0

        if (phi==0).any():
            dx_max = dxC[face].max() / 1000.0   # (i,) km
            dy_max = dyC[face].max() / 1000.0   # (j,) km

            dx_max = dx_max.values.item()
            dy_max = dy_max.values.item()

            # --- fast marching distance ---
            dist_km = skfmm.distance(
                phi,
                dx=(dy_max, dx_max)
            )

            # --- apply halo ---
            halo_mask[face] = dist_km >= halo_km

        else : # there are no values to mask out in this face
            halo_mask[face] = np.ones_like(mask_f, dtype=bool)

    return halo_mask
