import skfmm
from scipy.ndimage import binary_erosion
import numpy as np


def llc_halo_mask(mask, dxC, dyC, halo_km):
    """
    Generate a halo mask around masked regions on an LLC grid using
    fast marching distances.

    This function expands masked regions by a specified physical distance
    (the halo) measured in kilometers. The mask is interpreted such that
    ``True`` indicates locations to be masked out. A signed level-set field
    is constructed and the distance to the mask interface is computed using
    the fast marching method.

    Parameters
    ----------
    mask : ndarray of bool, shape (face, j, i)
        Boolean mask on the LLC grid where ``True`` indicates regions to be
        masked out.
    dxC : xarray.DataArray
        Grid spacing in the i-direction (meters) defined on LLC faces.
    dyC : xarray.DataArray
        Grid spacing in the j-direction (meters) defined on LLC faces.
    halo_km : float
        Halo distance in kilometers. Grid points within this distance of
        masked regions will be excluded.

    Returns
    -------
    ndarray of bool
        Boolean halo mask of the same shape as ``mask`` where ``True``
        indicates points retained after applying the halo criterion.

    Notes
    -----
    * A signed level-set field ``phi`` is constructed with values ``-1`` inside
      masked regions and ``+1`` elsewhere. The zero level set is implicitly
      defined at the interface between masked and unmasked points.
    * Distances are computed in physical space using the maximum local grid
      spacing per face.
    * Faces that are entirely masked are returned unchanged.
    * Faces with no masked points return a mask of all ``True``.
    """

    nface = mask.shape[0]
    halo_mask = np.zeros_like(mask, dtype=bool)

    for face in range(nface):

        mask_f = np.asarray(mask[face]) # todo this is likely causing slowdown
        
        # mask_eroded = binary_erosion(mask_f)
        # boundary = mask_f & ~mask_eroded

        phi = np.ones_like(mask_f, dtype=float)
        phi[mask_f] = -1.0
        # phi[boundary] = 0.0

        if (phi==-1).any() and (phi==1).any():
            dx_max = dxC[face].mean() / 1000.0   # (i,) km mean is the best approximation we can do here.
            dy_max = dyC[face].mean() / 1000.0   # (j,) km

            dx_v = dx_max.values.item()
            dy_v = dy_max.values.item()

            # --- fast marching distance ---
            dist_km = skfmm.distance(
                phi,
                dx=(dy_v, dx_v)
            )

            # --- apply halo ---
            halo_mask[face] = dist_km >= halo_km
        elif (phi==-1).any(): # the whole face is masked out
            return mask_f
        else : # there are no values to mask out in this face
            halo_mask[face] = np.ones_like(mask_f, dtype=bool)

    return halo_mask
