import xarray as xr

import dbof.preprocessing.halo_mask as halo_mask

def generate_static_land_face_masks_for_sampling(ds_grid, target_km_res):
    """
    Construct a composite sampling mask for the LLC native grid. These are the unchanging masks, so not ice.

    The mask excludes:
      - land points (via hFacC)
      - grid-face perimeter cells
    and applies a halo buffer based on the target physical resolution to land and face perimeter cells.

    Parameters
    ----------
    ds_grid : xarray.Dataset
        LLC grid dataset_creation containing metric terms.
    target_km_res : float
        Target physical resolution (km) used to define halo width.

    Returns
    -------
    ndarray of bool
        Boolean halo mask of the same shape as ``mask`` where ``True``
        indicates points retained after applying the halo criterion.
    """

    halo_km = target_km_res  # buffer to account for mean usage

    DXC = ds_grid["dxC"].persist()
    DYC = ds_grid["dyC"].persist()
    land_mask = (ds_grid.hFacC == 0).persist()

    halo_land_mask = halo_mask.llc_halo_mask(
        mask=land_mask,
        dxC=DXC,
        dyC=DYC,
        halo_km=halo_km
    )

    faces_perimeter_mask = xr.zeros_like(ds_grid.XC).astype(bool)
    faces_perimeter_mask.loc[dict(j=0)] = True
    faces_perimeter_mask.loc[dict(j=(faces_perimeter_mask.coords.sizes["j"] - 1))] = True
    faces_perimeter_mask.loc[dict(i=0)] = True
    faces_perimeter_mask.loc[dict(i=(faces_perimeter_mask.coords.sizes["i"] - 1))] = True

    halo_faces_perimeter_mask = halo_mask.llc_halo_mask(
        mask=faces_perimeter_mask,
        dxC=DXC,
        dyC=DYC,
        halo_km=halo_km
    )

    merged_mask = halo_land_mask & halo_faces_perimeter_mask

    return merged_mask