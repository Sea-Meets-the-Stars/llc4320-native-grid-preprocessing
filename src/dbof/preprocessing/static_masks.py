import xarray as xr

import dbof.preprocessing.halo_mask as halo_mask

def generate_halo_land_mask(ds_grid, target_km_res, DXC=None, DYC=None, stitched=True):
    """
    Construct a static *land-only* sampling mask for the LLC grid.

    Excludes land points (via hFacC) and applies a halo buffer based on the 
        target physical resolution to land perimeter cells.

    Parameters
    ----------
    ds_grid : xarray.Dataset
    target_km_res : int
    DXC : xarray.DataArray
    DYC : xarray.DataArray
    stitched : bool
    stitched is telling if the grid is in the native coords or has been stitched to a flat array.
    """

    halo_km = target_km_res  # buffer to account for mean usage

    DXC = ds_grid["dxC"].persist() if DXC is None else DXC
    DYC = ds_grid["dyC"].persist() if DYC is None else DYC
    land_mask = (ds_grid.hFacC == 0).persist()

    if stitched:
        halo_land_mask = halo_mask.stitched_halo_mask(
            mask=land_mask,
            dxC=DXC,
            dyC=DYC,
            halo_km=halo_km
        )
    else :
        halo_land_mask = halo_mask.llc_native_grid_halo_mask(
            mask=land_mask,
            dxC=DXC,
            dyC=DYC,
            halo_km=halo_km
        )
    return halo_land_mask

# Dead functions but kept for future use if we want to mask face boundaries in the future

# def generate_static_face_mask_for_sampling(ds_grid, target_km_res, DXC=None, DYC=None):
#     """
#     Construct a static *face-only* sampling mask for the LLC native grid.
#
#     Excludes grid-face perimeter cells and applies a halo buffer based on the
#         target physical resolution to face perimeter cells.
#
#     .. note::
#        Currently unused by the stitched-grid cutout pipeline (which has no face
#        perimeters). Kept for potential future native-grid / per-face sampling.
#
#     Notes
#     ----------
#     This is one component of the overall sampling mask, which also includes a land mask.
#     See ``generate_static_land_face_masks_for_sampling`` for full mask conventions (e.g., ``True`` means retained).
#     """
#
#     halo_km = target_km_res  # buffer to account for mean usage
#
#     DXC = ds_grid["dxC"].persist() if DXC is None else DXC
#     DYC = ds_grid["dyC"].persist() if DYC is None else DYC
#
#     faces_perimeter_mask = xr.zeros_like(ds_grid.XC).astype(bool)
#     faces_perimeter_mask.loc[dict(j=0)] = True
#     faces_perimeter_mask.loc[dict(j=(faces_perimeter_mask.coords.sizes["j"] - 1))] = True
#     faces_perimeter_mask.loc[dict(i=0)] = True
#     faces_perimeter_mask.loc[dict(i=(faces_perimeter_mask.coords.sizes["i"] - 1))] = True
#
#     halo_faces_perimeter_mask = halo_mask.llc_halo_mask(
#         mask=faces_perimeter_mask,
#         dxC=DXC,
#         dyC=DYC,
#         halo_km=halo_km
#     )
#
#     return halo_faces_perimeter_mask


# def generate_static_land_face_masks_for_sampling(ds_grid, target_km_res):
#     """
#     Construct a composite sampling mask for the LLC native grid. These are the unchanging masks, so not ice.
#
#     The mask excludes:
#       - land points (via hFacC)
#       - grid-face perimeter cells
#     and applies a halo buffer based on the target physical resolution to land and face perimeter cells.
#
#     Parameters
#     ----------
#     ds_grid : xarray.Dataset
#         LLC grid cutout_dataset_creation containing metric terms.
#     target_km_res : float
#         Target physical resolution (km) used to define halo width.
#
#     Returns
#     -------
#     ndarray of bool
#         Boolean halo mask of the same shape as ``mask`` where ``True``
#         indicates points retained after applying the halo criterion.
#     """
#
#     halo_km = target_km_res  # buffer to account for mean usage
#
#     DXC = ds_grid["dxC"].persist()
#     DYC = ds_grid["dyC"].persist()
#
#     halo_land_mask = generate_static_land_mask_for_sampling(ds_grid, target_km_res, DXC=DXC, DYC=DYC)
#     halo_faces_perimeter_mask = generate_static_face_mask_for_sampling(ds_grid, target_km_res, DXC=DXC, DYC=DYC)
#
#     merged_mask = halo_land_mask & halo_faces_perimeter_mask
#
#     return merged_mask