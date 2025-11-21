import xarray as xr
import numpy as np


def process_llc4320(ds, grid_ds):

    # Create a merged dataset of our ds values and the grid file
    # Filter only the values we need from the grid
    ds_grid = grid_ds
    coords_to_keep = ['XC', 'YC', 'dxC', 'dyC', 'dxG', 'dyG', 'rAz', 'rA', 'Depth', 'hFacC', 'SN', 'CS']
    ds_grid = ds_grid.reset_coords()[coords_to_keep]

    ds_merge = xr.merge([ds, ds_grid])

    # Create a land mask
    ds_merge['maskC'] = xr.where(ds_merge.hFacC > 0, 1, np.nan)
    ds_grid['maskC'] = xr.where(ds_merge.hFacC > 0, 1, np.nan)

    return ds_merge, ds_grid

