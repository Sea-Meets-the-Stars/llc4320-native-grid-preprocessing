import xarray as xr
import numpy as np


def process_llc4320(ds, ds_grid):
    """
    Merge LLC4320 state variables with grid geometry and construct a land mask.

    This function combines a dataset_creation containing LLC4320 model fields with the
    corresponding grid dataset_creation. ds_grid should be the product of process_llc4320_grid see bellow

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing LLC4320 model state variables (e.g., tracers,
        velocities) defined on LLC faces.
    ds_grid : xarray.Dataset
        Should be the product of process_llc4320_grid see bellow

    Returns
    -------
    ds_merge : xarray.Dataset
        Merged dataset_creation containing model state variables, selected grid
        geometry variables, and a land/ocean mask.

    """

    ds_merge = xr.merge([ds, ds_grid])

    # Create a land mask
    # ds_merge['maskC'] = xr.where(ds_merge.hFacC > 0, 1, np.nan)
    # ds_grid['maskC'] = xr.where(ds_merge.hFacC > 0, 1, np.nan)

    return ds_merge #, ds_grid


def process_llc4320_grid (grid_ds):
    """
    Merge LLC4320 state variables with grid geometry and construct a land mask.

    This function updates the grid ds to retain only the grid variables required for
    downstream analysis.

    Parameters
    ----------
    grid_ds : xarray.Dataset
        Dataset containing LLC4320 grid geometry variables.

    Returns
    -------
    ds_grid : xarray.Dataset
        Grid-only dataset_creation containing selected geometry variables and the
        land/ocean mask.

    """

    ds_grid = grid_ds
    coords_to_keep = ['XC', 'YC', 'dxC', 'dyC', 'dxG', 'dyG', 'rAz', 'rA', 'Depth', 'hFacC', 'SN', 'CS']
    ds_grid = ds_grid.reset_coords()[coords_to_keep]

    return ds_grid

