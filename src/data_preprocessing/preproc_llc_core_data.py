import xarray as xr
import numpy as np


def process_llc4320(ds, grid_ds):
    """
    Merge LLC4320 state variables with grid geometry and construct a land mask.

    This function combines a dataset containing LLC4320 model fields with the
    corresponding grid dataset, retaining only the grid variables required for
    downstream analysis. A land/ocean mask is constructed from the vertical
    cell fraction field and added to both the merged dataset and the grid-only
    dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing LLC4320 model state variables (e.g., tracers,
        velocities) defined on LLC faces.
    grid_ds : xarray.Dataset
        Dataset containing LLC4320 grid geometry variables.

    Returns
    -------
    ds_merge : xarray.Dataset
        Merged dataset containing model state variables, selected grid
        geometry variables, and a land/ocean mask.
    ds_grid : xarray.Dataset
        Grid-only dataset containing selected geometry variables and the
        land/ocean mask.

    Notes
    -----
    * Only a subset of grid variables required for analysis is retained.
    * The land mask ``maskC`` is defined such that ocean points (``hFacC > 0``)
      are set to 1 and land points are set to NaN.
    * Coordinates in the grid dataset are reset before subsetting to ensure
      consistent merging behavior.
    """
    
    ds_grid = grid_ds
    coords_to_keep = ['XC', 'YC', 'dxC', 'dyC', 'dxG', 'dyG', 'rAz', 'rA', 'Depth', 'hFacC', 'SN', 'CS']
    ds_grid = ds_grid.reset_coords()[coords_to_keep]

    ds_merge = xr.merge([ds, ds_grid])

    # Create a land mask
    ds_merge['maskC'] = xr.where(ds_merge.hFacC > 0, 1, np.nan)
    ds_grid['maskC'] = xr.where(ds_merge.hFacC > 0, 1, np.nan)

    return ds_merge, ds_grid

