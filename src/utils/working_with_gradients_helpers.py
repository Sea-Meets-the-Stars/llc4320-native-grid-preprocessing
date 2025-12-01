import numpy as np

# Warning - need some solid ram for this bad boy
def vmin_vmax_of_log_values_dask_based_xarray(da):
    vals = np.abs(da.values.ravel())
    vals = vals[(vals > 0) & ~np.isnan(vals)]
    logvals = np.log10(vals)

    # find vmin and max for plot
    vmin = np.quantile(logvals, 0.01)
    vmax = np.quantile(logvals, 0.99)
    return vmin, vmax