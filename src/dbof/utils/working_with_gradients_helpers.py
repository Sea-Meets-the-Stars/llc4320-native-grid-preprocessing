import numpy as np

def vmin_vmax_of_log_values_dask_based_xarray(da):
    """Compute robust log10-scale color limits for a DataArray.

    Loads all values into memory, takes the absolute value, filters out
    zeros and NaNs, then returns the 1st and 99th percentiles of the
    log10-transformed values. Useful for setting ``vmin``/``vmax`` when
    plotting fields that span many orders of magnitude.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array (may be dask-backed; will be computed in full).

    Returns
    -------
    vmin : float
        1st percentile of log10(|values|).
    vmax : float
        99th percentile of log10(|values|).

    Notes
    -----
    This loads the entire array into RAM, so ensure sufficient memory is
    available.
    """
    vals = np.abs(da.values.ravel())
    vals = vals[(vals > 0) & ~np.isnan(vals)]
    logvals = np.log10(vals)

    # find vmin and max for plot
    vmin = np.quantile(logvals, 0.01)
    vmax = np.quantile(logvals, 0.99)
    return vmin, vmax