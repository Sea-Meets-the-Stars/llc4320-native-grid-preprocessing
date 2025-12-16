import numpy as np
import dask.array as da

def compute_pdf(log_arr, nbins=200, eps=1e-12, q_low=0.001, q_high=0.999):
    # log_array = in memory numpy array

    vmin = float(np.nanpercentile(log_arr, 100 * q_low))
    vmax = float(np.nanpercentile(log_arr, 100 * q_high))


    # Clip extreme outliers to [vmin, vmax] - this squases them on the ends so expect to see spikes on either end
    darr_clipped = np.clip(log_arr, vmin, vmax)

    # Remove NaNs for histogram and flatten
    mask = ~da.isnan(darr_clipped) # should not be any nans for model data
    darr_flat = darr_clipped[mask].ravel()

    # build histogram
    hist_da, edges_da = np.histogram(darr_flat, bins=nbins, range=(vmin, vmax))

    pdf = hist_da / hist_da.sum()
    cdf = np.cumsum(pdf)

    return hist_da, edges_da, pdf, cdf