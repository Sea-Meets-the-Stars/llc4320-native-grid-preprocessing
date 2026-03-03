import numpy as np
import dask.array as da

def compute_pdf(log_arr, nbins=200, eps=1e-12, q_low=0.001, q_high=0.999):
    """Compute a histogram-based PDF and CDF from a log-transformed array.

    Clips the input to robust percentile bounds, removes NaNs, then
    builds a normalized histogram. Extreme outliers are squashed onto
    the bin edges, so expect spikes at the tails.

    Parameters
    ----------
    log_arr : array_like
        Input array (typically log10-transformed field values), already
        in memory.
    nbins : int, optional
        Number of histogram bins (default 200).
    eps : float, optional
        Reserved for future use (default 1e-12).
    q_low : float, optional
        Lower quantile for clipping (default 0.001, i.e. 0.1th percentile).
    q_high : float, optional
        Upper quantile for clipping (default 0.999, i.e. 99.9th percentile).

    Returns
    -------
    hist_da : numpy.ndarray
        Raw bin counts.
    edges_da : numpy.ndarray
        Bin edges (length ``nbins + 1``).
    pdf : numpy.ndarray
        Normalized probability density (sums to 1).
    cdf : numpy.ndarray
        Cumulative distribution function.
    """

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