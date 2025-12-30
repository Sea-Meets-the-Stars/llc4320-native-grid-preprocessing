import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tqdm
import xarray as xr
import data_preprocessing.stats as stats

def weighted_sample_on_grid(points_to_sample, bias, da, mask=None):
    """
    Draw weighted random samples from an LLC grid based on an input field.

    This function samples grid points with probability proportional to the
    values of an input xarray DataArray, optionally masked to exclude invalid
    regions. Higher field values result in higher sampling likelihood. Returned
    samples are given as native grid indices.

    This function takes a while.

    Parameters
    ----------
    points_to_sample : int
        Number of grid points to sample.
    bias : float or xarray.DataArray
        power bias applied to the input field before normalization.
        Can be a scalar or broadcastable to ``da``.
    da : xarray.DataArray
        Input field defined on the LLC grid (e.g., with dimensions
        ``(face, j, i)``) used to define sampling weights.
    mask : xarray.DataArray or None, optional
        Boolean mask with the same dimensions as ``da``. Points where
        ``mask == False`` are excluded from sampling. If None, no masking
        is applied.

    Returns
    -------
    indices : list of tuple
        List of sampled grid indices as tuples of integers corresponding
        to ``da.dims`` order (e.g., ``(face, j, i)``).

    Notes
    -----
    * Sampling is performed without replacement.
    * Masked or NaN values are excluded prior to normalization.
    * Coordinates are preserved through stacking to ensure correct index
      recovery.
    """

    if (mask is not None):
        da = da.where(mask) 

    # # make positive with lowest value 0
    # da = da - da.min()

    # weights = da ** bias # power law

    # exponential 
    weights = np.exp(bias * (da - da.min(skipna=True)))

    w_stacked = weights.stack(
        sample_dim=da.dims)  # stacks face j i into one dimension but keeps track of indexes

    w_valid = w_stacked.dropna("sample_dim")  # coordinates of xarray are preserved here
    p = w_valid / w_valid.sum()
    
    p.persist()
    
    # grab however many samples randomly with higher likelihood for high weights.
    choice = np.random.choice(
        p.sample_dim.size,
        size=points_to_sample,
        replace=False,
        p=p.values
    )

    sampled =  w_valid.isel(sample_dim=choice)

    indices = np.stack(
        [sampled.coords[dim].values for dim in da.dims],
        axis=1
    )

    indices = [tuple(row) for row in indices]  # list of tuples containing face j i into our grid
    indices = [tuple(int(x) for x in t) for t in indices]

    return indices

# The following is for sampling on a pdf and is not being used currently. Can probably be deleted. ---------------------------------

# x must = da.values
def sample_linearly_on_pdf(x, points_to_sample, display):
    # Calculate pdf
    hist, edges, pdf, cdf = stats.compute_pdf(x, nbins=10, eps=0)

    if (display):
        plt.bar(edges[:-1], pdf, width=(edges[1] - edges[0]), align='edge')

    # sample linearly along pdf 1000 points for a good spread
    samples = linear_sample(1000, edges, 0, 100)

    if display:
        plt.hist(samples, bins=10, density=True)
        plt.xlabel("value")
        plt.ylabel("PDF")
        plt.show()

    samples = np.random.choice(samples, size=points_to_sample, replace=False)

    if display:
        plt.hist(samples, bins=10, density=True)
        plt.xlabel("value")
        plt.ylabel("PDF")
        plt.show()

    return samples

# sample uniformly on pdf
def inverse_transform_sample(edges, cdf, n_samples, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # [0,1)
    u = rng.random(n_samples)

    # find bin indices (first index where cdf >= u)
    bin_idxs = np.searchsorted(cdf, u, side="left")  # length n_samples, in [0, nbins-1]

    # convert bin idx to a random value within the bin edges
    left_edges = edges[bin_idxs]
    right_edges = edges[bin_idxs + 1]

    # sample uniformly inside bin
    samples = left_edges + rng.random(n_samples) * (right_edges - left_edges)
    return samples, bin_idxs

# The following method samples on a pdf linearly, higher values are more likely to be sampled
def linear_sample(N, edges, min_per_bin, linear_growth):
    N_bins = len(edges) - 1

    # create linear weights over bins
    w = np.linspace(1, linear_growth, N_bins)
    w = w / w.sum() # normalize weights

    samples_per_bin = np.full(N_bins, min_per_bin)

    # samples left after guaranteeing minimum
    remaining = N - samples_per_bin.sum()
    samples_per_bin += (remaining * w).astype(int)

    samples = []
    for i in range(N_bins):
        lo, hi = edges[i], edges[i+1]
        # sample uniformly inside this bin
        s = np.random.uniform(lo, hi, samples_per_bin[i])
        samples.append(s)

    return np.concatenate(samples)

# FIND SAMPLES IN GRID
def first_match_coord(x, s, tol):
    mask = np.abs(x - s) < tol
    idx = np.argwhere(mask)
    if len(idx) > 0:
        return tuple(idx[0])  # first match
    else:
        return None # todo dont return none. Kill this match

def find_coords_first_parallel(x, samples, tol=1e-10, n_jobs=-1):
    print()
    results = Parallel(n_jobs=n_jobs)(
        delayed(first_match_coord)(x, s, tol) for s in tqdm.tqdm(samples)
    )
    return results