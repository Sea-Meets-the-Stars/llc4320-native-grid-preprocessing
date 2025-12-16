import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tqdm

import data_preprocessing.stats as stats

# SAMPLING ON NATIVE GRID ----------------------------------------

# Calculate a weight of all points on the grid based on input xarray
# Higher values of input xarray will be more liklely to be sampled
# Returns indices into grid with native coordinates
def weighted_sample_on_grid(points_to_sample, bias, da):

    weights = bias * da

    w_stacked = weights.stack(
        sample_dim=da.dims)  # stacks face j i into one dimension but keeps track of indexes

    w_valid = w_stacked.dropna("sample_dim")  # coordinates of xarray are preserved here
    p = w_valid / w_valid.sum()

    # grab however many samples randomly with higher likelihood for high weights.
    choice = np.random.choice(
        p.sample_dim.size,
        size=points_to_sample,
        replace=True,
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

# SAMPLING ON PDF ---------------------------------

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