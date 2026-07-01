import numpy as np

def weighted_sample_on_grid(points_to_sample, bias, field, mask=None):
    """
    Draw weighted random samples (without replacement) from a grid.

    Probability is proportional to ``exp(bias * (value - min))`` over the valid
    (finite, unmasked) cells.

    Parameters
    ----------
    points_to_sample : int
        Number of grid points to sample.
    bias : float
        Power bias applied (in log space) to the field before normalization.
    field : np.ndarray
        Input field, e.g. the stitched ``(j, i)`` grid. Array-likes (xarray
        DataArrays) are coerced via ``np.asarray``.
    mask : np.ndarray of bool or None, optional
        Same shape as ``field``; cells where ``mask == False`` are excluded.

    Returns
    -------
    list of tuple
        Sampled indices as positional tuples in ``field`` order (``(j, i)``).
    """
    arr = np.asarray(field)

    valid = np.isfinite(arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    flat_valid = np.flatnonzero(valid.ravel())
    vals = arr.ravel()[flat_valid]

    weights = np.exp(bias * (vals - vals.min()))
    p = weights / weights.sum()

    choice = np.random.choice(flat_valid.size, size=points_to_sample, replace=False, p=p)
    coords = np.unravel_index(flat_valid[choice], arr.shape)

    return [tuple(int(c) for c in idx) for idx in zip(*coords)]

# The following is for sampling on a pdf and is not being used currently. Can probably be deleted. ---------------------------------
# NOTE keping this functionality for now in case we decide to change our sampling pattern.
# # x must = da.values
# def sample_linearly_on_pdf(x, points_to_sample, display):
#     # Calculate pdf
#     hist, edges, pdf, cdf = stats.compute_pdf(x, nbins=10, eps=0)
#
#     if (display):
#         plt.bar(edges[:-1], pdf, width=(edges[1] - edges[0]), align='edge')
#
#     # sample linearly along pdf 1000 points for a good spread
#     samples = linear_sample(1000, edges, 0, 100)
#
#     if display:
#         plt.hist(samples, bins=10, density=True)
#         plt.xlabel("value")
#         plt.ylabel("PDF")
#         plt.show()
#
#     samples = np.random.choice(samples, size=points_to_sample, replace=False)
#
#     if display:
#         plt.hist(samples, bins=10, density=True)
#         plt.xlabel("value")
#         plt.ylabel("PDF")
#         plt.show()
#
#     return samples
#
# # sample uniformly on pdf
# def inverse_transform_sample(edges, cdf, n_samples, rng=None):
#
#     if rng is None:
#         rng = np.random.default_rng()
#
#     # [0,1)
#     u = rng.random(n_samples)
#
#     # find bin indices (first index where cdf >= u)
#     bin_idxs = np.searchsorted(cdf, u, side="left")  # length n_samples, in [0, nbins-1]
#
#     # convert bin idx to a random value within the bin edges
#     left_edges = edges[bin_idxs]
#     right_edges = edges[bin_idxs + 1]
#
#     # sample uniformly inside bin
#     samples = left_edges + rng.random(n_samples) * (right_edges - left_edges)
#     return samples, bin_idxs
#
# # The following method samples on a pdf linearly, higher values are more likely to be sampled
# def linear_sample(N, edges, min_per_bin, linear_growth):
#     N_bins = len(edges) - 1
#
#     # create linear weights over bins
#     w = np.linspace(1, linear_growth, N_bins)
#     w = w / w.sum() # normalize weights
#
#     samples_per_bin = np.full(N_bins, min_per_bin)
#
#     # samples left after guaranteeing minimum
#     remaining = N - samples_per_bin.sum()
#     samples_per_bin += (remaining * w).astype(int)
#
#     samples = []
#     for i in range(N_bins):
#         lo, hi = edges[i], edges[i+1]
#         # sample uniformly inside this bin
#         s = np.random.uniform(lo, hi, samples_per_bin[i])
#         samples.append(s)
#
#     return np.concatenate(samples)
#
# # FIND SAMPLES IN GRID
# def first_match_coord(x, s, tol):
#     mask = np.abs(x - s) < tol
#     idx = np.argwhere(mask)
#     if len(idx) > 0:
#         return tuple(idx[0])  # first match
#     else:
#         return None # todo dont return none. Kill this match
#
# def find_coords_first_parallel(x, samples, tol=1e-10, n_jobs=-1):
#     print()
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(first_match_coord)(x, s, tol) for s in tqdm.tqdm(samples)
#     )
#     return results