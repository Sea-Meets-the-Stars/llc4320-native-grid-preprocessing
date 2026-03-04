import tqdm
import numpy as np

def estimate_xarray_size(ds):
    """Estimate the in-memory size of an xarray Dataset.

    Iterates over all coordinates and data variables, computing the total
    number of bytes based on array shapes and dtypes. Useful for gauging
    download or compute costs before calling ``.compute()`` on lazy data.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset whose memory footprint is to be estimated.

    Returns
    -------
    int
        Estimated total size in bytes.
    """
    total_bytes = 0
    for name, var in tqdm(ds.coords.items(), desc="Estimating memory"):
        if all(dim in ds.dims for dim in var.dims):
            shape = [ds.dims[d] for d in var.dims]
            dtype_size = np.dtype(var.dtype).itemsize
            total_bytes += np.prod(shape) * dtype_size
    for name, var in tqdm(ds.data_vars.items(), desc="Estimating memory"):
        if all(dim in ds.dims for dim in var.dims):
            shape = [ds.dims[d] for d in var.dims]
            dtype_size = np.dtype(var.dtype).itemsize
            total_bytes += np.prod(shape) * dtype_size
    return total_bytes



# with ProgressBar():
#     co_compute = co_to_load.compute()
# print(co_compute)


# size_bytes = estimate_xarray_size(co)
# print(f"Estimated size grid: {size_bytes/1e9:.2f} GB")
#
# size_bytes = estimate_xarray_size(ds)
# print(f"Estimated size data: {size_bytes/1e9:.2f} GB")