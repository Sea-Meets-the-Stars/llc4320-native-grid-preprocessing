import tqdm
import numpy as np

# this function will give us an idea of download times
def estimate_xarray_size(ds):
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