import xarray as xr
import utils.jmd95_xgcm_implementation as jmd95

# returns merid** + zonal**
# expects dask arrays
def grad_squared(zonal_grad, merid_grad):
    zonal_grad.persist()
    merid_grad.persist()

    return zonal_grad ** 2 + merid_grad ** 2

def buoyancy_of_field(ds):
    #ds must have Theta and Salt

    g = 0.0098
    ref_rho: float = 1025.

    # chunk data
    ds = ds.chunk({'face': 1, 'j': 720, 'i': 720})
    p = xr.zeros_like(ds.Theta)  # surface pressure

    rho = xr.apply_ufunc(
        jmd95.jmd95,
        ds.Salt,
        ds.Theta,
        p,
        dask="parallelized",
        output_dtypes=[float],
    )

    rho = rho.persist()

    buoyancy = g * rho / ref_rho
    buoyancy = buoyancy.persist()

    return buoyancy