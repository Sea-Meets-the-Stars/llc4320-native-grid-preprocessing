import xarray as xr
import dbof.utils.jmd95_xgcm_implementation as jmd95

def grad_squared(zonal_grad, merid_grad):
    """Compute the squared magnitude of a 2D gradient vector.

    Parameters
    ----------
    zonal_grad : xarray.DataArray or dask array
        Zonal (east-west) component of the gradient.
    merid_grad : xarray.DataArray or dask array
        Meridional (north-south) component of the gradient.

    Returns
    -------
    xarray.DataArray or dask array
        Sum of the squared zonal and meridional gradient components.
    """
    return zonal_grad ** 2 + merid_grad ** 2

def buoyancy_of_field(ds):
    """Compute surface buoyancy from potential temperature and salinity.

    Uses the JMD95 equation of state to compute in-situ density at the
    surface (p=0), then converts to buoyancy as b = g * rho / rho_ref.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'Theta' (potential temperature in deg C) and
        'Salt' (salinity in PSU) variables with dimensions (face, j, i).

    Returns
    -------
    xarray.DataArray
        Surface buoyancy field [km/s^2 * kg/m^3 / (kg/m^3)] with dask
        backing, persisted into memory.
    """
    g = 0.0098 # km/s^2
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