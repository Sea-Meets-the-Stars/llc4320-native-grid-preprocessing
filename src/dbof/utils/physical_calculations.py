import xarray as xr
import dbof.utils.jmd95_xgcm_implementation as jmd95

def buoyancy_of_field(ds):
    """LEGACY — superseded; do not use in new code.

    Superseded by ``calculate_fields.buoyancy_of_field`` (lazy,
    b = G rho / RHO0_REFERENCE).  This version persists intermediates
    into memory and uses g = 9.8, rho_ref = 1025; kept only for
    reference during the field migration (prompts/field_migration.md).
    Nothing in src/ calls it anymore.

    Compute surface buoyancy from potential temperature and salinity.

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
        Surface buoyancy field [km/s^2] with dask
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

def density_of_field(ds):
    """LEGACY — superseded; do not use in new code.

    Superseded by ``calculate_fields.potential_density`` (lazy).
    This version persists the result into memory.  Note the output is
    surface-referenced POTENTIAL density (JMD95 at p=0), despite the
    historical wording below.  Kept only for reference during the field
    migration (prompts/field_migration.md).  Nothing in src/ calls it.

    Compute surface density from potential temperature and salinity.

    Uses the JMD95 equation of state to compute in-situ density at the
    surface (p=0).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'Theta' (potential temperature in deg C) and
        'Salt' (salinity in PSU) variables with dimensions (face, j, i).

    Returns
    -------
    xarray.DataArray
        Surface density field [kg/m^3] with dask
        backing, persisted into memory.
    """
    
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

    return rho