import dbof.utils.physical_calculations as physical_calculations
import dbof.utils.native_gradient as ng
import dask.array as da


def log_grad_b(ds_merge, grid):
    """Compute log10 of the squared buoyancy gradient magnitude.

    Derives surface buoyancy from Theta and Salt, computes zonal and
    meridional gradients on the native LLC grid, squares and sums them,
    then returns the base-10 logarithm (i.e., log10(|grad b|^2)).

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Merged dataset containing 'Theta', 'Salt', grid metrics
        ('dxC', 'dyC'), and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    dask.array.Array
        log10 of the squared buoyancy gradient magnitude.
        Units are log10((km/s^2/m)^2)
    """
    buoyancy = physical_calculations.buoyancy_of_field(ds_merge)

    # gradient of b
    #  buoyancy m^-1
    zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_merge, grid=grid)

    gradb2 = physical_calculations.grad_squared(zonal_grad_b, merid_grad_b)

    log_gradb = da.log10(gradb2)

    return log_gradb

def relative_vorticity(ds_merge, grid):
    """
    Compute relative vorticity from horizontal velocity components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    omega : xarray.DataArray
        Relative vorticity field computed as dv/dλ minus du/dφ.
    """
    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi = (
        ng.calculate_jacobian(u_x, v_y, ds_merge, grid))

    omega = dv_phi_dlambda - du_lambda_dphi

    return omega