import numpy as np
import dbof.utils.physical_calculations as physical_calculations
import dbof.utils.native_gradient as ng
import dask.array as da


def grad_b2(ds_merge, grid):
    """Compute the squared buoyancy gradient magnitude.

    Derives surface buoyancy from Theta and Salt, computes zonal and
    meridional gradients on the native LLC grid, squares and sums them

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
        squared buoyancy gradient magnitude.
        Units are s^4
    """
    buoyancy = physical_calculations.buoyancy_of_field(ds_merge)*1e3

    zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_merge, grid=grid)

    gradb2 = physical_calculations.grad_squared(zonal_grad_b, merid_grad_b)

    return gradb2

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
        Units are log10((s^4)
    """
    # Gradient of buoyancy^2
    gradb2 = grad_b2(ds_merge, grid)
    # Take the log10 of the squared buoyancy gradient magnitude
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


def coriolis_parameter(ds_merge, grid):
    """
    Compute coriolis parameter from latitude.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    coriolis : xarray.DataArray
        Coriolis parameter field.
    """

    omega = 7.292115e-5  # rad/s ; Earth's rotation rate
    coriolis_f = 2.0 * omega * np.sin(np.deg2rad(ds_merge['YC']))

    return coriolis_f


def rossby_number(ds_merge, grid):
    """
    Compute Rossby number.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    rossby_no : xarray.DataArray
        Rossby number field.
    """

    rossby_no = relative_vorticity(ds_merge, grid) / coriolis_parameter(ds_merge, grid)

    return rossby_no

def strain(ds_merge, grid):
    """
    Compute strain from horizontal velocity components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    strain_mag : xarray.DataArray
        Strain magnitude field.
    strain_n: xarray.DataArray
        Normal strain field.
    strain_s: xarray.DataArray
        Shear strain field.
    """

    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi = (
        ng.calculate_jacobian(u_x, v_y, ds_merge, grid))

    strain_n = du_lambda_dlambda - dv_phi_dphi
    strain_s = du_lambda_dphi + dv_phi_dlambda  
    strain_mag = np.sqrt(strain_n**2 + strain_s**2)

    return strain_mag, strain_n, strain_s


def divergence(ds_merge, grid):
    """
    Compute horizontal divergence from horizontal velocity components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    divergence : xarray.DataArray
        Horizontal divergence field.
    """

    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi = (
        ng.calculate_jacobian(u_x, v_y, ds_merge, grid))

    divergence = du_lambda_dlambda + dv_phi_dphi

    return divergence


def okubo_weiss_parameter(ds_merge, grid):
    """
    Compute the Okubo-Weiss parameter.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    OW : xarray.DataArray
        Okubo-Weiss parameter field.
    """

    vorticity = relative_vorticity(ds_merge, grid)
    _, strain_n, strain_s = strain(ds_merge, grid)

    okubo_weiss =  strain_n**2 + strain_s**2 - vorticity**2

    return okubo_weiss


def all_velocity_properties(ds_merge, grid):
    """
    Compute all velocity-derived properties from a single Jacobian pass.

    Computes the Jacobian once and derives relative vorticity, strain (normal,
    shear, magnitude), divergence, Coriolis parameter, Rossby number, and the
    Okubo-Weiss parameter from the same four gradient components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid, grid metrics,
        rotation coefficients ('CS', 'SN'), and latitude ('YC').
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    dict of str -> xarray.DataArray
        Keys: 'relative_vorticity', 'strain_n', 'strain_s', 'strain_mag',
              'divergence', 'coriolis_f', 'rossby_number', 'okubo_weiss'
    """
    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi = (
        ng.calculate_jacobian(u_x, v_y, ds_merge, grid))

    omega     = dv_phi_dlambda - du_lambda_dphi
    strain_n  = du_lambda_dlambda - dv_phi_dphi
    strain_s  = du_lambda_dphi + dv_phi_dlambda
    strain_mag = np.sqrt(strain_n**2 + strain_s**2)
    divergence = du_lambda_dlambda + dv_phi_dphi

    omega_earth = 7.292115e-5  # rad/s ; Earth's rotation rate
    coriolis_f  = 2.0 * omega_earth * np.sin(np.deg2rad(ds_merge['YC']))
    rossby_no   = omega / coriolis_f
    okubo_weiss = strain_n**2 + strain_s**2 - omega**2

    return {
        'relative_vorticity': omega,
        'strain_n':           strain_n,
        'strain_s':           strain_s,
        'strain_mag':         strain_mag,
        'divergence':         divergence,
        'coriolis_f':         coriolis_f,
        'rossby_number':      rossby_no,
        'okubo_weiss':        okubo_weiss,
    }