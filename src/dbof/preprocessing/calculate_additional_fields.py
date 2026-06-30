from collections import namedtuple

import numpy as np
import dask.array as da

import dbof.utils.physical_calculations as physical_calculations
import dbof.utils.native_gradient as ng
from dbof.preprocessing.physical_constants import (
    G,
    OMEGA_EARTH,
    ALPHA,
    BETA,
    RHO0_SEAWATER,
)


# ---------------------------------------------------------------------------
# Shared intermediate: velocity Jacobian
# ---------------------------------------------------------------------------

VelocityJacobian = namedtuple(
    'VelocityJacobian', ['du_dx', 'du_dy', 'dv_dx', 'dv_dy'],
)
"""The four components of the 2×2 velocity gradient tensor.

Computing the Jacobian is the single most expensive step shared by the
kinematic and frontogenesis subsets.  Individual property functions accept
an optional ``jacobian`` keyword so that a caller can compute it once and
pass it through, while still working standalone (computing their own
Jacobian when ``jacobian is None``).
"""

BuoyancyGradients = namedtuple(
    'BuoyancyGradients', ['zonal', 'merid'],
)
"""Zonal and meridional components of the surface buoyancy gradient.

Shared by ``frontogenesis_tendency`` and ``frontogenesis_geo``.  Individual
functions accept an optional ``buoyancy_gradients`` keyword so that a
caller can compute the gradients once and pass them through.
"""


def compute_velocity_jacobian(ds_merge, grid):
    """Compute the velocity Jacobian from U and V on the native LLC grid.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V on the staggered model grid, plus
        grid metrics ('dxC', 'dyC') and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    VelocityJacobian
        Named tuple ``(du_dx, du_dy, dv_dx, dv_dy)`` with each component
        as a dask-backed DataArray on tracer points [s^-1].
    """
    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_dx, du_dy, dv_dx, dv_dy = ng.calculate_jacobian(
        u_x, v_y, ds_merge, grid,
    )
    return VelocityJacobian(du_dx, du_dy, dv_dx, dv_dy)


# ---------------------------------------------------------------------------
# Native vector fields (geographic, tracer-collocated)
# ---------------------------------------------------------------------------

def geographic_velocity(ds_merge, grid):
    """Horizontal velocity rotated to geographic (east/north) components.

    The native ``U``/``V`` are stored on the staggered model grid and on
    model-relative axes; this returns them interpolated to tracer points and
    rotated to true east/north via the grid ``CS``/``SN`` coefficients.  Use
    when saving / analysing raw velocity for a chunk or tile, where the global
    face-stitching that would otherwise perform this rotation is not run.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing ``U``, ``V`` and rotation coefficients ``CS``/``SN``.
    grid : xgcm.Grid
        Grid used to interpolate the staggered components to tracer points.

    Returns
    -------
    u_east, v_north : xarray.DataArray
        Eastward / northward velocity [m s⁻¹] on tracer points.
    """
    u_east, v_north = ng.rotate_vector_to_geographic(
        ds_merge.U, ds_merge.V, ds_merge, grid,
    )
    u_east.name = "u_east"
    u_east.attrs.update({"long_name": "eastward velocity", "units": "m s-1"})
    v_north.name = "v_north"
    v_north.attrs.update({"long_name": "northward velocity", "units": "m s-1"})
    return u_east, v_north


def geographic_wind_stress(ds_merge, grid):
    """Wind stress rotated to geographic (east/north) components on tracer points.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing ``oceTAUX``, ``oceTAUY`` and ``CS``/``SN``.
    grid : xgcm.Grid
        Grid used to interpolate the staggered components to tracer points.

    Returns
    -------
    tau_east, tau_north : xarray.DataArray
        Eastward / northward wind stress [N m⁻²] on tracer points.
    """
    tau_east, tau_north = ng.rotate_vector_to_geographic(
        ds_merge.oceTAUX, ds_merge.oceTAUY, ds_merge, grid,
    )
    tau_east.name = "tau_east"
    tau_east.attrs.update({"long_name": "eastward wind stress", "units": "N m-2"})
    tau_north.name = "tau_north"
    tau_north.attrs.update({"long_name": "northward wind stress", "units": "N m-2"})
    return tau_east, tau_north


def compute_buoyancy_gradients(ds_merge, grid):
    """Compute zonal and meridional buoyancy gradients on the native LLC grid.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing 'Theta', 'Salt', grid metrics ('dxC', 'dyC'),
        and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    BuoyancyGradients
        Named tuple ``(zonal, merid)`` with each component as a
        dask-backed DataArray on tracer points [s -2].
    """
    buoyancy = physical_calculations.buoyancy_of_field(ds_merge) * 1e3
    zonal, merid = ng.calculate_native_gradient_tracer(
        buoyancy, ds_merge, grid=grid,
    )
    return BuoyancyGradients(zonal, merid)


# ------------------------------------------------------------------------------------
# ---------------------------- FRONTAL STRUCTURE -------------------------------------
# ------------------------------------------------------------------------------------ 

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
        Units are s^-4
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
        Units are log10((s^-4)
    """
    # Gradient of buoyancy^2
    gradb2 = grad_b2(ds_merge, grid)
    # Take the log10 of the squared buoyancy gradient magnitude
    log_gradb = da.log10(gradb2)

    return log_gradb

def grad_rho2(ds_merge, grid):
    """Compute squared surface density gradient from potential temperature and salinity.

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
        Spatial gradient of surface density field [(kg m-4)^2] with dask
        backing, persisted into memory.
    """

    rho = physical_calculations.density_of_field(ds_merge)
    
    zonal_grad_rho, merid_grad_rho = ng.calculate_native_gradient_tracer(rho, ds_merge, grid=grid)

    gradrho2 = physical_calculations.grad_squared(zonal_grad_rho, merid_grad_rho)

    return gradrho2


def grad_theta2(ds_merge, grid):
    """Compute the squared temperature gradient magnitude.

    Computes zonal and meridional gradients on the native LLC grid, 
    squares and sums them

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Merged dataset containing 'Theta', grid metrics
        ('dxC', 'dyC'), and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    dask.array.Array
        Spatial gradient of squared temperature magnitude.
        Units are (degrees C/m)^2
    """
    theta = ds_merge.Theta.copy(deep=True)

    zonal_grad_theta, merid_grad_theta = ng.calculate_native_gradient_tracer(theta, ds_merge, grid=grid)

    gradtheta2 = physical_calculations.grad_squared(zonal_grad_theta, merid_grad_theta)

    return gradtheta2

def grad_salt2(ds_merge, grid):
    """Compute the squared salinity gradient magnitude.

    Computes zonal and meridional gradients on the native LLC grid, 
    squares and sums them

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Merged dataset containing 'Theta', grid metrics
        ('dxC', 'dyC'), and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    dask.array.Array
        squared salinity gradient magnitude.
        Units are (psu/m)^2
    """
    salt = ds_merge.Salt.copy(deep=True)

    zonal_grad_salt, merid_grad_salt = ng.calculate_native_gradient_tracer(salt, ds_merge, grid=grid)

    gradsalt2 = physical_calculations.grad_squared(zonal_grad_salt, merid_grad_salt)
    return gradsalt2

def grad_eta2(ds_merge, grid):
    """Compute the squared SSH gradient magnitude.

    Computes zonal and meridional gradients on the native LLC grid, 
    squares and sums them

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Merged dataset containing 'Eta', grid metrics
        ('dxC', 'dyC'), and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    dask.array.Array
        squared SSH gradient magnitude.
        Units are (m/m)^2
    """
    eta = ds_merge.Eta.copy(deep=True)

    zonal_grad_eta, merid_grad_eta = ng.calculate_native_gradient_tracer(eta, ds_merge, grid=grid)
    gradeta2 = physical_calculations.grad_squared(zonal_grad_eta, merid_grad_eta)
    return gradeta2

def turner_angle(ds_merge, grid, *, gradtheta2=None, gradsalt2=None, gradrho2=None):
    """Compute the horizontal Turner Angle.

    Tu_h = arctan( ∇ρ·(α∇T + β∇S) / ∇ρ·(α∇T - β∇S) )

        Linear EOS  ∇ρ = ρ₀(−α∇T + β∇S) -->

        Numerator   = ∇ρ·(α∇T + β∇S) = ρ₀(β²|∇S|² - α²|∇T|²)
                 [T·S cross terms cancel exactly]

        Denominator = ∇ρ·(α∇T - β∇S) = -|∇ρ|²/ρ₀
                 [follows from |∇ρ|² = ρ₀²(α²|∇T|² - 2αβ∇T·∇S + β²|∇S|²)]

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Merged dataset containing 'Theta', 'Salt', grid metrics
        ('dxC', 'dyC'), and rotation coefficients ('CS', 'SN').
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    gradtheta2 : array-like, optional
        Pre-computed |∇θ|².  Computed from *ds_merge* when *None*.
    gradsalt2 : array-like, optional
        Pre-computed |∇S|².  Computed from *ds_merge* when *None*.
    gradrho2 : array-like, optional
        Pre-computed |∇ρ|².  Computed from *ds_merge* when *None*.

    Returns
    -------
    dask.array.Array
        Turner Angle.
        Units are degrees.
    """

    if gradtheta2 is None:
        gradtheta2 = grad_theta2(ds_merge, grid)
    if gradsalt2 is None:
        gradsalt2 = grad_salt2(ds_merge, grid)
    if gradrho2 is None:
        gradrho2 = grad_rho2(ds_merge, grid)

    numer      = RHO0_SEAWATER * (BETA**2 * gradsalt2 - ALPHA**2 * gradtheta2)
    denom      = np.where(gradrho2 > 0, -gradrho2 / RHO0_SEAWATER, np.nan) # Mask pixels where |∇ρ| = 0 to avoid divide-by-zero

    tu_rad = np.arctan(numer / denom)
    tu_h   = np.degrees(tu_rad)

    return tu_h

# ------------------------------------------------------------------------------------
# --------------------------------- KINEMATIC ----------------------------------------
# ------------------------------------------------------------------------------------

def relative_vorticity(ds_merge, grid, *, jacobian=None):
    """
    Compute relative vorticity from horizontal velocity components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  When *None* the Jacobian is
        computed internally (standalone mode).

    Returns
    -------
    omega : xarray.DataArray
        Relative vorticity field computed as dv/dλ minus du/dφ [1/s].
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian(ds_merge, grid)

    omega = jacobian.dv_dx - jacobian.du_dy

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
        Coriolis parameter field [s-1].
    """

    coriolis_f = 2.0 * OMEGA_EARTH * np.sin(np.deg2rad(ds_merge['YC']))

    return coriolis_f


def rossby_number(ds_merge, grid, *, jacobian=None):
    """
    Compute Rossby number.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  Forwarded to
        ``relative_vorticity``; when *None* the Jacobian is computed
        internally.

    Returns
    -------
    rossby_no : xarray.DataArray
        Rossby number field [dimensionless].
    """
    omega = relative_vorticity(ds_merge, grid, jacobian=jacobian)
    f = coriolis_parameter(ds_merge, grid)
    rossby_no = omega / f

    return rossby_no

def strain(ds_merge, grid, *, jacobian=None):
    """
    Compute strain from horizontal velocity components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  When *None* the Jacobian is
        computed internally (standalone mode).

    Returns
    -------
    strain_mag : xarray.DataArray
        Strain magnitude field [s^-1].
    strain_n: xarray.DataArray
        Normal strain field [s^-1].
    strain_s: xarray.DataArray
        Shear strain field [s^-1].
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian(ds_merge, grid)

    strain_n = jacobian.du_dx - jacobian.dv_dy
    strain_s = jacobian.du_dy + jacobian.dv_dx
    strain_mag = np.sqrt(strain_n**2 + strain_s**2)

    return strain_mag, strain_n, strain_s


def divergence(ds_merge, grid, *, jacobian=None):
    """
    Compute horizontal divergence from horizontal velocity components.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  When *None* the Jacobian is
        computed internally (standalone mode).

    Returns
    -------
    div : xarray.DataArray
        Horizontal divergence field [s^-1].
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian(ds_merge, grid)

    div = jacobian.du_dx + jacobian.dv_dy

    return div


def okubo_weiss_parameter(ds_merge, grid, *, jacobian=None):
    """
    Compute the Okubo-Weiss parameter.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U and V components on the model grid.
    grid : xgcm.Grid
        Grid object relating to ds_merge.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  Forwarded to
        ``relative_vorticity`` and ``strain``; when *None* the Jacobian
        is computed internally (once, then shared).

    Returns
    -------
    OW : xarray.DataArray
        Okubo-Weiss parameter field [s^-2].
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian(ds_merge, grid)

    omega = relative_vorticity(ds_merge, grid, jacobian=jacobian)
    _, strain_n, strain_s = strain(ds_merge, grid, jacobian=jacobian)

    okubo_weiss = strain_n**2 + strain_s**2 - omega**2

    return okubo_weiss

# ------------------------------------------------------------------------------------
# ------------------------------ FRONTOGENESIS ---------------------------------------
# ------------------------------------------------------------------------------------ 

def _frontogenesis_formula(du_dx, du_dy, dv_dx, dv_dy, grad_bx, grad_by):
    """Kinematic frontogenesis tendency from velocity gradient components.

    F = -(du/dx * bx² + (du/dy + dv/dx) * bx*by + dv/dy * by²)

    Used internally by ``frontogenesis_tendency`` and ``frontogenesis_geo``
    to avoid duplicating the formula.

    Parameters
    ----------
    du_dx, du_dy : xr.DataArray
        Velocity gradient components ∂u/∂x and ∂u/∂y (s⁻¹).
    dv_dx, dv_dy : xr.DataArray
        Velocity gradient components ∂v/∂x and ∂v/∂y (s⁻¹).
    grad_bx, grad_by : xr.DataArray
        Buoyancy gradient components ∂b/∂x and ∂b/∂y (m s⁻² m⁻¹).

    Returns
    -------
    xr.DataArray
        Frontogenesis tendency F [s^-5], dask-backed.
    """
    return -(du_dx * grad_bx**2 +
             (du_dy + dv_dx) * grad_bx * grad_by +
             dv_dy * grad_by**2)


def frontogenesis_tendency(ds_merge, grid, *, jacobian=None,
                           buoyancy_gradients=None):
    """Kinematic frontogenesis tendency F(u, v).

    F = -(du/dx · bx² + (du/dy + dv/dx) · bx·by + dv/dy · by²)

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U, V, Theta, Salt, grid metrics, and rotation
        coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    jacobian : VelocityJacobian, optional
        Pre-computed velocity Jacobian.  Computed internally when *None*.
    buoyancy_gradients : BuoyancyGradients, optional
        Pre-computed buoyancy gradient components.  Computed internally
        when *None*.

    Returns
    -------
    xarray.DataArray
        Frontogenesis tendency [s^-5] (dask-backed, lazy).
    """
    if jacobian is None:
        jacobian = compute_velocity_jacobian(ds_merge, grid)
    if buoyancy_gradients is None:
        buoyancy_gradients = compute_buoyancy_gradients(ds_merge, grid)

    return _frontogenesis_formula(
        jacobian.du_dx, jacobian.du_dy,
        jacobian.dv_dx, jacobian.dv_dy,
        buoyancy_gradients.zonal, buoyancy_gradients.merid,
    )


def geostrophic_velocity(ds_merge, grid):
    """Geostrophic velocity from sea-surface height gradient.

    ug = -(g/f) · ∂η/∂y,  vg = (g/f) · ∂η/∂x

    Eta is a scalar tracer (cell-centre), so its gradient uses
    ``calculate_native_gradient_tracer`` (not the Jacobian, which expects
    staggered U/V).  Near the equator f → 0, so ug/vg → inf/NaN in a
    narrow equatorial band — physically correct; mask downstream if needed.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing 'Eta', latitude 'YC', grid metrics, and
        rotation coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.

    Returns
    -------
    ug, vg : xarray.DataArray
        Zonal and meridional geostrophic velocity [m s^-1] (dask-backed, lazy).
    """
    f = coriolis_parameter(ds_merge, grid)

    zonal_grad_eta, merid_grad_eta = ng.calculate_native_gradient_tracer(
        ds_merge['Eta'], ds_merge, grid=grid,
    )
    ug = -(G / f) * merid_grad_eta
    vg =  (G / f) * zonal_grad_eta
    return ug, vg


def frontogenesis_geo(ds_merge, grid, *, ug=None, vg=None,
                      buoyancy_gradients=None):
    """Geostrophic frontogenesis tendency F(ug, vg).

    Same formula as ``frontogenesis_tendency`` but evaluated with the
    geostrophic velocity field.  ug/vg live at tracer points, so their
    gradients are computed as tracer gradients (not via
    ``calculate_jacobian``, which expects staggered U/V).

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing Eta, Theta, Salt, latitude, grid metrics, and
        rotation coefficients.
    grid : xgcm.Grid
        Grid object used for differencing and interpolation.
    ug, vg : xarray.DataArray, optional
        Pre-computed geostrophic velocity components.  Computed internally
        when *None*.
    buoyancy_gradients : BuoyancyGradients, optional
        Pre-computed buoyancy gradient components.  Computed internally
        when *None*.

    Returns
    -------
    xarray.DataArray
        Geostrophic frontogenesis tendency (dask-backed, lazy).
    """
    if ug is None or vg is None:
        ug, vg = geostrophic_velocity(ds_merge, grid)
    if buoyancy_gradients is None:
        buoyancy_gradients = compute_buoyancy_gradients(ds_merge, grid)

    dug_dx, dug_dy = ng.calculate_native_gradient_tracer(
        ug, ds_merge, grid=grid,
    )
    dvg_dx, dvg_dy = ng.calculate_native_gradient_tracer(
        vg, ds_merge, grid=grid,
    )
    return _frontogenesis_formula(
        dug_dx, dug_dy,
        dvg_dx, dvg_dy,
        buoyancy_gradients.zonal, buoyancy_gradients.merid,
    )