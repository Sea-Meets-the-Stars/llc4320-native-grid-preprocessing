import numpy as np
import dbof.utils.physical_calculations as physical_calculations
import dbof.utils.native_gradient as ng
import dask.array as da


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
        Surface density field [km/s^2] with dask
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
        squared temperature gradient magnitude.
        Units are (K/m)^2
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

    ALPHA = 2.0e-4   # thermal expansion coefficient  (°C⁻¹)
    BETA  = 7.4e-4   # haline contraction coefficient (PSU⁻¹)
    RHO0  = 1025.0   # reference density (kg m⁻³)

    if gradtheta2 is None:
        gradtheta2 = grad_theta2(ds_merge, grid)
    if gradsalt2 is None:
        gradsalt2 = grad_salt2(ds_merge, grid)
    if gradrho2 is None:
        gradrho2 = grad_rho2(ds_merge, grid)

    numer      = RHO0 * (BETA**2 * gradsalt2 - ALPHA**2 * gradtheta2)
    denom      = np.where(gradrho2 > 0, -gradrho2 / RHO0, np.nan) # Mask pixels where |∇ρ| = 0 to avoid divide-by-zero

    tu_rad = np.arctan(numer / denom)
    tu_h   = np.degrees(tu_rad)

    return tu_h

# ------------------------------------------------------------------------------------
# --------------------------------- KINEMATIC ----------------------------------------
# ------------------------------------------------------------------------------------

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

    Note: the arithmetic in this function intentionally duplicates that in the
    individual functions (relative_vorticity, strain, etc.) to ensure the
    Jacobian is computed exactly once across all properties. Do not refactor
    to call those individual functions here.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U, V on the model grid, grid metrics, rotation
        coefficients ('CS', 'SN'), latitude ('YC').
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    dict of str -> xarray.DataArray
        Keys: 'relative_vorticity', 'strain_n', 'strain_s', 'strain_mag',
              'divergence', 'coriolis_f', 'rossby_number', 'okubo_weiss',
    """
    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi = (
        ng.calculate_jacobian(u_x, v_y, ds_merge, grid))

    omega      = dv_phi_dlambda - du_lambda_dphi
    strain_n   = du_lambda_dlambda - dv_phi_dphi
    strain_s   = du_lambda_dphi + dv_phi_dlambda
    strain_mag = np.sqrt(strain_n**2 + strain_s**2)
    divergence = du_lambda_dlambda + dv_phi_dphi

    omega_earth = 7.292115e-5  # rad/s ; Earth's rotation rate
    coriolis_f  = 2.0 * omega_earth * np.sin(np.deg2rad(ds_merge['YC']))
    rossby_no   = omega / coriolis_f
    okubo_weiss = strain_n**2 + strain_s**2 - omega**2

    return {
        'relative_vorticity':     omega,
        'strain_n':               strain_n,
        'strain_s':               strain_s,
        'strain_mag':             strain_mag,
        'divergence':             divergence,
        'coriolis_f':             coriolis_f,
        'rossby_number':          rossby_no,
        'okubo_weiss':            okubo_weiss,
    }

# ------------------------------------------------------------------------------------
# ------------------------------ FRONTOGENESIS ---------------------------------------
# ------------------------------------------------------------------------------------ 

def _frontogenesis_tendency(du_dx, du_dy, dv_dx, dv_dy, grad_bx, grad_by):
    """Kinematic frontogenesis tendency from velocity gradient components.

    F = -(du/dx * bx² + (du/dy + dv/dx) * bx*by + dv/dy * by²)

    Used internally by all_velocity_properties to compute both the full and
    geostrophic frontogenesis without duplicating the formula.

    Parameters
    ----------
    du_dx, du_dy : xarray.DataArray  — ∂u/∂x, ∂u/∂y
    dv_dx, dv_dy : xarray.DataArray  — ∂v/∂x, ∂v/∂y
    grad_bx, grad_by : xarray.DataArray  — ∂b/∂x, ∂b/∂y
    """
    return -(du_dx * grad_bx**2 +
             (du_dy + dv_dx) * grad_bx * grad_by +
             dv_dy * grad_by**2)


#TODO: Lauren edit this, perhaps break it up and move the aggregate part upstream
def all_frontogenesis_properties(ds_merge, grid):
    """
    Compute all frontogenesis properties from a single Jacobian pass.

    Computes the Jacobian once and derives the kinematic frontogenesis tendency from the
    same four gradient components. Frontogenesis additionally requires the
    buoyancy gradient (from Theta and Salt), but shares the velocity Jacobian
    rather than recomputing it.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing U, V on the model grid, grid metrics, rotation
        coefficients ('CS', 'SN'), latitude ('YC'), and tracer fields
        ('Theta', 'Salt') required for frontogenesis.
    grid : xgcm.Grid
        Grid object relating to ds_merge.

    Returns
    -------
    dict of str -> xarray.DataArray
        Keys: 'frontogenesis_tendency', 'ug', 'vg',
              'frontogenesis_geo', 'frontogenesis_ageo'
    """
    u_x = ds_merge.U.copy(deep=True)
    v_y = ds_merge.V.copy(deep=True)

    du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi = (
        ng.calculate_jacobian(u_x, v_y, ds_merge, grid))

    omega_earth = 7.292115e-5  # rad/s ; Earth's rotation rate
    coriolis_f  = 2.0 * omega_earth * np.sin(np.deg2rad(ds_merge['YC']))


    # Buoyancy gradients — shared by full and geostrophic frontogenesis.
    buoyancy = physical_calculations.buoyancy_of_field(ds_merge) * 1e3
    zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_merge, grid=grid)

    # Full frontogenesis tendency — uses the velocity Jacobian computed above.
    # strain_s (= du_lambda_dphi + dv_phi_dlambda) is reused as the cross term.
    frontogenesis_tendency = _frontogenesis_tendency(
        du_lambda_dlambda, du_lambda_dphi,
        dv_phi_dlambda,    dv_phi_dphi,
        zonal_grad_b,      merid_grad_b,
    )

    # Geostrophic velocity from sea surface height gradient.
    # Eta is a scalar tracer (cell-centre), so its gradient uses
    # calculate_native_gradient_tracer (not the Jacobian).
    # ug = -(g/f) * deta/dy,  vg = (g/f) * deta/dx
    # coriolis_f is reused from above. Near the equator f → 0, so ug/vg → inf/NaN
    # in a narrow equatorial band — physically correct, mask downstream if needed.
    g = 9.81  # m/s²; gravitational acceleration
    zonal_grad_eta, merid_grad_eta = ng.calculate_native_gradient_tracer(
        ds_merge['Eta'], ds_merge, grid=grid)
    ug = -(g / coriolis_f) * merid_grad_eta
    vg =  (g / coriolis_f) * zonal_grad_eta

    # Geostrophic frontogenesis — ug/vg are at tracer points, so their
    # gradients are computed as tracer gradients (not via calculate_jacobian,
    # which expects staggered U/V). Buoyancy gradients are reused from above.
    zonal_grad_ug, merid_grad_ug = ng.calculate_native_gradient_tracer(ug, ds_merge, grid=grid)
    zonal_grad_vg, merid_grad_vg = ng.calculate_native_gradient_tracer(vg, ds_merge, grid=grid)
    frontogenesis_geo = _frontogenesis_tendency(
        zonal_grad_ug, merid_grad_ug,
        zonal_grad_vg, merid_grad_vg,
        zonal_grad_b,  merid_grad_b,
    )

    # Ageostrophic frontogenesis — residual of full minus geostrophic.
    # Note: this is F(u,v) - F(ug,vg), which includes both the purely
    # ageostrophic term F(u_ageo, v_ageo) and geostrophic/ageostrophic
    # cross terms. Treat as a qualitative measure of ageostrophic influence.
    frontogenesis_ageo = frontogenesis_tendency - frontogenesis_geo

    return {
        'frontogenesis_tendency': frontogenesis_tendency,
        'frontogenesis_geo':      frontogenesis_geo,
        'frontogenesis_ageo':     frontogenesis_ageo,
        'ug':                     ug,
        'vg':                     vg,
    }