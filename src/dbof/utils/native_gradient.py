"""
native_gradient.py
------------------
Horizontal differencing on the native LLC C-grid — the single home
of the gradient machinery.  Two stencil families: vector COMPONENTS
(ECCO recipe: diff -> interp to centre -> CS/SN rotate) and squared
MAGNITUDES (square/multiply on the staggered points BEFORE moving).

Geometry, basis/rotation rules, and usage limits live in
docs/Gradients.md.
"""

def rotate_vector_to_geographic(u_x, v_y, ds_merge, grid, *, interpolate=True):
    """Rotate a model-grid vector into geographic (east/north)
    components: interp to tracer points, then u_east = u·CS − v·SN,
    v_north = u·SN + v·CS.  See docs/Gradients.md.

    Applications
    ------------
    Signed VECTOR fields (U/V, wind stress, gradients-as-vectors).

    Parameters
    ----------
    u_x : xarray.DataArray
        Model x-direction component (e.g. ``U``, ``oceTAUX``).  On the staggered
        point when ``interpolate=True``; already at tracer points otherwise.
    v_y : xarray.DataArray
        Model y-direction component (e.g. ``V``, ``oceTAUY``).
    ds_merge : xarray.Dataset
        Dataset providing the rotation coefficients ``CS`` and ``SN``.
    grid : xgcm.Grid
        Grid used to interpolate staggered components to tracer points.
    interpolate : bool, default True
        If ``True``, interpolate the components to tracer points before
        rotating.  Set ``False`` when the inputs are already cell-centred.

    Returns
    -------
    u_east : xarray.DataArray
        Eastward (zonal) component on tracer points.
    v_north : xarray.DataArray
        Northward (meridional) component on tracer points.
    """
    if interpolate:
        u_x = grid.interp(u_x, 'X', boundary='fill')
        v_y = grid.interp(v_y, 'Y', boundary='fill')

    u_east = u_x * ds_merge['CS'] - v_y * ds_merge['SN']
    v_north = u_x * ds_merge['SN'] + v_y * ds_merge['CS']
    return u_east, v_north


def calculate_jacobian(u_x, v_y, ds_merge, grid):
    """Velocity-gradient tensor (ECCO recipe: rotate, diff, interp
    to centres, rotate) — geographic, signed, centre-interpolated.
    See docs/Gradients.md.

    Applications
    ------------
    Signed COMPONENT fields (ζ, δ, strain_n/s, frontogenesis Q) —
    NOT for squared magnitudes (use calculate_native_strain_vorticity).

    Parameters
       ----------
       u_x : xarray.DataArray
           Zonal velocity component defined in the model x-direction.
       v_y : xarray.DataArray
           Meridional velocity component defined in the model y-direction.
       ds_merge : xarray.Dataset
           Dataset containing grid metrics and rotation coefficients.
       grid : xgcm.Grid
           Grid object used to compute metric-aware interpolation and derivatives.

       Returns
       -------
       du_lambda_dlambda : xarray.DataArray
           Zonal derivative of the zonal velocity component [field_units m^-1].
       du_lambda_dphi : xarray.DataArray
           Meridional derivative of the zonal velocity component [field_units m^-1].
       dv_phi_dlambda : xarray.DataArray
           Zonal derivative of the meridional velocity component [field_units m^-1].
       dv_phi_dphi : xarray.DataArray
           Meridional derivative of the meridional velocity component [field_units m^-1].
       """
    # Move the values to tracer points and rotate the model (x, y) components
    # into the geographic zonal (lambda) / meridional (phi) basis.
    u_lambda, v_phi = rotate_vector_to_geographic(u_x, v_y, ds_merge, grid)

    # Calculate the zonal and meridional gradients of the zonal field ----------------

    # calculate the gradient in the model 'x' direction
    du_lambda_dx = grid.diff(u_lambda, 'X') / ds_merge.dxC
    # calculate the gradient in the model 'y' direction
    du_lambda_dy = grid.diff(u_lambda, 'Y') / ds_merge.dyC

    # interpolate the gradients from cell boundaries to the cell centers
    grad_u_lambda_to_ij_X = grid.interp(du_lambda_dx, 'X', boundary='fill')
    grad_u_lambda_to_ij_Y = grid.interp(du_lambda_dy, 'Y', boundary='fill')

    # rotate to zonal and meridional directions
    # Add the zonal components of the 'X' and 'Y' vector components
    du_lambda_dlambda = grad_u_lambda_to_ij_X * ds_merge['CS'] - grad_u_lambda_to_ij_Y * ds_merge['SN']
    # Add the meridional components
    du_lambda_dphi = grad_u_lambda_to_ij_X * ds_merge['SN'] + grad_u_lambda_to_ij_Y * ds_merge['CS']

    # Calculate the zonal and meridional gradients of the Meridional field ---------

    # calculate the gradient in the model 'x' direction
    dv_phi_dx = grid.diff(v_phi, 'X') / ds_merge.dxC
    # calculate the gradient in the model 'y' direction
    dv_phi_dy = grid.diff(v_phi, 'Y') / ds_merge.dyC

    # interpolate the gradients from cell boundaries to the cell centers
    grad_v_phi_to_ij_X = grid.interp(dv_phi_dx, 'X', boundary='fill')
    grad_v_phi_to_ij_Y = grid.interp(dv_phi_dy, 'Y', boundary='fill')

    # rotate to zonal and meridional directions
    # Add the zonal components of the 'X' and 'Y' vector components
    dv_phi_dlambda = grad_v_phi_to_ij_X * ds_merge['CS'] - grad_v_phi_to_ij_Y * ds_merge['SN']
    # Add the meridional components
    dv_phi_dphi = grad_v_phi_to_ij_X * ds_merge['SN'] + grad_v_phi_to_ij_Y * ds_merge['CS']

    return du_lambda_dlambda, du_lambda_dphi, dv_phi_dlambda, dv_phi_dphi


def calculate_native_gradient_tracer(ds_value, ds_grid, grid):
    """Geographic (zonal, meridional) tracer-gradient COMPONENTS
    (ECCO recipe: diff, interp to centres, CS/SN rotate).  See
    docs/Gradients.md.

    Applications
    ------------
    Signed VECTOR uses only (e.g. ∇b for frontogenesis).  Never
    square these — use calculate_grad_squared_tracer /
    calculate_grad_dot_tracer.

    Parameters
    ----------
    ds_value : xarray.DataArray
        Tracer field to differentiate, defined at cell centers with
        dimensions (face, j, i).
    ds_grid : xarray.Dataset
        Grid metrics dataset containing 'dxC', 'dyC' (cell-center
        distances) and 'CS', 'SN' (rotation coefficients).
    grid : xgcm.Grid
        Grid object used for metric-aware differencing and interpolation.

    Returns
    -------
    ds_dx_hatx_G : xarray.DataArray
        Zonal component of the tracer gradient [field_units m^-1].
    ds_dy_haty_G : xarray.DataArray
        Meridional component of the tracer gradient [field_units m^-1].
    """

    # gradient in X

    # print(f'dxC dimesions: {ds.dxC.dims}')

    # step 1
    # ... the difference in adjacent grid cells at [i,j] and [i-1, j] in the 'x' direction,
    # ... ds denotes the difference in field s
    # ... the _hatx suffix denotes that the difference is in the '\hat{x}' direction.
    # ... the _M suffix denotes we are working in the model basis
    s = ds_value.copy(deep=True)

    ds_hatx_M = grid.diff(s, 'X')

    # step 2
    # ... divide by the distance between
    # ... ds_dx denotes the derivative of field s with respect to distance in meters
    # ... the _hatx suffix denotes that the gradient is in the '\hat{x}' direction.
    ds_dx_hatx_M = ds_hatx_M / ds_grid.dxC


    # gradient in y

    # calculate the gradient of value in 'Y':

    # step 1
    # ... the difference in adjacent grid cells at [i,j] and [i, j-1] in the 'y' direction,
    # ... ds denotes the difference in field s
    # ... the _haty suffix denotes that the difference is in the '\hat{y}' direction.
    # ... the _M suffix denotes we are working in the model basis
    ds_haty_M = grid.diff(ds_value, 'Y')

    # step 2
    # ... divide by the distance between
    # ... ds_dx denotes the derivative of field s with respect to distance in meters
    # ... the _hatx suffix denotes that the gradient is in the '\hat{y}' direction.
    # ... the _M suffix denotes we are working in the model basis
    ds_dy_haty_M = ds_haty_M / ds_grid.dyC


    # Interpolate the gradients to the cell centers
    grad_s_at_cell_center_X = grid.interp(ds_dx_hatx_M, 'X', boundary='fill')
    grad_s_at_cell_center_Y = grid.interp(ds_dy_haty_M, 'Y', boundary='fill')

    # The zonal component of the gradient vector:
    # ... the gradient with respect to x in the G basis.
    ds_dx_hatx_G = grad_s_at_cell_center_X * ds_grid['CS'] - \
                   grad_s_at_cell_center_Y * ds_grid['SN']

    # The meridional component of the gradient vector
    # ... the gradient with respect to x in the G basis
    ds_dy_haty_G = grad_s_at_cell_center_X * ds_grid['SN'] + \
                   grad_s_at_cell_center_Y * ds_grid['CS']

    # update the variable names
    ds_dx_hatx_G.name = 'ds_dx_hatx_G'
    ds_dy_haty_G.name = 'ds_dy_haty_G'

    # ds_dx_hatx_G.attrs.update({'long_name': 'zonal gradient of SSS'})
    # ds_dy_haty_G.attrs.update({'long_name': 'meridional gradient of SSS'})

    # The gradients have units ?/m
    ds_dx_hatx_G.attrs.update({'units': '? m-1'})
    ds_dy_haty_G.attrs.update({'units': '? m-1'})


    return ds_dx_hatx_G, ds_dy_haty_G


def calculate_grad_squared_tracer(ds_value, ds_grid, grid):
    """|∇s|² — squared on the staggered points BEFORE the centre
    interpolation (rotation-free).  THE canonical
    |∇s|² for every ``grad_*2`` field.  See docs/Gradients.md.

    Applications
    ------------
    MAGNITUDE calculations only (no signed components exist here).

    Parameters
    ----------
    ds_value : xarray.DataArray
        Tracer field at cell centres, dims (face, j, i) (a leading
        k is fine — the operation is dimension-agnostic).
    ds_grid : xarray.Dataset
        Grid metrics with ``dxC`` and ``dyC``.
    grid : xgcm.Grid
        Grid used for differencing and interpolation.

    Returns
    -------
    xarray.DataArray
        |grad s|^2 at cell centres [(field units m^-1)^2], lazy.

    Generated by LH and Claude
    """
    # Finite differences on their native staggered points (same
    # stencils as calculate_native_gradient_tracer, steps 1-2).
    ds_dx_M = grid.diff(ds_value, 'X') / ds_grid.dxC
    ds_dy_M = grid.diff(ds_value, 'Y') / ds_grid.dyC

    # Square FIRST (still on the staggered points), then move the
    # non-negative squares to the centre — no cancellation possible.
    dx2_c = grid.interp(ds_dx_M ** 2, 'X', boundary='fill')
    dy2_c = grid.interp(ds_dy_M ** 2, 'Y', boundary='fill')

    out = dx2_c + dy2_c
    out.name = 'calculate_grad_squared_tracer'
    out.attrs.update({'units': '(? m-1)2'})
    return out


def calculate_grad_dot_tracer(da_a, da_b, ds_grid, grid):
    """∇a·∇b — products formed on the staggered points where the
    factors are co-located, moved to the centre afterwards
    (rotation-free; equals calculate_grad_squared_tracer when a is b).  
    See docs/Gradients.md.

    Applications
    ------------
    Consistent gradient dot products (e.g. the Turner angle).

    Parameters
    ----------
    da_a, da_b : xarray.DataArray
        Tracer-point fields, dims (face, j, i) (a leading k is
        fine — dimension-agnostic).
    ds_grid : xarray.Dataset
        Grid metrics with ``dxC`` and ``dyC``.
    grid : xgcm.Grid
        Grid used for differencing and interpolation.

    Returns
    -------
    xarray.DataArray
        grad(a)·grad(b) at tracer points [product units m^-2], lazy.

    Generated by LH and Claude
    """
    ax = grid.diff(da_a, 'X') / ds_grid.dxC
    bx = grid.diff(da_b, 'X') / ds_grid.dxC
    ay = grid.diff(da_a, 'Y') / ds_grid.dyC
    by = grid.diff(da_b, 'Y') / ds_grid.dyC
    return (grid.interp(ax * bx, 'X', boundary='fill')
            + grid.interp(ay * by, 'Y', boundary='fill'))


def calculate_native_strain_vorticity(u_x, v_y, ds_grid, grid):
    """Velocity-gradient combinations, each at its natural C-grid
    point in model-basis — no interpolation or rotation anywhere.
    See docs/Gradients.md.

    Applications
    ------------
    MAGNITUDE calculations only (the strain pair is model-basis;
    vector components are not rotated).

    Parameters
    ----------
    u_x, v_y : xarray.DataArray
        RAW staggered model-axis velocities (U on ``i_g``, V on
        ``j_g``) — NOT the rotated/centred versions.
    ds_grid : xarray.Dataset
        Grid metrics: ``dxC``, ``dyC``, ``dxG``, ``dyG``, ``rA``,
        ``rAz``.
    grid : xgcm.Grid
        Grid object used for the difference stencils.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``strain_normal_center`` and ``divergence_center`` (cell
        centres); ``vorticity_corner`` and ``strain_shear_corner``
        (cell corners).  All [s^-1], lazy.

    Generated by LH and Claude
    """
    dyG, dxG, rA = ds_grid.dyG, ds_grid.dxG, ds_grid.rA
    dxC, dyC, rAz = ds_grid.dxC, ds_grid.dyC, ds_grid.rAz

    # Cell centres: differencing each velocity along ITS OWN axis
    # lands here — zero interpolation (flux form over rA).
    strain_normal_center = (grid.diff(u_x * dyG, 'X')
                            - grid.diff(v_y * dxG, 'Y')) / rA
    divergence_center = (grid.diff(u_x * dyG, 'X')
                         + grid.diff(v_y * dxG, 'Y')) / rA

    # Cell corners: differencing each velocity across the OTHER
    # axis lands here — zero interpolation (circulation form over
    # rAz; the vorticity is MITgcm's momVort3).
    vorticity_corner = (grid.diff(v_y * dyC, 'X')
                        - grid.diff(u_x * dxC, 'Y')) / rAz
    strain_shear_corner = (grid.diff(v_y * dyC, 'X')
                           + grid.diff(u_x * dxC, 'Y')) / rAz

    return {"strain_normal_center": strain_normal_center,
            "divergence_center": divergence_center,
            "vorticity_corner": vorticity_corner,
            "strain_shear_corner": strain_shear_corner}


def interp_corner_squared(q_corner, grid):
    """Average a corner value onto the cell centres (2-pt mean in
    x, then y) — for ALREADY-SQUARED quantities only.  See
    docs/Gradients.md.

    Applications
    ------------
    Non-negative (squared) corner fields only; never signed ones.

    Parameters
    ----------
    q_corner : xarray.DataArray
        Non-negative field on corner points ``(j_g, i_g)``, e.g.
        ``vorticity_corner ** 2``.
    grid : xgcm.Grid
        Grid object used for the interpolation.

    Returns
    -------
    xarray.DataArray
        The field on tracer points ``(j, i)``, lazy.

    Generated by LH and Claude
    """
    return grid.interp(grid.interp(q_corner, 'X', boundary='fill'),
                       'Y', boundary='fill')
