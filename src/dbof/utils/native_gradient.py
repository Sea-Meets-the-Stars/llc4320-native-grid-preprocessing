def rotate_vector_to_geographic(u_x, v_y, ds_merge, grid, *, interpolate=True):
    """Rotate a model-grid vector into geographic (east/north) components.

    The LLC grid stores horizontal vector components on the staggered C-grid
    (``u`` on the west cell face / ``i_g``, ``v`` on the south face / ``j_g``)
    and on model-relative axes that are rotated relative to true east/north.
    This helper performs the two steps needed to make such a vector physically
    interpretable on a single native face (chunk, tile, or global):

    1. interpolate the components to the tracer (cell-centre) points, and
    2. rotate from the model ``(x, y)`` basis to the geographic
       ``(east, north)`` basis using the grid rotation coefficients
       ``CS``/``SN``::

           u_east  = u*CS - v*SN
           v_north = u*SN + v*CS

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
    """
       Compute zonal and meridional spatial derivatives of horizontal velocity components on a curvilinear grid.
       See https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Gradient_calc_on_native_grid.html#Part-2:-calculate-the-zonal-and-meridional-gradients-of-the-zonal-and-meridional-flow-fields

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
    """Compute zonal and meridional gradients of a tracer on the native LLC grid.

    Calculates finite-difference gradients in model (x, y) directions,
    interpolates them to cell centers, then rotates from model coordinates
    to geographic (zonal/meridional) coordinates using the grid rotation
    coefficients CS and SN.

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
    """|grad s|^2 with squaring BEFORE the centre interpolation.

    THE canonical |grad s|^2 for every ``grad_*2`` field.  The
    ECCO recipe (``calculate_native_gradient_tracer``) computes the finite differences on the staggered points,
    interpolates them to the cell centre (2-point mean), rotates,
    and squares.  The 2-point mean has a null space at the grid
    scale: at a local extremum of ``s`` the two flanking
    differences are equal-and-opposite and CANCEL in the average,
    so the centred |grad s|^2 can be orders of magnitude below its
    one-sided gradients — the white 'sparkle' on log-scaled maps.

    This version squares each component ON its native staggered
    point (where the finite difference is defined) and interpolates
    the non-negative squares to the centre — a mean of two
    non-negative numbers cannot cancel, so grid-scale gradient
    variance is retained (variance-preserving form).

    No CS/SN rotation is needed: |grad s|^2 is invariant under the
    orthogonal model->geographic rotation (zonal^2 + merid^2 =
    x_M^2 + y_M^2 pointwise), so squaring before rotation is exact.

    Both forms are consistent O(dx^2) discretizations of the same
    continuum quantity; they differ only in how they treat the
    2*dx scale.

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


def kinematic_invariants(u_x, v_y, ds_merge, grid):
    """C-grid-native rotation invariants of the velocity gradient.

    Each quantity is computed ON its natural C-grid point and NEVER
    interpolated (see
    https://mitgcm.readthedocs.io/en/latest/algorithm/horiz-grid.html):

    - ``sn_c``    normal strain  du/dx - dv/dy   (centres, flux
      form over ``rA``);
    - ``delta_c`` divergence     du/dx + dv/dy   (centres);
    - ``zeta_z``  vorticity      dv/dx - du/dy   (corners,
      circulation form over ``rAz`` -- MITgcm's own momVort3);
    - ``ss_z``    shear strain   dv/dx + du/dy   (corners).

    All four are invariant under the CS/SN model->geographic
    rotation, so no rotation is applied.  Squared kinematic fields
    (strain magnitude, Okubo-Weiss) must be built from THESE, squared
    at their native points, with only the non-negative squares moved
    between grid locations (``interp_corner_squared``) -- squaring
    centre-interpolated components manufactures near-zeros at
    grid-scale extrema (the 'sparkle';
    prompts/field_validation.md, 2026-08-05).

    A/B-validated against the interpolate-then-square recipe in
    notebooks/notebooks_dev/field_validation_sparkle.ipynb.

    Parameters
    ----------
    u_x, v_y : xarray.DataArray
        RAW staggered model-axis velocities (U on ``i_g``, V on
        ``j_g``) -- NOT the rotated/centred versions.
    ds_merge : xarray.Dataset
        Grid metrics: ``dxC``, ``dyC``, ``dxG``, ``dyG``, ``rA``,
        ``rAz``.
    grid : xgcm.Grid
        Grid object used for the difference stencils.

    Returns
    -------
    dict[str, xarray.DataArray]
        ``{'sn_c', 'delta_c', 'zeta_z', 'ss_z'}`` [s^-1], lazy.

    Generated by LH and Claude
    """
    dyG, dxG, rA = ds_merge.dyG, ds_merge.dxG, ds_merge.rA
    dxC, dyC, rAz = ds_merge.dxC, ds_merge.dyC, ds_merge.rAz

    # Centre invariants (flux form over the tracer cell area).
    sn_c = (grid.diff(u_x * dyG, 'X')
            - grid.diff(v_y * dxG, 'Y')) / rA
    delta_c = (grid.diff(u_x * dyG, 'X')
               + grid.diff(v_y * dxG, 'Y')) / rA

    # Corner invariants (circulation form over the vorticity cell).
    zeta_z = (grid.diff(v_y * dyC, 'X')
              - grid.diff(u_x * dxC, 'Y')) / rAz
    ss_z = (grid.diff(v_y * dyC, 'X')
            + grid.diff(u_x * dxC, 'Y')) / rAz

    return {"sn_c": sn_c, "delta_c": delta_c,
            "zeta_z": zeta_z, "ss_z": ss_z}


def interp_corner_squared(q_z, grid):
    """Corner -> centre interpolation of an ALREADY-SQUARED field.

    For non-negative corner quantities (``zeta_z**2``, ``ss_z**2``)
    only: a mean of non-negative numbers cannot cancel, so moving
    the SQUARES preserves grid-scale variance where moving the
    signed fields would not.

    Parameters
    ----------
    q_z : xarray.DataArray
        Non-negative field on corner points ``(j_g, i_g)``.
    grid : xgcm.Grid
        Grid object used for the interpolation.

    Returns
    -------
    xarray.DataArray
        The field on tracer points ``(j, i)``, lazy.

    Generated by LH and Claude
    """
    return grid.interp(grid.interp(q_z, 'X', boundary='fill'),
                       'Y', boundary='fill')
