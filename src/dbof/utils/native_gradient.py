from IPython import embed

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
           Zonal derivative of the zonal velocity component.
       du_lambda_dphi : xarray.DataArray
           Meridional derivative of the zonal velocity component.
       dv_phi_dlambda : xarray.DataArray
           Zonal derivative of the meridional velocity component.
       dv_phi_dphi : xarray.DataArray
           Meridional derivative of the meridional velocity component.
       """
    embed(header='29 of calculate_jacobian')
    # Move the values to tracer position
    vec_u_to_ij = grid.interp(u_x, 'X', boundary='fill')
    vec_v_to_ij = grid.interp(v_y, 'Y', boundary='fill')

    # rotate the interpolated vectors to the zonal (lambda) and meridional (phi) basis (basically just from model direction to real)
    # Add the zonal components of the 'X' and 'Y' vectors
    u_lambda = vec_u_to_ij * ds_merge['CS'] - vec_v_to_ij * ds_merge['SN']
    # Add the meridional components
    v_phi = vec_u_to_ij * ds_merge['SN'] + vec_v_to_ij * ds_merge['CS']

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
