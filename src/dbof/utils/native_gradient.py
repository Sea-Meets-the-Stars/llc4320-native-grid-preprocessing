
def calculate_native_gradient_tracer(ds_value, ds_grid, grid):
    """
    Calculate the gradient of a tracer variable in native model coordinates

    Parameters
   ----------
    ds : Xarray Dataset with value to calculate gradient
    must have been merged with grid file.

    value : String

    grid : Xgcm grid

    Returns
    -------

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
    #print(f'dyC dims: {ds.dyC.dims}')

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

    # print(f'dimensions of ds_haty      : {ds_haty_M.dims}')
    # print(f'dimensions of ds_dy_haty_M : {ds_dy_haty_M.dims}')





    # todo this is different from tutorial. Is it correct
    grad_s_at_cell_center_X = grid.interp(ds_dx_hatx_M, 'X', boundary='fill')
    grad_s_at_cell_center_Y = grid.interp(ds_dy_haty_M, 'Y', boundary='fill')

    # grad_s_at_cell_center = xr.merge([grad_s_at_cell_center_X, grad_s_at_cell_center_Y])

    # grad_s_at_cell_center = grid.interp_2d_vector({'X': ds_dx_hatx_M, 'Y': ds_dy_haty_M}, boundary='fill')

    # print(f'the keys of grad_s_at_cell_center X are {list(grad_s_at_cell_center_X.keys() )}')
    # print(f'the keys of grad_s_at_cell_center X are {list(grad_s_at_cell_center_Y.keys() )}')

    # print(f"\nds_grad_vec X component {grad_s_at_cell_center_X.dims}")
    # print(f"ds_grad_vec Y component {grad_s_at_cell_center_Y.dims}")

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

    ds_dx_hatx_G.attrs.update({'long_name': 'zonal gradient of SSS'})
    ds_dy_haty_G.attrs.update({'long_name': 'meridional gradient of SSS'})

    # The gradients have units ?/m
    ds_dx_hatx_G.attrs.update({'units': '? m-1'})
    ds_dy_haty_G.attrs.update({'units': '? m-1'})


    return ds_dx_hatx_G, ds_dy_haty_G
