# This dictionary is used to define the face connections for the LLC4320 grid.
# It is used to create the xgcm Grid object.

import xgcm

face_connections = {'face':  {
        0: {'X': ((12, 'Y', False), (3, 'X', False)),
            'Y': (None, (1, 'Y', False))},
        1: {'X': ((11, 'Y', False), (4, 'X', False)),
            'Y': ((0, 'Y', False), (2, 'Y', False))},
        2: {'X': ((10, 'Y', False), (5, 'X', False)),
            'Y': ((1, 'Y', False), (6, 'X', False))},
        3: {'X': ((0, 'X', False), (9, 'Y', False)),
            'Y': (None, (4, 'Y', False))},
        4: {'X': ((1, 'X', False), (8, 'Y', False)),
            'Y': ((3, 'Y', False), (5, 'Y', False))},
        5: {'X': ((2, 'X', False), (7, 'Y', False)),
            'Y': ((4, 'Y', False), (6, 'Y', False))},
        6: {'X': ((2, 'Y', False), (7, 'X', False)),
            'Y': ((5, 'Y', False), (10, 'X', False))},
        7: {'X': ((6, 'X', False), (8, 'X', False)),
            'Y': ((5, 'X', False), (10, 'Y', False))},
        8: {'X': ((7, 'X', False), (9, 'X', False)),
            'Y': ((4, 'X', False), (11, 'Y', False))},
        9: {'X': ((8, 'X', False), None),
            'Y': ((3, 'X', False), (12, 'Y', False))},
        10: {'X': ((6, 'Y', False), (11, 'X', False)),
                'Y': ((7, 'Y', False), (2, 'X', False))},
        11: {'X': ((10, 'X', False), (12, 'X', False)),
                'Y': ((8, 'Y', False), (1, 'X', False))},
        12: {'X': ((11, 'X', False), None),
                'Y': ((9, 'Y', False), (0, 'X', False))}
}}

def set_xgcm_grid(ds_grid, use_connections:bool=True):
    """ Set the xgcm Grid object for the LLC4320 grid 
    Args:
        ds_grid: xarray.Dataset
            The dataset containing the LLC4320 grid data 

    Returns:
        xgcm.Grid
            The xgcm Grid object for the LLC4320 grid
    """
    # Do it
    if use_connections:
        grid = xgcm.Grid(ds_grid, periodic=False, face_connections=face_connections)
    else:
        grid = xgcm.Grid(ds_grid, periodic=False)
    return grid