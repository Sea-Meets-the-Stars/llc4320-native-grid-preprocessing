""" Code for masking ice in the LLC4320 model """


def mask_by_theta(ds, theta_threshold: float = 0.0):
    """ Mask the dataset by the theta threshold 

    Args:
        ds: xarray.Dataset
        theta_threshold: float

    Returns:
        xarray.Dataset
    """
    ice_mask = ~(ds.Theta <= theta_threshold)
    return ice_mask