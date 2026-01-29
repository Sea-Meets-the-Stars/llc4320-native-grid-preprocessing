import dbof.utils.physical_calculations as physical_calculations
import dbof.utils.native_gradient as ng
import dask.array as da

#todo add wind curl here
# todo clean up comments after completing tests
def calculate_gradients(ds_merge, grid):
    """
    Compute log10 of squared buoyancy gradients on the native LLC grid.

    The resulting log-gradient field is added to the
    merged dataset_creation as a new variable.

    Parameters
    ----------
    ds_merge : xarray.Dataset
        Dataset containing fields.
    grid : xgcm.Grid
        XGCM grid object.

    Returns
    -------
    ds_merge : xarray.Dataset
        Dataset augmented with `log_gradb`.
    log_gradb : dask.array.Array
        log10(|∇b|^2) field used for weighted sampling.
    """

    buoyancy = physical_calculations.buoyancy_of_field(ds_merge)

    # gradient of b
    zonal_grad_b, merid_grad_b = ng.calculate_native_gradient_tracer(buoyancy, ds_merge, grid=grid)

    #zonal_grad_b, merid_grad_b = dask.persist(zonal_grad_b, merid_grad_b)

    # zonal_grad_b = zonal_grad_b.persist()
    # merid_grad_b = merid_grad_b.persist()

    gradb2 = physical_calculations.grad_squared(zonal_grad_b, merid_grad_b)
    # gradb2 = gradb2.persist()

    log_gradb = da.log10(gradb2)

    #log_gradb = log_gradb.persist()

    #log_gradb_ds = log_gradb.to_dataset(name="log_gradb")

    #ds_merge = xr.merge([ds_merge, log_gradb_ds])

    # ds_merge["log_gradb"] = ds_merge["log_gradb"].persist()

    return ds_merge, log_gradb