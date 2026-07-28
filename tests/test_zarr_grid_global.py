"""Unit tests for GlobalGridZarrReader.to_dataset_lazy."""
import dask.array as da
import zarr

from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader


def _reader_with(vars_):
    root = zarr.group()
    for name in vars_:
        root.create_array(name, shape=(4, 5), chunks=(4, 5), dtype="f4")
    reader = GlobalGridZarrReader.__new__(GlobalGridZarrReader)  # bypass S3 __init__
    reader.root = root
    reader.variables = list(vars_)
    return reader


def test_to_dataset_lazy_dims_and_lazy():
    reader = _reader_with(["XC", "YC"])
    ds = reader.to_dataset_lazy()
    assert set(ds.data_vars) == {"XC", "YC"}
    assert ds["XC"].dims == ("j", "i")
    assert ds["XC"].shape == (4, 5)
    assert isinstance(ds["XC"].data, da.Array)  # not materialized


def test_to_dataset_lazy_variable_subset():
    reader = _reader_with(["XC", "YC", "dxC"])
    ds = reader.to_dataset_lazy(variables=["XC"])
    assert set(ds.data_vars) == {"XC"}
