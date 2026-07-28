"""
test_zarr_to_netcdf.py
----------------------
Hermetic unit tests for ``dbof.cli.zarr_to_netcdf``.

Why this can be data-free
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``zarr_to_netcdf`` is *I/O-shaped*: it reads channel arrays out of a
``GlobalZarrDatasetReader`` (or grid arrays out of a
``GlobalGridZarrReader``) and writes one NetCDF file per timestep with a
fixed set of attributes.  The store readers (and their real S3 round-trip)
are exercised elsewhere; here we inject a tiny in-memory **fake reader** so
the conversion + CLI-dispatch logic is tested with no S3, no zarr store, and
no LLC4320 data.

What is tested
~~~~~~~~~~~~~~
* ``zarr_to_netcdf`` writes a NetCDF with the expected data variables,
  pixel values, dims and global attributes (default filename).
* a channel subset + an explicit ``output_filename`` are honoured.
* an unknown channel name raises ``ValueError``.
* the optional ice mask is applied to every exported channel.
* ``main`` dispatches the snapshots mode (with an explicit ``--date-prefix``)
  and requires ``run_id`` in snapshots mode.
* ``main`` dispatches the grid mode, promoting XC/YC to lon/lat coords.

CLI usage
---------
    pip install pytest
    pytest tests/test_zarr_to_netcdf.py -v
"""

import numpy as np
import pytest
import xarray as xr

import dbof.cli.zarr_to_netcdf as ztn


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

CHANNELS = ["Theta_sfc", "Salt_sfc", "N2_sfc"]
SHAPE = (4, 5)            # small rectangular (H, W)
DATE_PREFIX = "20121109_120000"
RUN_ID = "test_run00"


class FakeSnapshotReader:
    """Stand-in for ``GlobalZarrDatasetReader``.

    Holds an in-memory ``(C, H, W)`` array (single snapshot, no time axis).
    Channel ``c`` is filled with the constant ``c`` so every cell is uniquely
    identifiable in assertions.
    """

    def __init__(self, channel_names=CHANNELS, rectangular_shape=SHAPE,
                 iteration=42, **_ignored):
        self.channel_names = list(channel_names)
        self.rectangular_shape = tuple(rectangular_shape)
        self.iteration = iteration
        h, w = self.rectangular_shape
        c = len(self.channel_names)
        data = np.empty((c, h, w), dtype=np.float32)
        for ci in range(c):
            data[ci] = float(ci)
        self._data = data

    def get_channel_snapshot(self, channel):
        if isinstance(channel, str):
            channel = self.channel_names.index(channel)
        return self._data[channel]


class FakeGridReader:
    """Stand-in for ``GlobalGridZarrReader``."""

    def __init__(self, **_ignored):
        self.variable_names = ["XC", "YC", "Depth"]
        self.shape = SHAPE
        h, w = SHAPE
        self._vars = {
            "XC": np.full((h, w), 10.0, dtype=np.float32),
            "YC": np.full((h, w), 20.0, dtype=np.float32),
            "Depth": np.full((h, w), 30.0, dtype=np.float32),
        }

    def get_variable(self, name):
        return self._vars[name]

    def get_variable_attrs(self, name):
        return {"long_name": name}


@pytest.fixture
def fake_snapshot_reader(monkeypatch):
    """Patch the snapshot reader so a single fake store is used everywhere."""
    def _factory(**kwargs):
        return FakeSnapshotReader(**kwargs)
    monkeypatch.setattr(
        ztn.zarr_dataset_global, "GlobalZarrDatasetReader", _factory)
    return _factory


@pytest.fixture
def fake_grid_reader(monkeypatch):
    monkeypatch.setattr(
        ztn.zarr_grid_global, "GlobalGridZarrReader",
        lambda **kwargs: FakeGridReader())


# ---------------------------------------------------------------------------
# zarr_to_netcdf (snapshots) — core conversion
# ---------------------------------------------------------------------------

def test_writes_expected_default_file(tmp_path, fake_snapshot_reader):
    """Default run writes {run_id}_{date_prefix}.nc with all channels."""
    ztn.zarr_to_netcdf(
        s3_endpoint="https://example",
        bucket="dbof",
        folder="depth_fields",
        run_id=RUN_ID,
        dataset_name="stratification.zarr",
        output_dir=str(tmp_path),
        date_prefix=DATE_PREFIX,
        fs=object(),            # never touched by the fake reader
    )

    nc = tmp_path / f"{RUN_ID}_{DATE_PREFIX}.nc"
    assert nc.exists()

    with xr.open_dataset(nc) as ds:
        # All channels present as data variables.
        assert set(ds.data_vars) == set(CHANNELS)
        # Values match the fake reader's t*100 + channel_index pattern (t=0).
        for ci, ch in enumerate(CHANNELS):
            assert np.allclose(ds[ch].values, float(ci))
        # Dims and shape.
        assert ds[CHANNELS[0]].dims == ("y", "x")
        assert ds[CHANNELS[0]].shape == SHAPE
        # Global attributes.
        assert ds.attrs["date_prefix"] == DATE_PREFIX
        assert ds.attrs["run_id"] == RUN_ID
        assert list(ds.attrs["channel_names"]) == CHANNELS


def test_channel_subset_and_output_filename(tmp_path, fake_snapshot_reader):
    """A channel subset + explicit filename are both honoured."""
    ztn.zarr_to_netcdf(
        s3_endpoint="https://example",
        bucket="dbof",
        folder="depth_fields",
        run_id=RUN_ID,
        dataset_name="stratification.zarr",
        output_dir=str(tmp_path),
        date_prefix=DATE_PREFIX,
        output_filename="just_n2.nc",
        channels=["N2_sfc"],
        fs=object(),
    )

    nc = tmp_path / "just_n2.nc"
    assert nc.exists()
    with xr.open_dataset(nc) as ds:
        assert set(ds.data_vars) == {"N2_sfc"}
        # N2_sfc is channel index 2 -> filled with 2.0 at t=0.
        assert np.allclose(ds["N2_sfc"].values, 2.0)


def test_unknown_channel_raises(tmp_path, fake_snapshot_reader):
    with pytest.raises(ValueError, match="not found in store"):
        ztn.zarr_to_netcdf(
            s3_endpoint="https://example",
            bucket="dbof",
            folder="depth_fields",
            run_id=RUN_ID,
            dataset_name="stratification.zarr",
            output_dir=str(tmp_path),
            date_prefix=DATE_PREFIX,
            channels=["does_not_exist"],
            fs=object(),
        )


def test_ice_mask_applied(tmp_path, fake_snapshot_reader, monkeypatch):
    """When an ice-mask dataset is given, masked cells become NaN."""
    h, w = SHAPE
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True  # mask a single cell

    monkeypatch.setattr(ztn, "load_siarea_mask", lambda **kwargs: mask)
    monkeypatch.setattr(
        ztn, "apply_ice_mask",
        lambda arr, m: np.where(m, np.nan, arr))

    ztn.zarr_to_netcdf(
        s3_endpoint="https://example",
        bucket="dbof",
        folder="depth_fields",
        run_id=RUN_ID,
        dataset_name="stratification.zarr",
        output_dir=str(tmp_path),
        date_prefix=DATE_PREFIX,
        fs=object(),
        ice_mask_dataset_name="icearea.zarr",
    )

    nc = tmp_path / f"{RUN_ID}_{DATE_PREFIX}.nc"
    with xr.open_dataset(nc) as ds:
        for ch in CHANNELS:
            assert np.isnan(ds[ch].values[0, 0])
            assert not np.isnan(ds[ch].values[0, 1])


# ---------------------------------------------------------------------------
# main() — CLI dispatch
# ---------------------------------------------------------------------------

def test_main_snapshots_dispatch(tmp_path, fake_snapshot_reader, monkeypatch):
    """main(mode='snapshots') with an explicit date_prefix writes a file."""
    monkeypatch.setattr(
        ztn, "create_s3_filesystems", lambda endpoint: (object(), object()))

    ztn.main(
        output_dir=str(tmp_path),
        mode="snapshots",
        s3_endpoint="https://example",
        bucket="dbof",
        folder="depth_fields",
        run_id=RUN_ID,
        dataset_name="stratification.zarr",
        date_prefix=DATE_PREFIX,
    )

    assert (tmp_path / f"{RUN_ID}_{DATE_PREFIX}.nc").exists()


def test_main_snapshots_requires_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ztn, "create_s3_filesystems", lambda endpoint: (object(), object()))
    with pytest.raises(ValueError, match="run_id is required"):
        ztn.main(
            output_dir=str(tmp_path),
            mode="snapshots",
            s3_endpoint="https://example",
            bucket="dbof",
            folder="depth_fields",
            run_id=None,
            dataset_name="stratification.zarr",
            date_prefix=DATE_PREFIX,
        )


def test_main_grid_dispatch(tmp_path, fake_grid_reader, monkeypatch):
    """main(mode='grid') writes the grid file and promotes XC/YC to coords."""
    monkeypatch.setattr(
        ztn, "create_s3_filesystems", lambda endpoint: (object(), object()))

    ztn.main(
        output_dir=str(tmp_path),
        mode="grid",
        s3_endpoint="https://example",
        bucket="dbof",
        folder="LLC4320_GRID_2D",
        grid_dataset_name="llc4320_grid.zarr",
        grid_output_filename="grid_out.nc",
    )

    nc = tmp_path / "grid_out.nc"
    assert nc.exists()
    with xr.open_dataset(nc) as ds:
        # XC/YC promoted to lon/lat coordinates; Depth stays a data var.
        assert "lon" in ds.coords
        assert "lat" in ds.coords
        assert "Depth" in ds.data_vars
        assert "XC" not in ds.data_vars
        assert np.allclose(ds["Depth"].values, 30.0)


# ---------------------------------------------------------------------------
# Real writer -> reader round-trip (3-D, no time axis)
#
# Exercises the actual GlobalZarrDataset writer + GlobalZarrDatasetReader on an
# in-memory fsspec filesystem.  Needs zarr v3 (``zarr.storage.FsspecStore``),
# so it is skipped on environments with an older zarr (e.g. zarr 2.x).
# ---------------------------------------------------------------------------

import fsspec        # noqa: E402
import zarr          # noqa: E402

_HAS_FSSPEC_STORE = hasattr(getattr(zarr, "storage", None), "FsspecStore")


@pytest.mark.skipif(not _HAS_FSSPEC_STORE,
                    reason="needs zarr v3 (zarr.storage.FsspecStore)")
def test_writer_reader_roundtrip_3d():
    from dbof.global_dataset_creation.zarr_dataset_global import (
        GlobalZarrDataset, GlobalZarrDatasetReader,
    )

    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()

    chans = ["A", "B", "C"]
    h, w = 4, 5
    writer = GlobalZarrDataset(
        "b", "f", "run", "d.zarr", fs=fs,
        channel_names=chans, rectangular_shape=(h, w),
        date_prefix="20121109_120000",
    )
    data = np.stack(
        [np.full((h, w), i, dtype=np.float32) for i in range(len(chans))]
    )  # channel i filled with value i
    writer.write_snapshot(data, iteration=7)

    reader = GlobalZarrDatasetReader(
        "b", "f", "run", "d.zarr", fs=fs, date_prefix="20121109_120000",
    )
    # Storage is 3-D: (C, H, W); no time axis.
    assert reader.shape == (len(chans), h, w)
    assert reader.n_channels == len(chans)
    assert reader.channel_names == chans
    assert reader.rectangular_shape == (h, w)
    assert reader.iteration == 7
    # Accessors take no t.
    assert np.allclose(reader.get_channel_snapshot("B"), 1.0)
    assert np.allclose(reader.get_snapshot()[2], 2.0)


@pytest.mark.skipif(not _HAS_FSSPEC_STORE,
                    reason="needs zarr v3 (zarr.storage.FsspecStore)")
def test_incomplete_store_has_no_iteration_marker():
    """A store created but never written carries no 'iteration' attr."""
    from dbof.global_dataset_creation.zarr_dataset_global import GlobalZarrDataset
    import dbof.global_dataset_creation.check_existence as ce

    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()

    chans = ["A", "B"]
    store_path = "s3://b/f/run/20121109_120000/d.zarr"
    GlobalZarrDataset(  # construct only -> array exists, no write
        "b", "f", "run", "d.zarr", fs=fs,
        channel_names=chans, rectangular_shape=(3, 3),
        date_prefix="20121109_120000",
    )
    assert ce.plan_zarr(fs, store_path, chans) == ce.ZARR_INCOMPLETE
