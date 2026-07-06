import pytest
import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.io.filesystems as filesystems


# ---------------------------------------------------------------------------
# Integration-test gating
# ---------------------------------------------------------------------------
# Tests marked 'integration' touch real data stores (MIT filesystem, dbof S3,
# OSN) and are skipped by default so plain `pytest` stays fast and offline.
# Opt in explicitly:
#
#   pytest --run-integration                          # all integration tests
#   pytest --run-integration -m "mit and not s3_dbof" # e.g. dbof bucket down
#
# (Pattern from the pytest docs: "control skipping of tests according to
# command line option".)

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run tests marked 'integration' (touch real data stores)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="integration test: pass --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def zarr_reader():
    bucket = "llc"  # data_cfg["bucket"]
    folder = "native_grid_dbof_training_data/"
    s3_endpoint = "https://s3-west.nrp-nautilus.io"
    feature_channels = ['Eta', 'Salt', 'Theta', 'U', 'V', 'W', 'relative_vorticity', 'log_gradb']
    run_id = "test00" # this run has empty data.

    fs, fs_synch = filesystems.create_s3_filesystems(s3_endpoint)

    reader = zarr_dataset.ZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name="cutout_dataset_creation.zarr",
        fs=fs
    )
    return reader
