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
    bucket = "dbof"
    folder = "cutouts_dataset_v2_TESTING"
    run_id = "cutout_test_data_v1"
    s3_endpoint = "https://s3-west.nrp-nautilus.io"

    fs, fs_synch = filesystems.create_s3_filesystems(s3_endpoint)
    try:
        reader = zarr_dataset.ZarrDatasetReader(
            bucket=bucket,
            folder=folder,
            run_id=run_id,
            dataset_name="cutout_dataset_creation.zarr",
            fs=fs,
        )
    except Exception as exc:
        pytest.skip(f"cutout store unreachable (s3://{bucket}/{folder}/{run_id}): {exc}")
    return reader
