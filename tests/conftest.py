import pytest
import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.io.filesystems as filesystems


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