import pytest
import dbof.dataset_creation.zarr_dataset as zarr_dataset
import dbof.io.filesystems as filesystems


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
        dataset_name="dataset_creation.zarr",
        fs=fs
    )
    return reader