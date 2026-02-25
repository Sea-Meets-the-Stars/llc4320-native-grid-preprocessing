import numpy as np
import pytest
import dask.array as da

def is_empty_image_id(image_id):
    """
    Returns True if image_id represents an empty / unwritten entry.
    Handles fixed-length bytes, object bytes, and str.
    """
    if isinstance(image_id, (bytes, bytearray, np.bytes_)):
        return image_id.decode("ascii") == ""
    if isinstance(image_id, str):
        return image_id == ""
    return False


def test_zarr_reader_returns_only_valid_data(zarr_reader):
    """
    Ensure that iterating via valid_indices never yields empty data.
    """
    images_da, ids_da, valid_mask_da = zarr_reader.full_dataset_as_dask()

    valid_idx = da.nonzero(valid_mask_da)[0].compute()  # numpy array in RAM

    assert valid_idx.size > 0, "No valid samples found — dataset may be empty"

    valid_images = da.take(images_da, valid_idx, axis=0).compute()
    valid_ids = da.take(ids_da, valid_idx, axis=0).compute()

    # Iterate using the safe path
    for img, image_id in zip(valid_images, valid_ids):
        # iter_images yields batches; unwrap
        img = img[0]
        image_id = image_id[0]

        assert not is_empty_image_id(image_id), (
            "Empty image_id passed through valid iteration"
        )

        ch0 = img[0]
        assert np.any(ch0 != 0), (
            "Image content is empty (channel 0 is all zeros)"
        )
