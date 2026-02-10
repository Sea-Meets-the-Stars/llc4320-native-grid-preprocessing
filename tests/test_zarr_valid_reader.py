import numpy as np
import pytest

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

    # Build valid index once
    valid_indices = zarr_reader.build_valid_indices()
    assert valid_indices.size > 0, "No valid samples found — dataset may be empty"

    # Iterate using the safe path
    for img, image_id in zarr_reader.iter_images(batch_size=1):
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
