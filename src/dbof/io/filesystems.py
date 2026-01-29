import fsspec

def create_s3_filesystems(s3_endpoint):
    """
    Create asynchronous and synchronous S3 filesystems.

    The asynchronous filesystem is used for Zarr writes,
    while the synchronous filesystem is used for Parquet metadata.

    Parameters
    ----------
    s3_endpoint : str
        S3-compatible endpoint URL.

    Returns
    -------
    fs : fsspec.AbstractFileSystem
        Asynchronous S3 filesystem.
    fs_synch : fsspec.AbstractFileSystem
        Synchronous S3 filesystem.
    """

    fs = fsspec.filesystem(
        "s3",  #
        asynchronous=True,
        client_kwargs={
            "endpoint_url": s3_endpoint,

        },

        # These become botocore.client.Config(...)
        config_kwargs={
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "s3": {
                "addressing_style": "path",
                "payload_signing_enabled": False,
                "use_accelerate_endpoint": False,
                "use_dualstack_endpoint": False,
            },
        },
    )

    fs_synch = fsspec.filesystem(
        "s3",  #
        asynchronous=False,
        client_kwargs={
            "endpoint_url": s3_endpoint,

        },

        # These become botocore.client.Config(...)
        config_kwargs={
            "signature_version": "s3v4",
            "request_checksum_calculation": "when_required",
            "s3": {
                "addressing_style": "path",
                "payload_signing_enabled": False,
                "use_accelerate_endpoint": False,
                "use_dualstack_endpoint": False,
            },
        },
    )

    return fs, fs_synch