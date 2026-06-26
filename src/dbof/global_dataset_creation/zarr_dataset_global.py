"""
zarr_dataset_global.py
----------------------
Zarr writer/reader for global LLC4320 snapshots in rectangular lat/lon format.

This writer:
  - Stores one full global snapshot per store in rectangular lat/lon format
  - The date/timestep is encoded in the store path (``…/{date_prefix}/…``),
    so a store holds exactly one snapshot and carries no time dimension
  - Records the LLC4320 iteration number as a root attribute
  - Stores rectangular grid shape and channel names as Zarr attributes

Storage layout
--------------
  data  : float32, shape (C, rectangular_h, rectangular_w)  — all channels, one snapshot

Root group attributes
---------------------
  channel_names     : list[str]       — ordered channel labels, matching axis C
  rectangular_shape : [rectangular_h, rectangular_w]  — 2D shape of each global field
  iteration         : int             — LLC4320 iteration number for this snapshot.
                                        Written *last* by ``write_snapshot``, so its
                                        presence marks a fully-written store.

Chunk strategy
--------------
  data chunks are (1, rectangular_h, rectangular_w): one channel field per chunk.
  This matches the expected access pattern (reading one full global field at a time).

Notes on Dask / .compute()
---------------------------
  This writer accepts plain numpy arrays. Calling .values / .compute() to materialise
  data before passing it here is intentional — we are inside a sequential Python
  for-loop, NOT inside a dask.delayed task graph. The warning in the cutout pipeline
  about avoiding .compute() applies only inside delayed/distributed tasks.
"""

import numpy as np
import zarr
from pathlib import PurePosixPath


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_run_prefix(
    bucket: str,
    folder: str,
    run_id: str,
    dataset_name: str,
    date_prefix: str | None = None,
) -> str:
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    if date_prefix:
        return f"s3://{str(PurePosixPath(bucket, folder, run_id, date_prefix, dataset_name))}"
    return f"s3://{str(PurePosixPath(bucket, folder, run_id, dataset_name))}"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class GlobalZarrDataset:
    """
    Sequential writer for global LLC4320 snapshots in LLC compact format.

    One call to write_snapshot() appends one timestep. The store grows along
    the time axis with each call. Safe to resume: if the store already exists,
    writing picks up from where the last run left off.

    Parameters
    ----------
    bucket : str
    folder : str
    run_id : str
    dataset_name : str
        S3 path components
    fs : fsspec AbstractFileSystem
        Async S3 filesystem returned by create_s3_filesystems().
    channel_names : list[str]
        Ordered channel names, e.g. ["Eta", "Salt", "Theta", "U", "V", "W", "log_gradb"].
        Length must match axis C of every array passed to write_snapshot().
    rectangular_shape : tuple[int, int]
        (rectangular_h, rectangular_w) — the 2D shape of a single rectangular global field
        as returned by xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon().
        Typically (12960, 17280). Determined once via a dry-run in main() before
        the writer is constructed.
    date_prefix : str or None, optional
        Date subdirectory inserted between *run_id* and *dataset_name* in the
        S3 path (e.g. ``'20121109_120000'``).  When provided the store path
        becomes ``s3://{bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}``.
    """

    def __init__(
        self,
        bucket: str,
        folder: str,
        run_id: str,
        dataset_name: str,
        fs,
        channel_names: list,
        rectangular_shape: tuple,
        date_prefix: str | None = None,
    ):
        path = make_run_prefix(bucket, folder, run_id, dataset_name,
                               date_prefix=date_prefix)
        self.store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="a", use_consolidated=False)

        self.channel_names = list(channel_names)
        self.n_channels = len(self.channel_names)
        self.rectangular_h, self.rectangular_w = rectangular_shape
        self.rectangular_shape = (self.rectangular_h, self.rectangular_w)

        # ---- create the data array once; idempotent on resume/clobber ----

        if "data" in self.root:
            # Re-opening an existing store: the channel axis (C) is fixed at
            # creation, so the requested channel list MUST match what the store
            # was built with.  Overwriting channel_names attrs against a
            # mismatched data array would silently corrupt the store, so refuse
            # loudly instead.  (Deletion is deliberately left to the user --
            # pipeline code never deletes S3 prefixes.)
            existing_names = list(self.root.attrs.get("channel_names", []))
            existing_c = int(self.root["data"].shape[0])
            if existing_names and existing_names != self.channel_names:
                raise ValueError(
                    f"Channel mismatch for existing zarr store at {path}.\n"
                    f"  store was built with : {existing_names}\n"
                    f"  requested            : {self.channel_names}\n"
                    "Refusing to write -- this would corrupt the store.  "
                    "Either delete the store manually, or write under a "
                    "different run_id."
                )
            if existing_c != self.n_channels:
                raise ValueError(
                    f"Channel-count mismatch for existing zarr store at {path}: "
                    f"data array has C={existing_c}, requested "
                    f"C={self.n_channels}.  Refusing to write.  Either delete "
                    "the store manually, or write under a different run_id."
                )
        else:
            self.root.create_array(
                "data",
                shape=(self.n_channels, self.rectangular_h, self.rectangular_w),
                # (1, h, w): one channel field per chunk — efficient for field-level reads
                chunks=(1, self.rectangular_h, self.rectangular_w),
                dtype="float32",
            )

        # Root-level metadata (safe to overwrite on re-open).
        self.root.attrs["channel_names"] = self.channel_names
        self.root.attrs["rectangular_shape"] = list(self.rectangular_shape)

    # ------------------------------------------------------------------

    def write_snapshot(self, data: np.ndarray, iteration: int) -> None:
        """
        Write the single global snapshot for this store.

        Each store holds exactly one timestep (the date is encoded in the store
        path), so all channels are written at once into a ``(C, H, W)`` array.
        The ``iteration`` attribute is written **last**, after the data chunks,
        so its presence is a reliable "fully written" marker for the existence
        planner -- a store from a crashed mid-write run will lack it.  Re-running
        (clobber) simply overwrites the array in place.

        Parameters
        ----------
        data : np.ndarray, shape (C, rectangular_h, rectangular_w), dtype float32
            All channels for this snapshot, already converted to rectangular
            lat/lon format via faces_dataset_to_latlon(). Channel order must
            match channel_names passed to __init__.
        iteration : int
            LLC4320 model iteration number for this snapshot. Stored as the
            ``iteration`` root attribute.

        Raises
        ------
        AssertionError
            If data.shape does not match (n_channels, rectangular_h, rectangular_w).
        """
        expected = (self.n_channels, self.rectangular_h, self.rectangular_w)
        assert data.shape == expected, (
            f"write_snapshot: expected shape {expected}, got {data.shape}. "
            f"Check that all channels were converted to rectangular lat/lon correctly."
        )

        # Write all channels; dtype cast ensures float32 regardless of input.
        self.root["data"][:] = data.astype(np.float32)
        # Completion marker -- written last (see method docstring).
        self.root.attrs["iteration"] = int(iteration)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class GlobalZarrDatasetReader:
    """
    Read-only accessor for a single global snapshot written by GlobalZarrDataset.

    Each store holds exactly one timestep (the date is encoded in the store
    path), so the accessors take no time index.

    Access patterns
    ---------------
    reader.get_snapshot()           -> np.ndarray (C, rectangular_h, rectangular_w)
    reader.get_channel_snapshot(c)  -> np.ndarray (rectangular_h, rectangular_w)
    reader.get_channel_snapshot("Theta")  -> np.ndarray (...)  # by name
    reader.iteration                -> int   # LLC4320 iteration for this snapshot
    """

    def __init__(self, bucket: str, folder: str, run_id: str, dataset_name: str, fs,
                 date_prefix: str | None = None):
        path = make_run_prefix(bucket, folder, run_id, dataset_name,
                               date_prefix=date_prefix)
        store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=store, mode="r", use_consolidated=False)

        self.data = self.root["data"]           # (C, rectangular_h, rectangular_w)
        self.rectangular_shape = tuple(self.root.attrs["rectangular_shape"])
        self.channel_names = list(self.root.attrs["channel_names"])
        self.iteration = self.root.attrs.get("iteration")

    # ------------------------------------------------------------------
    # Properties

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape)

    # ------------------------------------------------------------------
    # Core accessors

    def get_snapshot(self) -> np.ndarray:
        """Return all channels for this snapshot. Shape: (C, lat, lon)."""
        return self.data[:]

    def get_channel_snapshot(self, channel) -> np.ndarray:
        """
        Return a single channel. Shape: (H, W).

        Loads only 1 out of C chunks — much cheaper than get_snapshot()
        when you only need one channel.

        Parameters
        ----------
        channel : int or str
            Channel index, or name as found in channel_names.
        """
        if isinstance(channel, str):
            channel = self.channel_names.index(channel)
        return self.data[channel]
