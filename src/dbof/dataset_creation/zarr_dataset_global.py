"""
zarr_dataset_global.py
----------------------
Zarr writer/reader for global LLC4320 snapshots in the LLC compact format.

The LLC4320 model output lives on 13 curvilinear faces (the "tiled" format).
ecco_v4_py.llc_tiles_to_compact() stitches those 13 faces into a single coherent
2D array (the "compact" format) without any geographic interpolation. Values are
pixel-shifted and some faces are rotated so they tile correctly, but you remain on
the native LLC grid. This is distinct from resample_to_latlon, which would
interpolate to an equal-degree geographic grid.

This writer:
  - Stores full global snapshots in LLC compact format
  - Writes sequentially, one timestep at a time
  - Uses the LLC4320 iteration number as the natural time coordinate
  - Stores compact grid shape and channel names as Zarr attributes

Storage layout
--------------
  data  : float32, shape (T, C, compact_h, compact_w)  — all channels, all timesteps
  time  : int64,   shape (T,)                           — LLC4320 iteration numbers

Root group attributes
---------------------
  channel_names : list[str]        — ordered channel labels, matching axis C
  compact_shape : [compact_h, compact_w]  — 2D shape of each compact global field

Chunk strategy
--------------
  data chunks are (1, 1, compact_h, compact_w): one channel-slice of one timestep
  per chunk. This matches the expected access pattern (reading one full global
  field at a time).

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

def make_run_prefix(bucket: str, folder: str, run_id: str, dataset_name: str) -> str:
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
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
        S3 path components — same convention as the patch ZarrDataset.
    fs : fsspec AbstractFileSystem
        Async S3 filesystem returned by create_s3_filesystems().
    channel_names : list[str]
        Ordered channel names, e.g. ["Eta", "Salt", "Theta", "U", "V", "W", "log_gradb"].
        Length must match axis C of every array passed to write_snapshot().
    compact_shape : tuple[int, int]
        (compact_h, compact_w) — the 2D shape of a single compact global field
        as returned by ecco_v4_py.llc_tiles_to_compact(). Determined once via a
        dry-run in main() before the writer is constructed.
    """

    def __init__(
        self,
        bucket: str,
        folder: str,
        run_id: str,
        dataset_name: str,
        fs,
        channel_names: list,
        compact_shape: tuple,
    ):
        path = make_run_prefix(bucket, folder, run_id, dataset_name)
        self.store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="a")

        self.channel_names = list(channel_names)
        self.n_channels = len(self.channel_names)
        self.compact_h, self.compact_w = compact_shape

        # ---- create arrays once on first run; idempotent on resume ----

        if "data" not in self.root:
            self.root.create_array(
                "data",
                shape=(0, self.n_channels, self.compact_h, self.compact_w),
                # (1, 1, h, w): one channel-slice per chunk — efficient for field-level reads
                chunks=(1, 1, self.compact_h, self.compact_w),
                dtype="float32",
            )

        if "time" not in self.root:
            self.root.create_array(
                "time",
                shape=(0,),
                chunks=(1,),
                dtype="int64",
            )

        # Root-level metadata (safe to overwrite on resume).
        self.root.attrs["channel_names"] = self.channel_names
        self.root.attrs["compact_shape"] = list(compact_shape)

        # Resume from wherever the last run finished.
        self.t_index = int(self.root["data"].shape[0])

    # ------------------------------------------------------------------

    def write_snapshot(self, data: np.ndarray, iteration: int) -> None:
        """
        Append one global snapshot to the store.

        Parameters
        ----------
        data : np.ndarray, shape (C, compact_h, compact_w), dtype float32
            All channels for this timestep, already compacted via
            ecco_v4_py.llc_tiles_to_compact(). Channel order must match
            channel_names passed to __init__.
        iteration : int
            LLC4320 model iteration number for this snapshot. Stored as
            the time coordinate so snapshots can be looked up later.

        Raises
        ------
        AssertionError
            If data.shape does not match (n_channels, compact_h, compact_w).
        """
        expected = (self.n_channels, self.compact_h, self.compact_w)
        assert data.shape == expected, (
            f"write_snapshot: expected shape {expected}, got {data.shape}. "
            f"Check that all channels were compacted and stacked correctly."
        )

        t = self.t_index

        # Grow the time axis by one slot.
        self.root["data"].resize((t + 1, self.n_channels, self.compact_h, self.compact_w))
        self.root["time"].resize((t + 1,))

        # Write. dtype cast here ensures float32 regardless of interpolation output.
        self.root["data"][t] = data.astype(np.float32)
        self.root["time"][t] = int(iteration)

        self.t_index += 1

    @property
    def n_timesteps(self) -> int:
        return int(self.root["data"].shape[0])


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class GlobalZarrDatasetReader:
    """
    Read-only accessor for datasets written by GlobalZarrDataset.

    Access patterns
    ---------------
    reader.get_snapshot(t)          -> np.ndarray (C, compact_h, compact_w)
    reader.get_channel(c)           -> np.ndarray (T, compact_h, compact_w)
    reader.get_channel("Theta")     -> np.ndarray (T, compact_h, compact_w)  # by name
    reader.iteration_to_index(it)   -> int t
    reader[t]                       -> np.ndarray (C, compact_h, compact_w)
    len(reader)                     -> int T
    """

    def __init__(self, bucket: str, folder: str, run_id: str, dataset_name: str, fs):
        path = make_run_prefix(bucket, folder, run_id, dataset_name)
        store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=store, mode="r")

        self.data = self.root["data"]           # (T, C, compact_h, compact_w)
        self.time = self.root["time"]           # (T,)
        self.compact_shape = tuple(self.root.attrs["compact_shape"])
        self.channel_names = list(self.root.attrs["channel_names"])

        # Built lazily by iteration_to_index().
        self._iter_to_t = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def n_timesteps(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape)

    # ------------------------------------------------------------------
    # Core accessors

    def get_snapshot(self, t: int) -> np.ndarray:
        """Return all channels for timestep index t. Shape: (C, lat, lon)."""
        return self.data[t]

    def get_channel(self, channel) -> np.ndarray:
        """
        Return all timesteps for a single channel. Shape: (T, lat, lon).

        Parameters
        ----------
        channel : int or str
            Channel index, or name as found in channel_names.
        """
        if isinstance(channel, str):
            channel = self.channel_names.index(channel)
        return self.data[:, channel, :, :]

    def iteration_to_index(self, iteration: int) -> int:
        """Map an LLC4320 iteration number to its time axis index t."""
        if self._iter_to_t is None:
            self._iter_to_t = {int(v): i for i, v in enumerate(self.time[:])}
        return self._iter_to_t[iteration]

    def __getitem__(self, t: int) -> np.ndarray:
        return self.get_snapshot(t)

    def __len__(self) -> int:
        return self.n_timesteps
