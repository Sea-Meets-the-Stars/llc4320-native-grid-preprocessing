from pathlib import PurePosixPath
import pandas as pd
import fsspec
import threading
import uuid

#Factory
def create_metadata_writer(bucket: str, folder: str, run_id: str, fs_sync=None, flush_every: int = 10_000):
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")

    base = str(PurePosixPath(bucket, folder, run_id, "metadata"))
    return MetadataWriter(base, flush_every=flush_every, fs=fs_sync)

#Multi thread safe writer NOT process safe
class MetadataWriter:
    def __init__(self, path, flush_every=10_000, fs=None):
        self.path = path
        self.flush_every = flush_every
        self.buffer = []

        if fs is None:
            self.fs, _ = fsspec.core.url_to_fs(path)
        else :
            self.fs = fs

        self.lock = threading.Lock()

    def add(self, meta: dict):
        # add one record
        with self.lock:
            self.buffer.append(meta)
            if len(self.buffer) >= self.flush_every:
                self._flush_locked()

    def _flush_locked(self):
        #Write buffered metadata to Parquet

        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        fname = f"part-{uuid.uuid4().hex}.parquet"
        full_path = f"{self.path.rstrip('/')}/{fname}"

        df.to_parquet(
            full_path,
            engine="pyarrow",
            filesystem=self.fs,
        )

        self.buffer.clear()

    def close(self):
        with self.lock:
            self._flush_locked()


# Factory
def create_metadata_reader(bucket: str, folder: str, run_id: str, fs_sync):
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    metadata_glob = f"{PurePosixPath(bucket, folder, run_id, 'metadata')}/*.parquet"
    return MetadataReader(metadata_glob, fs=fs_sync)


class MetadataReader:
    """Reads all parquet parts written for a run into a single DataFrame."""
    def __init__(self, metadata_glob, fs):
        self.metadata_glob = metadata_glob
        self.fs = fs

    def read(self) -> pd.DataFrame:
        files = self.fs.glob(self.metadata_glob)
        if not files:
            raise FileNotFoundError(f"No metadata parquet files at {self.metadata_glob}")
        return pd.read_parquet(files, filesystem=self.fs)