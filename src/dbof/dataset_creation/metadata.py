from pathlib import PurePosixPath
import pandas as pd
import fsspec
import threading
import uuid

#Factory
def create_metadata_writer(bucket: str, folder: str, run_id: str, fs_sync=None, flush_every: int = 10_000):
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")

    base = f"s3://{str(PurePosixPath(bucket, folder, run_id, "metadata"))}"
    return MetadataWriter(str(base), flush_every=flush_every, fs=fs_sync)

#Multi thread safe writer
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

        # if not os.path.exists(meda_data_file_path):
        #     pd.DataFrame(columns=metadata_cols).to_parquet(meda_data_file_path)

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


        # if self.fs.exists(self.path):
        #     old = pd.read_parquet(self.path, filesystem=self.fs)
        #     df = pd.concat([old, df], ignore_index=True)

        df.to_parquet(
            full_path,
            engine="pyarrow",
            filesystem=self.fs,
        )

        self.buffer.clear()

    def close(self):
        with self.lock:
            self._flush_locked()