from pathlib import Path
import pandas as pd
import fsspec

class MetadataWriter:
    def __init__(self, path, flush_every=10_000, fs=None):
        self.path = path
        self.flush_every = flush_every
        self.buffer = []

        if fs is None:
            self.fs, _ = fsspec.core.url_to_fs(path)

        # if not os.path.exists(meda_data_file_path):
        #     pd.DataFrame(columns=metadata_cols).to_parquet(meda_data_file_path)

    def add(self, meta: dict):
        # add one record
        self.buffer.append(meta)

        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        #Write buffered metadata to Parquet
        if not self.buffer:
            print("NO BUFFER")
            return

        df = pd.DataFrame(self.buffer)

        if self.fs.exists(self.path):
            old = pd.read_parquet(self.path, filesystem=self.fs)
            df = pd.concat([old, df], ignore_index=True)

        df.to_parquet(
            self.path,
            engine="pyarrow",
            filesystem=self.fs,
        )

        self.buffer.clear()

    def close(self):
        #Flush remaining metadata (call at end).
        self.flush()