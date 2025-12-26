from pathlib import Path

class MetadataWriter:
    def __init__(self, path, flush_every=10_000):
        self.path = Path(path)
        self.flush_every = flush_every
        self.buffer = []

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

        if self.path.exists():
            meta_df = pd.read_parquet("metadata.parquet")
            appended_df = pd.concat([meta_df, df], ignore_index=True)

            appended_df.to_parquet(
                self.path,
                engine="pyarrow"
            )

        else:
            df.to_parquet(
                self.path,
                engine="pyarrow"
            )

        self.buffer.clear()

    def close(self):
        #Flush remaining metadata (call at end).
        self.flush()