"""Unit tests for the metadata reader."""
import fsspec
import pandas as pd

from dbof.cutout_dataset_creation.metadata import create_metadata_reader, MetadataWriter


def test_metadata_reader_concatenates_parts():
    fs = fsspec.filesystem("memory")
    base = "dbof/native_grid_dbof_training_data/run00/metadata"
    pd.DataFrame({"image_id": [b"a"], "center_lat": [1.0]}).to_parquet(
        f"{base}/part-1.parquet", filesystem=fs)
    pd.DataFrame({"image_id": [b"b"], "center_lat": [2.0]}).to_parquet(
        f"{base}/part-2.parquet", filesystem=fs)

    reader = create_metadata_reader("dbof", "native_grid_dbof_training_data", "run00", fs)
    df = reader.read()

    assert len(df) == 2
    assert set(df["center_lat"]) == {1.0, 2.0}


def test_metadata_writer_flushes_on_threshold_and_close():
    fs = fsspec.filesystem("memory")
    base = "b/f/run/metadata"
    w = MetadataWriter(base, flush_every=2, fs=fs)

    w.add({"image_id": b"a", "x": 1})
    assert fs.glob(f"{base}/*.parquet") == []           # below threshold, not flushed
    w.add({"image_id": b"b", "x": 2})
    assert len(fs.glob(f"{base}/*.parquet")) == 1        # flushed at threshold
    w.add({"image_id": b"c", "x": 3})
    w.close()                                            # flushes remainder

    parts = fs.glob(f"{base}/*.parquet")
    assert len(parts) == 2
    df = pd.read_parquet(parts, filesystem=fs)
    assert len(df) == 3 and set(df["x"]) == {1, 2, 3}
