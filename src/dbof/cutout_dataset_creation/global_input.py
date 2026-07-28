"""S3 input-source resolution and loading for generate-global output."""
import dbof.cutout_dataset_creation.config as config
import xarray as xr
from dbof.global_dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader


def resolve_input_locations(input_cfg: config.InputConfig):
    """Resolve S3 locations for the generate-global input from config (no I/O)."""
    base = f"s3://{input_cfg.bucket.strip('/')}/{input_cfg.folder.strip('/')}"
    grid = input_cfg.grid_access
    grid_uri = f"s3://{grid.bucket.strip('/')}/{grid.folder.strip('/')}/{grid.dataset_name.strip('/')}"
    return base, grid_uri


def resolve_date_prefixes(input_cfg: config.InputConfig, fs) -> list[str]:
    """Date prefixes to process: the configured list, or every date folder
    discovered under the input location when none are configured.

    """
    if input_cfg.date_prefixes:
        return list(input_cfg.date_prefixes)
    base = f"{input_cfg.bucket.strip('/')}/{input_cfg.folder.strip('/')}"
    entries = fs.ls(base, detail=True)
    return sorted(
        e["name"].rstrip("/").split("/")[-1]
        for e in entries
        if e.get("type") == "directory"
    )


def _open_subset_readers(input_cfg: config.InputConfig, date_prefix: str, fs, fs_sync) -> dict:
    """Open a GlobalZarrDatasetReader for each .zarr subset store in a date folder.

    `fs_sync` lists the stores; `fs` (async) backs the readers.
    """
    date_dir = f"{input_cfg.bucket.strip('/')}/{input_cfg.folder.strip('/')}/{date_prefix}"
    stores = [
        e["name"].rstrip("/").split("/")[-1]
        for e in fs_sync.ls(date_dir, detail=True)
        if e.get("type") == "directory" and e["name"].rstrip("/").endswith(".zarr")
    ]
    return {
        store: GlobalZarrDatasetReader(
            bucket=input_cfg.bucket,
            folder=input_cfg.folder,
            run_id="",
            dataset_name=store,
            date_prefix=date_prefix,
            fs=fs,
        )
        for store in stores
    }


def available_channels(input_cfg: config.InputConfig, date_prefix: str, fs, fs_sync) -> set:
    """Union of channel_names across all subset zarr stores in a date folder."""
    readers = _open_subset_readers(input_cfg, date_prefix, fs, fs_sync)
    channels = set()
    for reader in readers.values():
        channels.update(reader.channel_names)
    return channels


def open_feature_readers(input_cfg: config.InputConfig, date_prefix: str,
                         feature_channels, fs, fs_sync) -> dict:
    """Map each requested feature channel to the reader of the subset store holding it."""
    readers = _open_subset_readers(input_cfg, date_prefix, fs, fs_sync)
    channel_to_reader = {}
    for reader in readers.values():
        for ch in reader.channel_names:
            if ch in feature_channels and ch not in channel_to_reader:
                channel_to_reader[ch] = reader
    return channel_to_reader


def verify_feature_channels(input_cfg: config.InputConfig, date_prefix: str,
                            feature_channels, fs, fs_sync) -> None:
    """Raise ValueError if any requested feature_channel is absent from the date folder."""
    available = available_channels(input_cfg, date_prefix, fs, fs_sync)
    missing = [c for c in feature_channels if c not in available]
    if missing:
        raise ValueError(
            f"Requested feature_channels not found in {date_prefix}: {missing}. "
            f"Available: {sorted(available)}"
        )


def verify_required_channels(input_cfg: config.InputConfig, date_prefix: str, fs, fs_sync) -> None:
    """Ensure the always-required channels (gradb2, SIarea) are present in the snapshot's data."""
    verify_feature_channels(input_cfg, date_prefix, config.REQUIRED_FEATURE_CHANNELS, fs, fs_sync)


def load_snapshot_features(input_cfg: config.InputConfig, date_prefix: str,
                           feature_channels, fs, fs_sync) -> xr.Dataset:
    """Read the requested feature channels for one snapshot into an xr.Dataset
    with dims (j, i).  Raises ValueError if any requested channel is missing."""
    readers = open_feature_readers(input_cfg, date_prefix, feature_channels, fs, fs_sync)
    missing = [c for c in feature_channels if c not in readers]
    if missing:
        raise ValueError(f"Requested feature_channels not found in {date_prefix}: {missing}")
    data_vars = {
        ch: (("j", "i"), readers[ch].get_channel_snapshot(ch))
        for ch in feature_channels
    }
    return xr.Dataset(data_vars)
