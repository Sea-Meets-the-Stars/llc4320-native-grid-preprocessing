"""Read a generated cutout dataset (zarr images + parquet metadata) as one object."""
from dataclasses import dataclass

import dask.array as da
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean.cm as cmo

import dbof.cutout_dataset_creation.config as config
import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.cutout_dataset_creation.metadata as metadata
from dbof.io.filesystems import create_s3_filesystems
from dbof.plotting.field_cmaps import load_field_cmaps
from dbof.plotting.global_maps import make_field_norm


def _to_str(v):
    return v.decode("ascii") if isinstance(v, (bytes, bytearray, np.bytes_)) else str(v)


@dataclass
class CutoutDataset:
    """Valid cutouts with images row-aligned to metadata: images[k] <-> metadata.iloc[k]."""
    images: da.Array          # (N, C, H, W), lazy
    metadata: pd.DataFrame
    channel_names: list


def load_cutout_dataset(config_path, run_id=None) -> CutoutDataset:
    access_cfg = config.load_data_access_config(config_path)
    # run_id overrides the config's run_id when given, so one config serves many runs.
    run_id = run_id if run_id is not None else access_cfg.run_id
    fs, fs_sync = create_s3_filesystems(access_cfg.s3_endpoint)
    reader = zarr_dataset.ZarrDatasetReader(
        bucket=access_cfg.bucket, folder=access_cfg.folder, run_id=run_id,
        dataset_name=access_cfg.dataset_name, fs=fs,
    )
    meta_df = metadata.create_metadata_reader(
        access_cfg.bucket, access_cfg.folder, run_id, fs_sync).read()

    images_da, image_ids_da, _ = reader.full_dataset_as_dask()

    # image_ids is a small (N,) array -- materialize it once and index with numpy.
    # (images_da stays lazy; we only fancy-index it at the end.)
    image_ids = np.asarray(image_ids_da)

    meta = meta_df.copy()
    meta["image_id"] = meta["image_id"].map(_to_str)
    meta = meta.set_index("image_id")

    # Keep written (non-empty id) images that also have a metadata row, in zarr order.
    keep_pos, keep_ids = [], []
    for pos in np.flatnonzero(image_ids != b""):
        iid = _to_str(image_ids[pos])
        if iid in meta.index:
            keep_pos.append(int(pos))
            keep_ids.append(iid)

    return CutoutDataset(
        images=images_da[keep_pos],
        metadata=meta.loc[keep_ids].reset_index(),
        channel_names=reader.channel_names,
    )


def plot_random_cutouts(dataset, n=5, seed=None):
    """Build a figure of n cutouts sampled without replacement, all channels,
    using the per-feature colormaps from dbof.plotting.

    Returns (fig, selected_metadata).  The caller decides whether to show
    (plt.show) or save (fig.savefig).  dataset is a CutoutDataset whose images
    are already hole-free and row-aligned to metadata, so sampling indices over
    [0, N) stays aligned -- the same indices select both images and metadata.
    """
    total = dataset.images.shape[0]
    n = min(n, total)
    sel = np.random.default_rng(seed).choice(total, size=n, replace=False)

    imgs = np.asarray(dataset.images[list(sel)])          # (n, C, H, W)
    meta = dataset.metadata.iloc[sel].reset_index(drop=True)
    C = imgs.shape[1]
    channel_names = dataset.channel_names or [f"ch{c}" for c in range(C)]
    cmap_cfg, diverging = load_field_cmaps()

    fig, axes = plt.subplots(n, C, figsize=(2.4 * C, 2.4 * n), squeeze=False)
    for r in range(n):
        for c in range(C):
            ax = axes[r][c]
            field = channel_names[c]
            cmap_name, label = cmap_cfg.get(field, ("viridis", field))
            cmap = getattr(cmo, cmap_name, plt.cm.viridis)
            arr = imgs[r, c]
            try:
                norm = make_field_norm(arr, cmap_name, diverging_cmaps=diverging)
            except Exception:
                norm = None
            ax.imshow(arr, origin="lower", cmap=cmap, norm=norm)
            if r == 0:
                ax.set_title(label, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig, meta
