"""Integration test: global_input against the real S3 example directory.

Requires access to the NRP S3 endpoint named in the example config.
Skips gracefully when the store is unreachable (offline / no credentials).
"""
from pathlib import Path

import pytest

from dbof.cutout_dataset_creation.config import load_config, InputConfig
from dbof.io.filesystems import create_s3_filesystems
from dbof.cutout_dataset_creation.global_input import (
    resolve_date_prefixes,
    available_channels,
    verify_feature_channels,
)

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "configs/cutouts/run/run_from_globals_example.yaml"
DATE = "20121109_120000"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "output"


@pytest.fixture(scope="module")
def example_input():
    cfg = load_config(str(EXAMPLE_CONFIG))
    fs, fs_sync = create_s3_filesystems(cfg.input.s3_endpoint)
    base = f"{cfg.input.bucket.strip('/')}/{cfg.input.folder.strip('/')}"
    try:
        fs_sync.ls(base, detail=True)
    except Exception as exc:
        pytest.skip(f"input store unreachable ({base}): {exc}")
    return cfg, fs, fs_sync


def test_discovers_real_date_prefix(example_input):
    cfg, fs, fs_sync = example_input
    discover_cfg = InputConfig(
        folder=cfg.input.folder, bucket=cfg.input.bucket, s3_endpoint=cfg.input.s3_endpoint,
    )  # no date_prefixes -> force discovery
    assert DATE in resolve_date_prefixes(discover_cfg, fs_sync)


def test_available_channels_contains_known_fields(example_input):
    cfg, fs, fs_sync = example_input
    chans = available_channels(cfg.input, DATE, fs, fs_sync)
    assert {"Theta", "Salt", "Eta", "U", "V", "W"}.issubset(chans)
    assert "gradb2" in chans


def test_verify_present_and_absent(example_input):
    cfg, fs, fs_sync = example_input
    verify_feature_channels(cfg.input, DATE, ["Theta", "gradb2"], fs, fs_sync)  # present -> no raise
    with pytest.raises(ValueError, match="relative_vorticity"):
        verify_feature_channels(cfg.input, DATE, ["relative_vorticity"], fs, fs_sync)


def test_land_mask_renders(example_input):
    """Build the stitched halo land mask from the real grid and render it.

    Mirrors set_up_grid_data_and_masks (grid reader + generate_halo_land_mask)
    without importing the CLI.  Heavy: full-grid fast-marching.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np

    from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
    from dbof.preprocessing import native_grid_masks
    from dbof.plotting.field_cmaps import load_field_cmaps
    from dbof.plotting.global_maps import plot_global_field

    cfg, fs, fs_sync = example_input

    grid = cfg.input.grid_access
    ds_grid = GlobalGridZarrReader(
        bucket=grid.bucket, folder=grid.folder, dataset_name=grid.dataset_name, fs=fs,
    ).to_dataset_lazy()

    land_mask = native_grid_masks.generate_halo_land_mask(
        ds_grid, cfg.output.target_km_res, stitched=True,
    )
    assert land_mask.shape == ds_grid["XC"].shape
    assert land_mask.any() and not land_mask.all()

    step = 20
    XC = np.asarray(ds_grid["XC"][::step, ::step])
    YC = np.asarray(ds_grid["YC"][::step, ::step])
    arr = land_mask[::step, ::step].astype("float32")

    cmap_cfg, diverging = load_field_cmaps()
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    im, _ = plot_global_field(ax, XC, YC, arr, "land_halo_mask", cmap_cfg,
                              diverging_cmaps=diverging, transform=ccrs.PlateCarree())
    ax.set_title("Stitched halo land mask (retained = 1)")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out = ARTIFACTS_DIR / "land_halo_mask.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    assert out.exists() and out.stat().st_size > 0
    print(f"\nland mask figure written: {out}")


def test_theta_field_renders(example_input):
    """Read Theta for the snapshot and render it to a PNG via plot_global_field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np

    from dbof.cutout_dataset_creation.global_input import load_snapshot_features
    from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
    from dbof.plotting.field_cmaps import load_field_cmaps
    from dbof.plotting.global_maps import plot_global_field

    cfg, fs, fs_sync = example_input

    ds = load_snapshot_features(cfg.input, DATE, ["Theta"], fs, fs_sync)
    assert "Theta" in ds.data_vars

    grid = cfg.input.grid_access
    gds = GlobalGridZarrReader(
        bucket=grid.bucket, folder=grid.folder, dataset_name=grid.dataset_name, fs=fs,
    ).to_dataset_lazy(["XC", "YC"])

    step = 20
    XC = np.asarray(gds["XC"][::step, ::step])
    YC = np.asarray(gds["YC"][::step, ::step])
    arr = np.asarray(ds["Theta"][::step, ::step])

    cmap_cfg, diverging = load_field_cmaps()
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    im, label = plot_global_field(ax, XC, YC, arr, "Theta", cmap_cfg,
                                  diverging_cmaps=diverging, transform=ccrs.PlateCarree())
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.7,
                 label=label, orientation="horizontal")
    ax.set_title("Theta")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out = ARTIFACTS_DIR / "theta.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    assert out.exists() and out.stat().st_size > 0
    print(f"\ntheta figure written: {out}")
