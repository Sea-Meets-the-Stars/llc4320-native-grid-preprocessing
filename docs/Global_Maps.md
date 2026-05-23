# Global DBOF Maps

**Build global LLC4320 maps on the native curvilinear grid.**

This document describes the pipelines for turning raw LLC4320 model output
(13 curvilinear faces) into stitched 2D `(12960, 17280)` arrays written to
S3-backed Zarr stores, plus the NetCDF export step. The native grid geometry is
preserved — no interpolation to a regular lat/lon grid is performed.

---

## Pipelines at a glance

There are three `generate_global` pipelines, each suited to different input
sources and depth handling:

| Pipeline                  | Depth support           | Input source                       | Prerequisite               |
|---------------------------|-------------------------|------------------------------------|----------------------------|
| `generate_global.py`      | Surface only            | S3 timestep stores (from MIT)      | `transfer_llc4320.py`      |
| `generate_global_OSN.py`  | Surface only            | OSN kerchunk (direct)              | None                       |
| `generate_global_depth.py`| Surface + depth-aware   | S3 timestep stores (from MIT)      | `transfer_llc4320.py`      |

All three share the same output format: S3 Zarr stores shaped
`(T, C, 12960, 17280)` with chunk shape `(1, 1, 12960, 17280)`.

---

## 1. `generate_global.py` — Surface-only (S3 source)

Surface-only global maps. Reads raw LLC4320 fields from the OSN kerchunk
endpoint for core variables (`Theta`, `Salt`, `Eta`, `U`, `V`, `W`) and from
S3 timestep stores for additional variables not available via kerchunk
(e.g. `oceTAUX`, `SIarea`).

**The desired timestep must first be transferred from MIT using
`transfer_llc4320.py`** (see below) before any S3-sourced variables can be
read.

Config: `configs/global.yaml`. Console command: `generate-global`.

```bash
generate-global \
    --config configs/global.yaml \
    --subset kinematic \
    --run_id kinematic_$(date +%Y%m%d_%H%M%S)
```

Flags: `--subset`, `--run_id`, `--no-icemask` (ice mask off by default).

---

## 2. `generate_global_OSN.py` — Surface-only (OSN direct)

Surface-only global maps using inputs directly from the OSN S3 store via
kerchunk. Uses two OSN endpoints: the standard one for core variables and a
second (`llc_wind`) for wind stress and sea-ice variables (`KPPhbl`,
`PhiBot`, `oceTAUX`, `oceTAUY`, `SIarea`).

**No transfer step is required** — all data is read directly from OSN.
However, the wind/sea-ice endpoint has a limited date range, so not all
timesteps are available for all variables.

Config: `configs/global_OSN.yaml`. Console command: `generate-global-osn`.

Key differences from `generate_global.py`:

- Uses OSN-native iteration offsets (`osn_date_to_iteration`).
- Ice mask is on by default (`--no-icemask` to disable).
- No S3 timestep store fallback — all data comes from OSN.

---

## 3. `generate_global_depth.py` — Depth-aware pipeline

Fully-lazy Dask pipeline for depth-resolved diagnostics. Reads full 3D
(depth-resolved) data from S3 timestep stores, computes derived fields using
xgcm on dask arrays, reduces each field to a 2D surface using one or more
depth strategies, stitches the 13 LLC faces, and writes to S3 Zarr.

**Requires data transferred from the MIT Zarr store using
`transfer_llc4320.py`.**

Config: `configs/global_depth.yaml`. Console command: `generate-global-depth`.

```bash
generate-global-depth \
    --config configs/global_depth.yaml \
    --subset stratification
```

### Depth strategies (suffixes)

Each subset can request one or more depth strategies. A depth strategy
controls how the pipeline reduces a full-depth 3D field to a single 2D
layer for output. The available strategies are:

| Suffix      | Meaning                                                   |
|-------------|-----------------------------------------------------------|
| `sfc`       | Surface (k=0)                                             |
| `z25m`      | Nearest model level to 25 m (configurable, e.g. `z50m`)   |
| `mld`       | Value at the mixed-layer depth                            |
| `mld_mean`  | Thickness-weighted mean over 0 ≤ z ≤ MLD                 |

In the config, each subset lists its base channel names in
`compute_features_channels` and the desired depth variants in
`depth_suffixes`. The pipeline expands every base × suffix combination:

```yaml
compute_features_channels: [N2]
depth_suffixes: [sfc, z25m, mld, mld_mean]
# → channels: N2_sfc, N2_z25m, N2_mld, N2_mld_mean
```

Standalone diagnostics with no depth variants (e.g. `mixed_layer_depth`)
go in `extra_channels` and are appended unchanged.

### Surface vs. depth-aware within the depth pipeline

Subset entries can set `surface_only: true` to skip depth processing entirely.
These subsets (e.g. `surface_wind`, `icearea`) read only surface-level data
from the S3 timestep stores — they use the depth pipeline's S3 data access but
not its 3D computation machinery.

---

## Subsets

A **subset** is a named group of related calculated fields/properties that are
computed and written together as a single Zarr store. Each subset has its own
`dataset_name` (e.g. `stratification.zarr`) and channel list defined in the
config YAML.

Subsets exist because the full set of derived fields is large — grouping
related fields together lets you generate and export only what you need.

### Surface pipeline subsets (`generate_global.py` / `generate_global_OSN.py`)

| Subset              | Fields                                                                                      |
|---------------------|---------------------------------------------------------------------------------------------|
| `native_fields`     | `Theta`, `Salt`, `Eta`, `U`, `V`, `W`                                                      |
| `frontal_structure` | `gradb2`, `gradsalt2`, `gradtheta2`, `gradeta2`, `gradrho2`, `turner_angle`                  |
| `kinematic`         | `relative_vorticity`, `strain_n`, `strain_s`, `strain_mag`, `divergence`, `coriolis_f`, `rossby_number`, `okubo_weiss` |
| `frontogenesis`     | `frontogenesis_tendency`, `frontogenesis_geo`, `frontogenesis_ageo`, `ug`, `vg`              |

### Depth pipeline subsets (`generate_global_depth.py`)

| Subset              | Base fields                                        | Depth suffixes              | Extra channels                          |
|---------------------|----------------------------------------------------|-----------------------------|-----------------------------------------|
| `stratification`    | `N2`                                               | sfc, z25m, mld, mld_mean   | `mixed_layer_depth`, `ml_heat_content`  |
| `vertical_shear`    | `vertical_shear`, `Ri`                             | mld                         |                                         |
| `mixing_parameters` | `Fr`, `Ro`, `Bu`                                   | mld                         |                                         |
| `ertel_pv`          | `ertel_pv`, `ertel_pv_vertical`, `ertel_pv_tilt`  | mld                         |                                         |
| `buoyancy_fluxes`   | `uB`, `vB`, `wB`                                   | mld                         |                                         |
| `energetics`        | `KE`                                               | sfc, z25m, mld, mld_mean   |                                         |
| `frontal_structure` | `gradb2`, `gradtheta2`, `gradsalt2`, `gradrho2`, `gradeta2`, `turner_angle` | sfc |                                         |
| `kinematic`         | `relative_vorticity`, `strain_n`, `strain_s`, `strain_mag`, `divergence`, `okubo_weiss` | sfc |                           |
| `frontogenesis`     | `frontogenesis_tendency`, `frontogenesis_geo`, `frontogenesis_ageo`, `ug`, `vg` | sfc |                                   |
| `native_fields`     | `Theta`, `Salt`, `Eta`, `U`, `V`, `W`             | sfc                         |                                         |
| `surface_wind`      | *(surface_only)* `wind_stress_curl`, `ekman_pumping`, `u_ekman`, `v_ekman` + model fields `oceTAUX`, `oceTAUY`, `oceQnet` | — | |
| `icearea`           | *(surface_only)* model field `SIarea`              | —                           |                                         |

---

## Batch driver: `run_all_subsets.py`

The batch driver (`dbof.cli.run_all_subsets`) automates the two-phase
workflow across all subsets and configs:

1. **Phase 1 — Generate:** calls `generate_global_depth.main()` for each
   subset, producing Zarr stores on S3.
2. **Phase 2 — Export:** calls `zarr_to_netcdf.main()` once per channel,
   writing individual NetCDF files.

### NetCDF output layout

```
{netcdf_base}/{run_id}/{date_prefix}/LLC4320_{date}_{channel}_{run_id}.nc
```

where `netcdf_base` defaults to `/mnt/tank/Oceanography/data/OGCM/LLC/Fronts`,
and `date` is formatted as `2012-11-09T12_00_00`.

Example:
```
/mnt/tank/Oceanography/data/OGCM/LLC/Fronts/global_depth_test00/20121109_120000/LLC4320_2012-11-09T12_00_00_N2_sfc_global_depth_test00.nc
```

### CLI usage

```bash
# Generate + export all subsets:
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml

# Only specific subsets:
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml \
    --subsets stratification native_fields

# Export only (assumes Zarr stores already exist):
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml --export-only

# Generate only (skip NetCDF export):
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml --generate-only

# Override run_id:
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml --run-id my_run_01

# Dry run:
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml --dry-run

# With ice masking:
python -m dbof.cli.run_all_subsets --config configs/global_depth.yaml --ice-mask
```

### Ice masking

Ice masking is **not** applied during the generate step — the Zarr stores
contain unmasked data. Instead, ice masking is applied as a post-processing
step during the NetCDF export (Phase 2), via `zarr_to_netcdf.py`. When
`--ice-mask` is passed, the exporter reads `SIarea` from the `icearea.zarr`
store (same bucket/folder/run_id/date_prefix) and sets all points where
`SIarea > 0` to NaN in the exported NetCDF. The `icearea` subset itself is
never self-masked.

---

## Preprocessing: `transfer_llc4320.py`

Both `generate_global.py` and `generate_global_depth.py` read from S3
timestep stores. These stores must be created by transferring data from the
MIT Zarr store using `transfer_llc4320.py`.

Console command: `transfer-timestep`. Config: `configs/transfer.yaml`.

The transfer writes two stores per timestep:

- **Static grid:** `s3://{bucket}/{folder}/grid.zarr` (one-time).
- **Timestep data:** `s3://{bucket}/{folder}/{YYYYMMDDTHH}.zarr` (per date,
  includes full-depth 3D fields).

Only `generate_global_OSN.py` does not require this step — it reads directly
from OSN kerchunk endpoints.

---

## Static grid: `generate_grid_global.py`

Extracts LLC4320 static grid variables, stitches the 13 faces, and writes to
`s3://{bucket}/{folder}/{dataset_name}` (default `llc4320_grid.zarr`). Run
once per S3 folder. No `run_id` — the grid store is shared across runs.

Console command: `generate-global-grid-zarr`.

Variables: `XC`, `YC`, `Depth`, `hFacC`, `rA`, `rAz`, `SN`, `CS`, `dxC`,
`dyG`, `dyC`, `dxG` — all shape `(12960, 17280)`.

---

## NetCDF export: `zarr_to_netcdf.py`

Exports S3 Zarr data to local NetCDF. Two modes:

- **`snapshots`** — per-timestep data. Select timesteps with `--dates`,
  `--iterations`, or `--indices`. Select channels with `--channel`.
- **`grid`** — the static grid store.

Snapshot files do not include lat/lon or grid spacings — load the grid
NetCDF alongside them.

```bash
# Snapshots
zarr-to-netcdf --mode snapshots \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof --folder properties \
    --run-id global_depth_test00 \
    --dates '2012-11-09 12:00:00' \
    --output-dir /path/to/output

# Grid
zarr-to-netcdf --mode grid \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof --folder properties \
    --grid-dataset-name llc4320_grid.zarr \
    --output-dir /path/to/output
```

---

## Zarr output layout

All pipelines write to the same S3 path structure:

```
s3://{bucket}/{folder}/{run_id}/{date_prefix}/{dataset_name}
```

Each store is shaped `(T, C, 12960, 17280)` with one chunk per
channel-slice per timestep.

---

## Iteration / date conventions

```
LLC4320_START_DATE    = 2011-09-13 00:00:00 UTC
LLC4320_TIMESTEP_SECS = 25
iteration             = round((date - start) / 25 s)
```

- Output cadence on OSN: hourly (144 iterations per saved hour).
- First valid post-spin-up iteration: **10368** (`FIRST_WIND_RECORD_OFFSET`).
- Last available iteration: **1,495,008**.
- All CLI / YAML date strings use `YYYY-MM-DD HH:MM:SS` (UTC).

---

## Python readers

```python
from dbof.dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader
from dbof.dataset_creation.zarr_grid_global import GlobalGridZarrReader
from dbof.utils.filesystems import get_filesystem

fs = get_filesystem(s3_endpoint, anon=False)

reader = GlobalZarrDatasetReader(bucket, folder, run_id, dataset_name, fs)
snap   = reader.get_snapshot(t=0)                          # (C, H, W)
field  = reader.get_channel_snapshot(t=0, channel="N2_sfc") # (H, W)
t_idx  = reader.iteration_to_index(iteration=1463616)

grid   = GlobalGridZarrReader(bucket, folder, dataset_name="llc4320_grid.zarr", fs=fs)
lon, lat = grid.lon, grid.lat
land_mask = grid.land_mask
ds_grid  = grid.to_dataset()
```

---

## Known issues

### `utils.faces_to_latlon.faces_dataset_to_latlon` re-implementation

`llcreader.llcmodel` provides a function for stitching the 13 LLC faces into a
rectangular array, but two lines in the upstream implementation are
incompatible with recent xarray versions. The function is re-implemented
locally in `utils.faces_to_latlon`.

### Asyncio cleanup traceback

A harmless `RuntimeError: Event loop is closed` from `s3fs` / `aiohttp` can
appear when `zarr-to-netcdf` exits. It is a teardown race and does not affect
output files.

### Grid store coordinate variables

`faces_dataset_to_latlon` promotes `XC` / `YC` to coordinates. The current
`GlobalGridZarrWriter` calls `ds.reset_coords()` and filters to 2D arrays
before writing. Older stores written before this fix should be regenerated.

### Geostrophic velocity near the equator

`ug`, `vg` are `±(g/f) ∂η/∂{x,y}`; as `f → 0` at the equator both diverge
in a narrow band. Physically correct, but may require masking downstream.
