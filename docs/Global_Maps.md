# Global DBOF Maps

**Build global maps of ocean variables from LLC4320**

This workflow converts raw LLC4320 model output from its native 13-face layout into a single stitched 2D lat/lon array of shape `(12960, 17280)`, while preserving the native curvilinear grid geometry. *Note: Grid interpolation is not performed.* The results are written to S3-backed Zarr stores. This is done separately for model output variables (SST, SSS, SSH, etc.) and static spatial metadata (coordinates, masks). Utilities to export selected snapshots from Zarr to NetCDF are also provided.

---

## Overview

The global pipeline consists of five source modules and two configuration files:

| Module | Role |
|---|---|
| `cli.generate_fronts_global` | Data generation: loads raw LLC4320 faces, computes derived features, stitches faces, writes timestep snapshots to S3 Zarr |
| `cli.generate_grid_global` | One-time run: extracts static LLC4320 grid variables from raw data and writes them to a separate S3 Zarr store |
| `cli.zarr_to_netcdf` | Exports selected snapshots or the full grid from Zarr to NetCDF, for use in external tools |
| `dataset_creation.zarr_dataset_global` | Low-level Zarr writer (`GlobalZarrDataset`) and reader (`GlobalZarrDatasetReader`) for snapshot data |
| `dataset_creation.zarr_grid_global` | Low-level Zarr writer (`GlobalGridZarrWriter`) and reader (`GlobalGridZarrReader`) for the static grid |

Configuration is split into two files:

| Config file | Purpose |
|---|---|
| `configs/global.yaml` | Controls data *generation* (time range, features, S3 output path, concurrency) |
| `configs/data_access/global.yaml` | Controls data *access* (S3 path to a specific generated run, feature channel order, grid store path) *note: run_id is not automatically updated to match that from generation and must be entered manually*|

---

## LLC4320 Grid and Coordinate System

LLC4320 output is stored across **13 curvilinear faces**. The stitching of the 13-faces into a 2D grid is performed by `utils.faces_to_latlon.faces_dataset_to_latlon()`, a local re-implementation of the equivalent function in `llcreader.llcmodel`. See the [Known Issues](#known-issues) section for why this was re-implemented rather than imported directly.

The 2D output shape for all variables is `(H=12960, W=17280)`.

---

## Data Generation Pipeline

### 1. Generate the Grid Zarr (one-time)

Before generating snapshots, the static grid must be extracted and stored. This is a **one-time operation** that only needs to be re-run if the grid store is deleted or the dataset name changes.

```bash
generate-global-grid-zarr \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --dataset-name llc4320_grid.zarr
```

This writes the following variables (all at shape `(12960, 17280)`) to the Zarr store:

| Variable | Description |
|---|---|
| `XC` | Longitude of T-grid cell centres |
| `YC` | Latitude of T-grid cell centres |
| `Depth` | Ocean bathymetric depth (m) |
| `hFacC` | Fractional open cell thickness (land mask proxy) |
| `rA` | T-grid cell area (m²) |
| `SN`, `CS` | Sine and cosine of grid rotation angle |
| `dxC`, `dyG` | U-grid spacing |
| `dyC`, `dxG` | V-grid spacing |
| `rAz` | Z-grid (vorticity) cell area |

**Source module:** `cli.generate_grid_global` → `generate_grid_zarr()`

**Raw data source:** `https://mghp.osn.xsede.org` (anonymous S3, kerchunk JSON references)

### 2. Generate Snapshot Data

Run the main generation script, passing the generation config and a unique `run_id`:

```bash
generate-global-llc-dataset \
    --config configs/global.yaml \
    --run-id year_1xglobal_$(date +%Y%m%d_%H%M%S)
```

Key `configs/global.yaml` settings:

```yaml
data:
  sampling_step: 1           # hours between snapshots (168 = weekly)
  start_record: 1180        # first record index after spin-up
  timestep_hours: 168       # total hours to process (336 = 2 snapshots)
  
  date_iterations:
    - '11092012-12:00:00'
  # --- alternative: specify exact timestamps in DDMMYYYY-HH:MM:SS format ---
  # date_iterations:
  #   - '01012012-00:00:00'
  #   - '01042012-00:00:00'
  # If set, date_iterations overrides sampling_step / start_record / timestep_hours.
  # Dates are converted to LLC4320 iteration numbers automatically
  # (model start: 2011-09-13 00:00 UTC, 25-second timestep).

features:
  model_data_feature_channels: [Eta, Salt, Theta, U, V, W]
  compute_features_channels: [relative_vorticity, log_gradb]

output:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof"
  folder: "native_grid_dbof_training_data"
  dataset_name: "dataset_creation.zarr"

runtime:
  zarr_async_concurrency: 128
```

The 8 output channels in order are: `Eta, Salt, Theta, U, V, W, relative_vorticity, log_gradb`.

The Zarr store layout is `(T, C, 12960, 17280)` with chunk shape `(1, 1, 12960, 17280)` — one chunk per timestep per channel.

**Source module:** `cli.generate_fronts_global` → `main()` → `process_time_snapshot()`

---

## Accessing Generated Data

Update `configs/data_access/global.yaml` to point to the generated run_id:

```yaml
data_access:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof"
  folder: "native_grid_dbof_training_data"
  run_id: "year_1xglobal_20260226_043824"   # ← set to your run_id
  feature_channels: [Eta, Salt, Theta, U, V, W, relative_vorticity, log_gradb]

grid_access:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof"
  folder: "native_grid_dbof_training_data"
  dataset_name: "llc4320_grid.zarr"
```

---

## Python API

### Snapshot Reader

```python
from dbof.dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader
from dbof.utils.filesystems import get_filesystem

fs = get_filesystem(s3_endpoint, anon=False)
reader = GlobalZarrDatasetReader(bucket, folder, run_id, dataset_name, fs)

# Load all channels for one timestep — shape (C, H, W), ~5.7 GB
snapshot = reader.get_snapshot(t=0)

# Load a single channel for one timestep — shape (H, W), ~700 MB
field = reader.get_channel_snapshot(t=0, channel="log_gradb")
# channel can also be an integer index

# Convert a model iteration number to a timestep index
t = reader.iteration_to_index(iteration=1463616)
```

**Properties:** `reader.n_timesteps`, `reader.n_channels`, `reader.shape`, `reader.channel_names`

### Grid Reader

```python
from dbof.dataset_creation.zarr_grid_global import GlobalGridZarrReader

grid_reader = GlobalGridZarrReader(bucket, folder, dataset_name, fs)

lon = grid_reader.lon      # XC (falls back to 'lon' for older stores)
lat = grid_reader.lat      # YC (falls back to 'lat' for older stores)
mask = grid_reader.land_mask  # boolean, True = land; None if hFacC not present

# Access any variable by name
depth = grid_reader["Depth"]

# Export all variables as an xarray.Dataset
ds = grid_reader.to_dataset()
```

---

## Exporting to NetCDF

`zarr-to-netcdf` has two modes: `snapshots` (per-timestep data) and `grid` (static grid).

### Snapshot export

```bash
zarr-to-netcdf \
    --mode snapshots \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --run-id year_1xglobal_20260226_043824 \
    --dates 11092012-12:00:00 \          # DDMMYYYY-HH:MM:SS
    --output-dir /path/to/output \
    --output-filename LLC4320_2012-09-11T12_00_00_props.nc
```

Optionally restrict to a subset of channels with `--channel` (repeatable):

```bash
    --channel log_gradb
```

Date format is `DDMMYYYY-HH:MM:SS`. Multiple dates can be passed as a space-separated list to `--dates`.

### Grid export

```bash
zarr-to-netcdf \
    --mode grid \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --grid-dataset-name llc4320_grid.zarr \
    --output-dir /path/to/output \
    --grid-output-filename LLC4320_grid.nc
```

**Source module:** `cli.zarr_to_netcdf` → `zarr_to_netcdf()` / `grid_zarr_to_netcdf()`

---

## Notebooks

### `notebooks_dev/running_generate_global_front_script.ipynb`

Step-by-step guide to launching the generation pipeline. Covers:
- Environment setup (local and NRP JupyterHub)
- AWS credential configuration
- YAML config parameter reference
- Executing `generate-global-llc-dataset` with a timestamped `run_id`
- Useful AWS CLI commands for inspecting the S3 output

### `notebooks_dev/assess_generate_global_front_script.ipynb`

Interactive exploration of a generated Zarr dataset. Key sections:
- Loads both snapshot and grid readers from `configs/data_access/global.yaml`
- Precomputes downsampled coordinates (`DS_GLOBAL=20`, giving a `(648, 864)` display grid)
- **Single-channel global map** 
- **8-panel global overview** — all 8 channels at once, each loaded via `get_channel_snapshot()` to minimise memory
- **Region selection** — specify `LAT_CENTER`, `LON_CENTER`, `HALF_H_KM`, `HALF_W_KM`; the notebook finds the nearest grid pixel and converts km to pixel extent using local grid spacing
- **Global context plot** — downsampled Theta with a red bounding box showing the selected region
- **Regional subplots** — 8-channel `pcolormesh` panel for the selected sub-domain

### `notebooks_dev/assess_global_netcdf.ipynb`

Minimal validation notebook for NetCDF output. Loads `LLC4320_2012-09-11T12_00_00_Divb2.nc` and `LLC4320_grid.nc` via `xarray.open_dataset`, downsamples by a factor of 20, and plots a single global map of `log_gradb`. Intended as a sanity check after running `zarr-to-netcdf`.

---

## Iteration and Date Maths

The LLC4320 model has a timestep of **25 seconds**. The data available on OSN are stored at **hourly** intervals, giving 144 model iterations per saved hour.

```
LLC4320_START_DATE = 2011-09-13 00:00:00 UTC
LLC4320_TIMESTEP_SECS = 25

iteration = (datetime - LLC4320_START_DATE).total_seconds() / 25
```

The first valid record after spin-up/wind forcing initialisation begins at approximately iteration **10368** (offset `FIRST_WIND_RECORD_OFFSET`). The last available iteration is **1,495,008**.

CLI date strings use the format `DDMMYYYY-HH:MM:SS`, e.g. `11092012-12:00:00` = 11 September 2012 at 12:00 UTC.

---

## Known Issues

### `utils.faces_to_latlon.faces_dataset_to_latlon` re-implementation

The `llcreader.llcmodel` library contains a function for assembling the 13 LLC faces into a rectangular lat/lon array. However, **two lines in that function are incompatible with recent xarray versions**, causing a runtime error. Rather than pin a specific xarray version, the function was re-implemented locally in both `cli.generate_fronts_global` and `cli.generate_grid_global` with the lines corrected.


### Asyncio cleanup traceback on exit

When `zarr-to-netcdf` finishes, a harmless `RuntimeError` from `s3fs`/`aiohttp` may appear:

```
RuntimeError: Event loop is closed
```

This is a known race condition in the `aiohttp` session teardown and does not affect the output file. It can be safely ignored.

### Grid store coordinate variables

After `_faces_dataset_to_latlon`, xarray promotes `XC` and `YC` to *coordinate* variables (not data variables). Earlier versions of the writer iterated only over `ds.data_vars`, which silently skipped them. The current writer calls `ds.reset_coords()` before writing and filters to 2D arrays only, ensuring all spatial variables are stored correctly. If you have a grid Zarr store generated before this fix, re-run `generate-global-grid-zarr` to repopulate it.
