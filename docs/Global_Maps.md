# Global DBOF Maps

**Build global maps of ocean variables from LLC4320**

This workflow converts raw LLC4320 model output from its native 13-face layout into a single stitched 2D lat/lon array of shape `(12960, 17280)`, while preserving the native curvilinear grid geometry. *Note: Grid interpolation is not performed.* The results are written to S3-backed Zarr stores. Three generation modes are available — fronts, properties, and frontogenesis — each sharing a common pipeline base and generating a different set of properties. Utilities to export selected snapshots from Zarr to NetCDF are also provided.

---

## Overview

The global pipeline consists of a shared base, three mode-specific scripts, and two support modules:

| Module | Role |
|---|---|
| `cli._generate_global_base` | Shared pipeline engine: constants, date→iteration conversion, grid setup, `process_time_snapshot`, `run_global_pipeline` |
| `cli.generate_fronts_global` | **Fronts mode**: `gradb2` + optional `relative_vorticity` |
| `cli.generate_properties_global` | **Properties mode**: all velocity-derived scalar fields via a single Jacobian pass |
| `cli.generate_frontogenesis_global` | **Frontogenesis mode**: kinematic frontogenesis tendency (full, geostrophic, ageostrophic) + geostrophic velocities |
| `cli.generate_grid_global` | One-time: extracts static LLC4320 grid variables and writes to a separate S3 Zarr store |
| `cli.zarr_to_netcdf` | Exports selected snapshots or the full grid from Zarr to NetCDF |
| `dataset_creation.zarr_dataset_global` | Low-level Zarr writer (`GlobalZarrDataset`) and reader (`GlobalZarrDatasetReader`) for snapshot data |
| `dataset_creation.zarr_grid_global` | Low-level Zarr writer (`GlobalGridZarrWriter`) and reader (`GlobalGridZarrReader`) for the static grid |

Each mode script defines only a `_compute_xxx_fields(ds_merge, grid, computed_feature_channels) -> dict` callback and a `main()` that calls `run_global_pipeline`. All shared logic lives in `_generate_global_base.py`.

Configuration is split per mode:

| Generation config | Access config | Mode |
|---|---|---|
| `configs/global.yaml` | `configs/data_access/global.yaml` | Fronts |
| `configs/global_properties.yaml` | `configs/data_access/global_properties.yaml` | Properties |
| `configs/frontogenesis_global.yaml` | `configs/data_access/frontogenesis_global.yaml` | Frontogenesis |

*`run_id` in the access config must be updated manually to match the generation run.*

---

## LLC4320 Grid and Coordinate System

LLC4320 output is stored across **13 curvilinear faces**. Stitching to a 2D grid is handled by `utils.faces_to_latlon.faces_dataset_to_latlon()`, a local re-implementation of the equivalent function in `llcreader.llcmodel`. See [Known Issues](#known-issues).

The 2D output shape for all variables is `(H=12960, W=17280)`.

---

## Computed Fields

`gradb2` is always computed and appended as the last channel in every mode. Mode-specific computed channels come from `preprocessing.calculate_additional_fields`:

### Always present: `gradb2`

| Channel | Formula | Description |
|---|---|---|
| `gradb2` | `∣∇b∣²` | Squared surface buoyancy gradient magnitude (s⁻⁴). Buoyancy derived from Theta and Salt via the linear equation of state. Used for front detection. |

### Fronts mode (`generate_fronts_global`)

| Channel | Description |
|---|---|
| `relative_vorticity` | Vertical component of relative vorticity, `∂v/∂x − ∂u/∂y` (s⁻¹) |

### Properties mode (`generate_properties_global`)

All fields derived from a single Jacobian pass over U/V:

| Channel | Description |
|---|---|
| `relative_vorticity` | `∂v/∂x − ∂u/∂y` (s⁻¹) |
| `strain_n` | Normal strain, `∂u/∂x − ∂v/∂y` (s⁻¹) |
| `strain_s` | Shear strain, `∂u/∂y + ∂v/∂x` (s⁻¹) |
| `strain_mag` | Strain magnitude, `√(strain_n² + strain_s²)` (s⁻¹) |
| `divergence` | Horizontal divergence, `∂u/∂x + ∂v/∂y` (s⁻¹) |
| `coriolis_f` | Coriolis parameter, `2Ω sin(φ)` (s⁻¹). Uses only grid latitude — no S3 reads. |
| `rossby_number` | `ζ / f` — ratio of relative to planetary vorticity (dimensionless) |
| `okubo_weiss` | `strain_n² + strain_s² − ζ²` — distinguishes strain-dominated (fronts/filaments) from rotation-dominated (vortex cores) regions (s⁻²) |

### Frontogenesis mode (`generate_frontogenesis_global`)

**This mode is a work-in-progress** 

All fields derived from a single Jacobian pass over U/V plus buoyancy and SSH gradients:

| Channel | Description |
|---|---|
| `frontogenesis_tendency` | Kinematic frontogenesis tendency `F(u,v)`: rate of buoyancy gradient intensification by the full velocity field |
| `frontogenesis_geo` | Geostrophic frontogenesis `F(ug,vg)`: frontogenesis due to the geostrophic velocity alone |
| `frontogenesis_ageo` | Ageostrophic frontogenesis `F − F_geo`: residual, qualitative measure of ageostrophic influence |
| `ug` | Geostrophic zonal velocity `−(g/f) ∂η/∂y` (m/s). Near the equator f→0 so values diverge; mask if needed. |
| `vg` | Geostrophic meridional velocity `(g/f) ∂η/∂x` (m/s) |

**Note:** `gradb2` is always appended automatically. Do not include it in `compute_features_channels` in the YAML — list only the mode-specific channels.

---

## Data Generation Pipeline

### 1. Generate the Grid Zarr (one-time)

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

Run the script for the desired mode, passing its config and a unique `run_id`:

```bash
# Fronts
generate-global-llc-dataset \
    --config configs/global.yaml \
    --run-id fronts_$(date +%Y%m%d_%H%M%S)

# Properties
generate-global-properties-dataset \
    --config configs/global_properties.yaml \
    --run-id properties_$(date +%Y%m%d_%H%M%S)

# Frontogenesis
generate-global-frontogenesis-dataset \
    --config configs/frontogenesis_global.yaml \
    --run-id frontogenesis_$(date +%Y%m%d_%H%M%S)
```

Key settings shared across all mode configs:

```yaml
data:
  date_iterations:
    - '2012-09-11 12:00:00'   # YYYY-MM-DD HH:MM:SS (ISO). Overrides range mode.
  # Range mode (if date_iterations is not set):
  sampling_step: 168          # hours between snapshots
  start_record: 1180
  timestep_hours: 168

features:
  model_data_feature_channels:
    # Comment out channels to exclude them from the Zarr output.
    # They are still loaded and used in computation, just not saved.
    # - Eta
    # - Salt
    # - Theta
    # - U
    # - V
    # - W
  compute_features_channels:
    - relative_vorticity       # (properties mode example)
    - strain_n
    # ... (see Computed Fields section above)
    # gradb2 is always appended automatically — do not list it here

output:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof"
  folder: "native_grid_dbof_training_data"
  dataset_name: "properties.zarr"   # or "dataset_creation.zarr", "frontogenesis.zarr"

runtime:
  zarr_async_concurrency: 128
```

The Zarr store layout is `(T, C, 12960, 17280)` with chunk shape `(1, 1, 12960, 17280)` — one chunk per timestep per channel.

**Source modules:** `cli.generate_*_global` → `main()` → `cli._generate_global_base.run_global_pipeline()` → `process_time_snapshot()`

---

## Accessing Generated Data

Update the appropriate access config to point to the generated `run_id`:

```yaml
data_access:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof"
  folder: "native_grid_dbof_training_data"
  run_id: "properties_20260306_174221"   # ← set to your run_id
  feature_channels:
    - "relative_vorticity"
    - "strain_n"
    - "strain_s"
    - "strain_mag"
    - "divergence"
    - "coriolis_f"
    - "rossby_number"
    - "okubo_weiss"
    - "gradb2"

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
field = reader.get_channel_snapshot(t=0, channel="okubo_weiss")
# channel can also be an integer index

# Convert a model iteration number to a timestep index
t = reader.iteration_to_index(iteration=1463616)
```

**Properties:** `reader.n_timesteps`, `reader.n_channels`, `reader.shape`, `reader.channel_names`

### Grid Reader

```python
from dbof.dataset_creation.zarr_grid_global import GlobalGridZarrReader

grid_reader = GlobalGridZarrReader(bucket, folder, dataset_name, fs)

lon = grid_reader.lon         # XC (falls back to 'lon' for older stores)
lat = grid_reader.lat         # YC (falls back to 'lat' for older stores)
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
    --run-id properties_20260306_174221 \
    --dates '2012-11-09 12:00:00' \
    --output-dir /path/to/output \
    --output-filename LLC4320_2012-11-09T12_00_00_props.nc
```

Optionally restrict to a subset of channels with `--channel`:

```bash
    --channel okubo_weiss
```

Date format is `YYYY-MM-DD HH:MM:SS`. Multiple dates can be passed as a space-separated list to `--dates`.

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

Step-by-step guide for the fronts mode. Covers environment setup, AWS credentials, YAML config reference, and running `generate-global-llc-dataset`.

### `notebooks_dev/running_generate_global_properties_script.ipynb`

Same as above for the properties mode (`generate-global-properties-dataset`, `global_properties.yaml`).

### `notebooks_dev/running_generate_global_frontogenesis_script.ipynb`

Same as above for the frontogenesis mode (`generate-global-frontogenesis-dataset`, `frontogenesis_global.yaml`).

### `notebooks_dev/assess_generate_global_front_script.ipynb` / `..._properties_script.ipynb` / `..._frontogenesis_script.ipynb`

Interactive exploration of a generated Zarr dataset. Each notebook:
- Loads both snapshot and grid readers from the relevant `configs/data_access/*.yaml`
- **Single-channel global map**
- **Multi-panel global overview** — all channels at once via `get_channel_snapshot()`
- **Region selection** — specify `LAT_CENTER`, `LON_CENTER`, `HALF_H_KM`, `HALF_W_KM`
- **Global context plot** with a red bounding box on the selected region
- **Regional subplots** — `pcolormesh` panel for the selected sub-domain

### `notebooks_dev/assess_global_netcdf.ipynb`

Minimal validation notebook for NetCDF output. Loads a snapshot `.nc` and `LLC4320_grid.nc` via `xarray.open_dataset` and plots a single global map. Sanity check after running `zarr-to-netcdf`.

---

## Iteration and Date Maths

The LLC4320 model has a timestep of **25 seconds**. The data available on OSN are stored at **hourly** intervals, giving 144 model iterations per saved hour.

```
LLC4320_START_DATE = 2011-09-13 00:00:00 UTC
LLC4320_TIMESTEP_SECS = 25

iteration = (datetime - LLC4320_START_DATE).total_seconds() / 25
```

The first valid record after spin-up begins at approximately iteration **10368** (`FIRST_WIND_RECORD_OFFSET`). The last available iteration is **1,495,008**.

CLI date strings use ISO format `YYYY-MM-DD HH:MM:SS`, e.g. `2012-09-11 12:00:00` = 11 September 2012 at 12:00 UTC.

---

## Known Issues

### `utils.faces_to_latlon.faces_dataset_to_latlon` re-implementation

The `llcreader.llcmodel` library contains a function for assembling the 13 LLC faces into a rectangular lat/lon array. However, **two lines in that function are incompatible with recent xarray versions**, causing a runtime error. Rather than pin a specific xarray version, the function was re-implemented locally in `utils.faces_to_latlon` and is used by all three generate scripts via `_generate_global_base`.

### Asyncio cleanup traceback on exit

When `zarr-to-netcdf` finishes, a harmless `RuntimeError` from `s3fs`/`aiohttp` may appear:

```
RuntimeError: Event loop is closed
```

This is a known race condition in the `aiohttp` session teardown and does not affect the output file. It can be safely ignored.

### Grid store coordinate variables

After `faces_dataset_to_latlon`, xarray promotes `XC` and `YC` to *coordinate* variables (not data variables). Earlier versions of the writer iterated only over `ds.data_vars`, which silently skipped them. The current writer calls `ds.reset_coords()` before writing and filters to 2D arrays only, ensuring all spatial variables are stored correctly. If you have a grid Zarr store generated before this fix, re-run `generate-global-grid-zarr` to repopulate it.

### Geostrophic velocity near the equator

`ug` and `vg` are computed as `±(g/f) ∂η/∂x,y`. Because `f → 0` near the equator, both fields diverge to large values in a narrow equatorial band. This is physically correct (geostrophic balance breaks down there) but may require masking before use in downstream analysis.
