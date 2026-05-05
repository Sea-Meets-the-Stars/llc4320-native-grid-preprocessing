# Global DBOF Maps

**Build global LLC4320 maps on the native curvilinear grid.**

This document describes the current pipeline for turning raw LLC4320 model
output (13 curvilinear faces) into stitched 2D `(12960, 17280)` arrays written
to S3-backed Zarr stores, plus the NetCDF export step. The native grid geometry
is preserved — no interpolation to a regular lat/lon grid is performed.

---

## Main pipeline

Three scripts, run in this order, make up the supported workflow:

1. `cli.generate_grid_global` — build the static grid Zarr (run once).
2. `cli.generate_global` — build per-timestep snapshot Zarr stores (run per dataset).
3. `cli.zarr_to_netcdf` — export selected snapshots or the grid to NetCDF.

The CLI entry points defined in `pyproject.toml` are:

| Script                         | Console command              | Config                                 |
|--------------------------------|------------------------------|----------------------------------------|
| `cli.generate_grid_global`     | `generate-global-grid-zarr`  | CLI flags only                         |
| `cli.generate_global`          | `generate-global`            | `configs/global.yaml` (+ `--subset`)   |
| `cli.zarr_to_netcdf`           | `zarr-to-netcdf`             | CLI flags only                         |

All three target the same S3 bucket/folder. `run_id` ties snapshot stores to a
specific generation run; the grid store is shared across runs.

---

### 1. `generate_grid_global.py` — static grid (one-time)

Extracts LLC4320 static grid variables, stitches the 13 faces into a single
`(12960, 17280)` array per variable, and writes them to a single Zarr store at
`s3://{bucket}/{folder}/{dataset_name}` (default `llc4320_grid.zarr`). There is
no `run_id` in the grid path — this store is shared by every downstream run.

Variables written (all shape `(12960, 17280)`):

| Variable      | Description                                   |
|---------------|-----------------------------------------------|
| `XC`, `YC`    | Longitude / latitude of T-grid cell centres   |
| `Depth`       | Ocean bathymetric depth (m)                   |
| `hFacC`       | Fractional open cell thickness (land mask)    |
| `rA`, `rAz`   | T-grid and Z-grid cell areas (m²)             |
| `SN`, `CS`    | Sine / cosine of grid rotation angle          |
| `dxC`, `dyG`  | U-grid spacing                                |
| `dyC`, `dxG`  | V-grid spacing                                |

Run:

```bash
generate-global-grid-zarr \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --dataset-name llc4320_grid.zarr
```

Raw grid variables are read from OSN
(`https://mghp.osn.xsede.org`, anonymous, kerchunk JSON references).

You only need to re-run this when:
- The grid store does not yet exist in the target S3 folder, **or**
- The variable list / stitching logic has changed.

---

### 2. `generate_global.py` — snapshot generation

Single-entry-point script for all per-timestep global outputs. A `--subset`
flag selects which group of fields is computed and written for each iteration.
The subsets are defined in `configs/global.yaml` under the top-level `subsets:`
key; each subset has its own `dataset_name` and channel lists.

Available subsets (see `SUBSET_COMPUTE_FNS` in `cli/generate_global.py`):

| Subset              | What it writes                                                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `native_fields`     | Raw model fields only: `Theta`, `Salt`, `Eta`, `U`, `V`, `W`. No derived quantities.                                             |
| `frontal_structure` | Scalar gradient magnitudes: `gradsalt2`, `gradtheta2`, `gradeta2`,`gradb2`, `gradrho2`,`turner_angle`.                             |
| `kinematic`         | Single Jacobian pass over U/V: `relative_vorticity`, `strain_n`, `strain_s`, `strain_mag`, `divergence`, `coriolis_f`, `rossby_number`, `okubo_weiss`. |
| `frontogenesis`     | Kinematic frontogenesis + geostrophic decomposition: `frontogenesis_tendency`, `frontogenesis_geo`, `frontogenesis_ageo`, `ug`, `vg`. Materialised with a single `dask.compute()` to fuse the shared subgraph. |


Properties: 

| Field | Subset | Units | Equation | Description |
|---|---|---|---|---|
| `Theta` | `native_fields` | °C | — | Potential temperature (LLC4320 native field) |
| `Salt` | `native_fields` | PSU | — | Salinity (LLC4320 native field) |
| `Eta` | `native_fields` | m | — | Sea surface height (LLC4320 native field) |
| `U` | `native_fields` | m/s | — | Zonal velocity (LLC4320 native field) |
| `V` | `native_fields` | m/s | — | Meridional velocity (LLC4320 native field) |
| `W` | `native_fields` | m/s | — | Vertical velocity (LLC4320 native field) |
| `gradb2` | `frontal_structure` | s⁻⁴ | \|∇b\|² = (∂b/∂x)² + (∂b/∂y)² | Squared surface buoyancy gradient magnitude |
| `gradsalt2` | `frontal_structure` | (PSU/m)² | \|∇S\|² = (∂S/∂x)² + (∂S/∂y)² | Squared salinity gradient magnitude |
| `gradtheta2` | `frontal_structure` | (K/m)² | \|∇θ\|² = (∂θ/∂x)² + (∂θ/∂y)² | Squared temperature gradient magnitude |
| `gradeta2` | `frontal_structure` | (m/m)² | \|∇η\|² = (∂η/∂x)² + (∂η/∂y)² | Squared SSH gradient magnitude |
| `relative_vorticity` | `kinematic` | s⁻¹ | ω = ∂v/∂x − ∂u/∂y | Relative vorticity |
| `strain_n` | `kinematic` | s⁻¹ | σ_n = ∂u/∂x − ∂v/∂y | Normal (stretching) strain |
| `strain_s` | `kinematic` | s⁻¹ | σ_s = ∂u/∂y + ∂v/∂x | Shear strain |
| `strain_mag` | `kinematic` | s⁻¹ | \|σ\| = √(σ_n² + σ_s²) | Strain magnitude |
| `divergence` | `kinematic` | s⁻¹ | δ = ∂u/∂x + ∂v/∂y | Horizontal velocity divergence |
| `coriolis_f` | `kinematic` | s⁻¹ | f = 2Ω sin(φ) | Coriolis parameter |
| `rossby_number` | `kinematic` | dimensionless | Ro = ω/f | Rossby number |
| `okubo_weiss` | `kinematic` | s⁻² | OW = σ_n² + σ_s² − ω² | Okubo-Weiss parameter |
| `frontogenesis_tendency` | `frontogenesis` | s⁻⁵ | F = −(∂u/∂x · b_x² + (∂u/∂y + ∂v/∂x) · b_x b_y + ∂v/∂y · b_y²) | Kinematic frontogenesis tendency |
| `ug` | `frontogenesis` | m/s | u_g = −(g/f) ∂η/∂y | Geostrophic zonal velocity |
| `vg` | `frontogenesis` | m/s | v_g = (g/f) ∂η/∂x | Geostrophic meridional velocity |
| `frontogenesis_geo` | `frontogenesis` | s⁻⁵ | F(u_g, v_g) | Geostrophic frontogenesis tendency |
| `frontogenesis_ageo` | `frontogenesis` | s⁻⁵ | F − F_geo | Ageostrophic frontogenesis tendency |

Run:

```bash
generate-global \
    --config configs/global.yaml \
    --subset kinematic \
    --run_id kinematic_$(date +%Y%m%d_%H%M%S)
```

Other flags:
- `--subset` — overrides `active_subset:` in the YAML.
- `--run_id` — overrides `run:run_id:` in the YAML.
- `--no-icemask` — disables the `Theta <= 0` sea-ice mask (land mask is always applied).

Output layout: `s3://{bucket}/{folder}/{run_id}/{dataset_name}`, shaped
`(T, C, 12960, 17280)` with chunk shape `(1, 1, 12960, 17280)` — one chunk per
timestep per channel.

Key config shape:

```yaml
run:
  run_id: "kinematic_test00"

data:
  # Option A: explicit ISO timestamps (overrides range mode).
  date_iterations:
    - '2012-11-09 12:00:00'
  # Option B: range mode (used when date_iterations is unset).
  sampling_step: 168       # hours between snapshots
  start_record: 1180
  timestep_hours: 168

output:
  s3_endpoint: "https://s3-west.nrp-nautilus.io"
  bucket: "dbof/"
  folder: "native_grid_dbof_training_data/"

active_subset: kinematic

subsets:
  kinematic:
    dataset_name: "kinematic.zarr"
    model_data_feature_channels: [Theta, Salt, Eta, U, V]
    compute_features_channels:
      - relative_vorticity
      - strain_n
      # ...
  frontogenesis:
    dataset_name: "frontogenesis.zarr"
    # ...
```

---

### 3. `zarr_to_netcdf.py` — NetCDF export

Exports either snapshot data or the static grid from S3 Zarr to local NetCDF.
Two modes, selected with `--mode`.

Snapshots:

```bash
zarr-to-netcdf \
    --mode snapshots \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof \
    --folder native_grid_dbof_training_data \
    --run-id kinematic_20260306_174221 \
    --dates '2012-11-09 12:00:00' \
    --output-dir /path/to/output \
    --output-filename LLC4320_2012-11-09T12_kin.nc
```

Select timesteps with `--dates` (ISO strings), `--iterations`, or `--indices`.
Select a subset of channels with repeated `--channel NAME`. Omitting these
writes every timestep in the store, all channels.

Grid:

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

Snapshot files do **not** include lat/lon or grid spacings — load the grid
NetCDF alongside them.

---

## Typical workflow

1. **Once per S3 folder:** `generate-global-grid-zarr` → `llc4320_grid.zarr`.
2. **Once per dataset you want:** `generate-global --config configs/global.yaml --subset <name> --run_id <id>`.
3. **When you need NetCDF locally:** `zarr-to-netcdf --mode snapshots ...` (and, if you don't have it yet, `--mode grid`).

Snapshot and grid stores live in the same S3 folder; the `run_id` path segment
separates runs.

---

## Iteration / date conventions

LLC4320 reference:

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
field  = reader.get_channel_snapshot(t=0, channel="okubo_weiss")  # (H, W)
t_idx  = reader.iteration_to_index(iteration=1463616)

grid   = GlobalGridZarrReader(bucket, folder, dataset_name="llc4320_grid.zarr", fs=fs)
lon, lat = grid.lon, grid.lat
land_mask = grid.land_mask
ds_grid  = grid.to_dataset()
```

---

## Side notes / special-case scripts

These scripts exist in `src/dbof/cli/` but are **not** part of the main
pipeline above.

### `transfer_llc4320.py` — MIT → S3 Zarr transfer

Console command: `transfer-timestep`. Config: `configs/transfer.yaml`.

- Transfers LLC4320 variables from a local (MIT) Zarr store into S3 with a
  unified tiled layout.
- Writes two separate stores:
  `{folder}/grid.zarr` (static grid, one-time) and
  `{folder}/{YYYYMMDDTHH}.zarr` (per-timestep, including full-depth fields).
- These timestep stores are the input that `generate_global_depth.py` reads.

It is purely a data-movement utility — it does not generate maps and is not
part of the "generate maps / export NetCDF" pipeline described above.

---

## Known issues

### `utils.faces_to_latlon.faces_dataset_to_latlon` re-implementation

`llcreader.llcmodel` provides a function for stitching the 13 LLC faces into a
rectangular lat/lon array, but two lines in the upstream implementation are
incompatible with recent xarray versions. Rather than pin xarray, the function
is re-implemented locally in `utils.faces_to_latlon` and is used by every
generate script.

### Asyncio cleanup traceback

A harmless `RuntimeError: Event loop is closed` from `s3fs` / `aiohttp` can
appear when `zarr-to-netcdf` exits. It is a teardown race and does not affect
output files.

### Grid store coordinate variables

`faces_dataset_to_latlon` promotes `XC` / `YC` to coordinates. The current
`GlobalGridZarrWriter` calls `ds.reset_coords()` and filters to 2D arrays
before writing. Older stores written before this fix should be regenerated with
`generate-global-grid-zarr`.

### Geostrophic velocity near the equator

`ug`, `vg` are `±(g/f) ∂η/∂x,y`; as `f → 0` at the equator both diverge in a
narrow band. Physically correct, but may require masking downstream.

---
