# Global DBOF Maps

**Build global LLC4320 maps on the native curvilinear grid.**

This document describes the pipelines for turning raw LLC4320 model output
(13 curvilinear faces) into stitched 2D `(12960, 17280)` arrays written to
S3-backed Zarr stores, plus the NetCDF export step. The native grid geometry is
preserved — no interpolation to a regular lat/lon grid is performed.

> **Need just one 720×720 depth-resolved tile instead of a full global map?**
> See [Single-Tile 3D Property Extraction](Tiles.md), which reads from the same
> LLC_DEPTH source described below.

---

## Choosing a pipeline

Three pipeline variants exist. **You must choose one before running.** The
right choice depends on what variables you need, at what depths, and whether
you have transferred data from MIT.

**Start here:**

1. **Do you need depth-resolved fields (below the surface)?** → Use **DEPTH**.
   Requires timesteps transferred from MIT via `transfer_llc4320.py`.

2. **Do you only need surface fields?**
   - **Do you need wind stress (`oceTAUX`, `oceTAUY`) or sea ice (`SIarea`)?**
     - Is your date within **2011-11-01 to 2012-07-15**? → Use **OSN**.
       No data transfer needed.
     - Is your date outside that window? → Use **SURF**. Requires timesteps
       transferred from MIT via `transfer_llc4320.py`.
   - **Do you only need core variables (`Theta`, `Salt`, `Eta`, `U`, `V`, `W`)
     or derived fields (frontal_structure, kinematic, frontogenesis)?**
     → Use **OSN**. No data transfer needed. Dates must fall within
     **2011-09-13 to 2012-11-15**.

### Data availability

| Source     | Variables                                  | Date range                    | Prerequisite            |
|------------|--------------------------------------------|-------------------------------|-------------------------|
| OSN (`llc_surf`)  | Theta, Salt, Eta, U, V, W           | 2011-09-13 – 2012-11-15      | None                    |
| OSN (`llc_wind`)  | oceTAUX, oceTAUY, SIarea            | 2011-11-01 – 2012-07-15      | None                    |
| LLC_SURF (S3)     | All surface variables + grid         | Per-timestep, on request      | `transfer_llc4320.py`   |
| LLC_DEPTH (S3)    | All variables, 51 depth levels + grid| Per-timestep, on request      | `transfer_llc4320.py`   |

The `surface_wind` and `icearea` subsets require wind/ice variables. In the
OSN pipeline these come from the `llc_wind` kerchunk endpoint (limited date
range). In the SURF pipeline they come from S3 timestep stores — the date
range is unlimited, but the data must first be transferred from the MIT Zarr
store using `transfer_llc4320.py`. There is currently no pre-flight check
that a timestep store exists in S3, so a missing transfer will surface as an
S3 read error at runtime.

---

## Pipeline overview

A single unified entry point (`generate_global.py`) dispatches on the
`pipeline` key in the YAML config:

| Pipeline | Depth support         | Input source                  | Prerequisite          |
|----------|-----------------------|-------------------------------|-----------------------|
| `SURF`   | Surface only          | OSN kerchunk + S3 forcing     | `transfer_llc4320.py` |
| `OSN`    | Surface only          | OSN kerchunk (direct)         | None                  |
| `DEPTH`  | Surface + depth-aware | S3 timestep stores (from MIT) | `transfer_llc4320.py` |

All variants share the same output format: S3 Zarr stores shaped
`(T, C, 12960, 17280)` with chunk shape `(1, 1, 12960, 17280)`.

Config: `configs/global/run/run.yaml`. Console command: `generate-global`.

```bash
generate-global --config configs/global/run/run.yaml
generate-global --config configs/global/run/run.yaml --subset kinematic
generate-global --config configs/global/run/run.yaml --pipeline OSN
```

Flags: `--subset`, `--pipeline`, `--run_id`, `--no-icemask`.

### Pipeline variants

**SURF** — Surface-only global maps. Reads core variables (`Theta`, `Salt`,
`Eta`, `U`, `V`, `W`) from OSN kerchunk and forcing variables (`oceTAUX`,
`oceTAUY`, `SIarea`) from S3 timestep stores written by
`transfer_llc4320.py`.

**OSN** — Surface-only global maps from OSN kerchunk endpoints (no S3
timestep stores needed). Uses two OSN endpoints: the standard one for core
variables and `llc_wind` for wind stress and sea-ice variables.

**DEPTH** — Fully-lazy Dask pipeline for depth-resolved diagnostics. Reads
full 3D data from S3 timestep stores, computes derived fields using xgcm,
reduces to 2D via depth strategies, stitches the 13 LLC faces, and writes to
S3 Zarr. **Requires data transferred from MIT using `transfer_llc4320.py`.**

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

Each subset definition has two channel lists:

- **Base fields** (`compute_features_channels`) are crossed with every active
  depth suffix. For example, `N2` with suffixes `[sfc, z25m, mld, mld_mean]`
  produces output channels `N2_sfc`, `N2_z25m`, `N2_mld`, `N2_mld_mean`.

- **Extra channels** (`extra_channels`) are appended as-is, with no suffix
  expansion. Use these for diagnostics that are inherently 2D and would be
  nonsensical at multiple depths — e.g. `mixed_layer_depth`, `coriolis_f`.

The expansion is handled by `expand_channels_with_suffixes()` in
`subset_definitions.py`:

```yaml
compute_features_channels: [N2]
depth_suffixes: [sfc, z25m, mld, mld_mean]
extra_channels: [mixed_layer_depth]
# → output channels: N2_sfc, N2_z25m, N2_mld, N2_mld_mean, mixed_layer_depth
```

### Surface vs. depth-aware within the depth pipeline

Subset entries can set `surface_only: true` to skip depth processing entirely.
These subsets (e.g. `surface_wind`, `icearea`) read only surface-level data
from the S3 timestep stores — they use the depth pipeline's S3 data access but
not its 3D computation machinery.

---

## Subsets

Each pipeline supports a specific set of subsets.  Running a subset that
does not belong to the chosen pipeline will raise a `ValueError`.  The
canonical reference for valid pipeline × subset combinations is in
`configs/global/run/run.yaml`.

A **subset** is a named group of related calculated fields/properties that are
computed and written together as a single Zarr store. Each subset has its own
`dataset_name` (e.g. `stratification.zarr`) and channel list defined in the
config YAML.

Subsets exist because the full set of derived fields is large — grouping
related fields together lets you generate and export only what you need.

### Re-running and changing `depth_suffixes`

Re-runs are incremental ("`.nc`-first"): for each subset × date,
`run-all-subsets` checks the exported `.nc` files first. If all exist the
subset/date is skipped entirely; if some are missing it checks the Zarr
store, exports the missing channels when the store is complete, and only
regenerates the store when it is missing or incomplete (wrong channels, or
no data written). When a store **is** regenerated, *all* of its channels are
re-exported (existing `.nc` files overwritten), so every export for a
subset/date always comes from the same store build. The work plan is logged
at startup — use `--dry-run` to preview it.

**Changing `depth_suffixes` (or channel definitions) for an existing
run_id/date is refused by design.** A store's channel axis is fixed at
creation, so writing a different channel set would corrupt it. You will see:

```
ValueError: Channel mismatch for existing zarr store at s3://...
Refusing to write -- this would corrupt the store.
```

The pipeline never deletes data; to move forward either:

1. **Delete that subset's store manually** and re-run (only the affected
   subset/date is rebuilt):

   ```bash
   aws --endpoint-url https://s3-west.nrp-nautilus.io s3 rm --recursive \
     s3://dbof/{folder}/{run_id}/{date_prefix}/{subset}.zarr
   ```

2. **Use a new `run_id`**, keeping the old outputs intact.

`run_meta.yaml` (next to the logs, and at
`s3://{bucket}/{folder}/{run_id}/run_meta.yaml`) records the suffixes and
channels each subset was actually built with.

### Surface pipeline subsets (SURF / OSN)

| Subset              | Fields                                                                                      | Requires wind/ice data |
|---------------------|---------------------------------------------------------------------------------------------|------------------------|
| `native_fields`     | `Theta`, `Salt`, `Eta`, `U`, `V`, `W`                                                      | No                     |
| `surface_wind`      | `oceTAUX`, `oceTAUY`, `wind_stress_curl`, `ekman_pumping`, `u_ekman`, `v_ekman` (+ `oceQnet`, SURF only) | Yes  |
| `icearea`           | `SIarea`                                                                                    | Yes                    |
| `frontal_structure` | `gradb2`, `gradsalt2`, `gradtheta2`, `gradeta2`, `gradrho2`, `turner_angle`, `density`, `buoyancy` | No              |
| `kinematic`         | `relative_vorticity`, `strain_n`, `strain_s`, `strain_mag`, `divergence`, `coriolis_f`, `rossby_number`, `okubo_weiss` | No |
| `frontogenesis`     | `frontogenesis_tendency`, `frontogenesis_geo`, `frontogenesis_ageo`, `ug`, `vg`              | No                     |

Subsets marked "Requires wind/ice data" need `oceTAUX`, `oceTAUY`, or
`SIarea`. On **OSN** these come from the `llc_wind` kerchunk endpoint, which
only covers **2011-11-01 to 2012-07-15**. On **SURF** they come from S3
timestep stores, which have no date restriction but must be transferred from
MIT first using `transfer_llc4320.py`.

### Depth pipeline subsets (DEPTH)

| Subset              | Base fields                                        | Depth suffixes              | Extra channels                          |
|---------------------|----------------------------------------------------|-----------------------------|-----------------------------------------|
| `stratification`    | `N2`                                               | sfc, z25m, mld, mld_mean   | `mixed_layer_depth`, `ml_heat_content`  |
| `vertical_shear`    | `vertical_shear`, `Ri`                             | sfc, z25m, mld, mld_mean   |                                         |
| `mixing_parameters` | `Fr`, `Ro`, `Bu`                                   | sfc, z25m, mld, mld_mean   |                                         |
| `ertel_pv`          | `ertel_pv`, `ertel_pv_vertical`, `ertel_pv_tilt`  | sfc, z25m, mld, mld_mean   |                                         |
| `buoyancy_fluxes`   | `uB`, `vB`, `wB`                                   | sfc, z25m, mld, mld_mean   |                                         |
| `energetics`        | `KE`                                               | sfc, z25m, mld, mld_mean   |                                         |
| `frontal_structure` | `gradb2`, `gradtheta2`, `gradsalt2`, `gradrho2`, `gradeta2`, `turner_angle` | sfc, z25m, mld, mld_mean |                   |
| `kinematic`         | `relative_vorticity`, `strain_n`, `strain_s`, `strain_mag`, `divergence`, `rossby_number`, `okubo_weiss` | sfc, z25m, mld, mld_mean | `coriolis_f` |
| `frontogenesis`     | `frontogenesis_tendency`, `frontogenesis_geo`, `frontogenesis_ageo`, `ug`, `vg` | sfc, z25m, mld, mld_mean |                   |
| `native_fields`     | `Theta`, `Salt`, `Eta`, `U`, `V`, `W`             | sfc, z25m, mld, mld_mean   |                                         |
| `surface_wind`      | *(surface_only)* `oceTAUX`, `oceTAUY`, `wind_stress_curl`, `ekman_pumping`, `u_ekman`, `v_ekman` + model field `oceQnet` | — | |
| `icearea`           | *(surface_only)* model field `SIarea`              | —                           |                                         |

---

## Batch driver: `run_all_subsets.py`

The batch driver (`dbof.cli.run_all_subsets`) automates the two-phase
workflow across all active subsets in a single `run.yaml` config (the same
config used by `generate_global.py`):

1. **Phase 1 — Generate:** calls `generate_global.main()` for each
   subset, producing Zarr stores on S3.
2. **Phase 2 — Export:** calls `zarr_to_netcdf.main()` once per channel,
   writing individual NetCDF files.

Subset definitions and channel lists come from code
(`subset_definitions.py`), not from the YAML.

### NetCDF output layout

```
{netcdf_base}/{run_id}/{date_prefix}/LLC4320_{date}_{channel}_{run_id}.nc
```

`--netcdf-base` is a required CLI argument (no default).
`date` is formatted as `2012-11-09T12_00_00`.

Example:
```
/my/output/dir/global_test00/20121109_120000/LLC4320_2012-11-09T12_00_00_N2_sfc_global_test00.nc
```

### CLI usage

The `--pipeline` flag selects which pipeline variant to run (SURF, OSN,
or DEPTH).  It can also be set via the `pipeline` key in the YAML config;
the CLI flag takes precedence.

```bash
# DEPTH pipeline — generate + export all active subsets:
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output

# SURF pipeline:
run-all-subsets --pipeline SURF \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output

# OSN pipeline:
run-all-subsets --pipeline OSN \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output

# Only specific subsets (overrides active_subsets in YAML):
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output \
    --subsets stratification native_fields

# Export only (assumes Zarr stores already exist):
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output --export-only

# Generate only (skip NetCDF export):
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output --generate-only

# Override run_id:
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output --run-id my_run_01

# Dry run:
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output --dry-run

# With ice masking:
run-all-subsets --pipeline DEPTH \
    --config configs/global/run/run.yaml --netcdf-base /path/to/output --ice-mask
```

### Ice masking

Ice masking is **not** applied during the generate step — the Zarr stores
contain unmasked data. Instead, ice masking is applied as a post-processing
step during the NetCDF export (Phase 2), via `zarr_to_netcdf.py`. When
`--ice-mask` is passed, the exporter reads `SIarea` from the `icearea.zarr`
store (same bucket/folder/run_id/date_prefix) and sets all points where
`SIarea > 0` to NaN in the exported NetCDF. The `icearea` subset itself is
never self-masked.

The batch driver automatically checks whether `icearea.zarr` exists before
Phase 1 when `--ice-mask` is set. If it is missing for any date, the driver
generates it first so that the export phase can proceed. This check is
skipped in `--export-only` mode (a warning is logged instead).

---

## Preprocessing: `transfer_llc4320.py`

The SURF and DEPTH pipelines read from S3 timestep stores. These stores must
be created by transferring data from the MIT Zarr store using
`transfer_llc4320.py`.

**`transfer_llc4320.py` must be run on the MIT machines**, since it reads
from a local Zarr store that is only accessible there. Each date you want
to process downstream must be transferred before running the generate
pipelines — there is currently no pre-flight check that the timestep store
exists in S3, so a missing transfer will surface as an S3 read error at
runtime.

Console command: `transfer-timestep`. Config: `configs/transfer/run.yaml`.

The transfer writes two stores per timestep:

- **Static grid:** `s3://{bucket}/{folder}/grid.zarr` (one-time).
- **Timestep data:** `s3://{bucket}/{folder}/{YYYYMMDDTHH}.zarr` (per date,
  includes full-depth 3D fields).

Only the OSN pipeline does not require this step — it reads directly from
OSN kerchunk endpoints.

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
    --bucket dbof --folder depth_fields \
    --run-id global_depth_test00 \
    --dates '2012-11-09 12:00:00' \
    --output-dir /path/to/output

# Grid
zarr-to-netcdf --mode grid \
    --s3-endpoint https://s3-west.nrp-nautilus.io \
    --bucket dbof --folder depth_fields \
    --grid-dataset-name llc4320_grid.zarr \
    --output-dir /path/to/output
```

---

## Zarr output layout

SURF / OSN write to ``surface_fields/``, DEPTH writes to ``depth_fields/``.
All pipelines use the same path structure within their folder:

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
from dbof.global_dataset_creation.zarr_dataset_global import GlobalZarrDatasetReader
from dbof.global_dataset_creation.zarr_grid_global import GlobalGridZarrReader
from dbof.io.filesystems import get_filesystem

fs = get_filesystem(s3_endpoint, anon=False)

reader = GlobalZarrDatasetReader(bucket, folder, run_id, dataset_name, fs)
snap = reader.get_snapshot(t=0)  # (C, H, W)
field = reader.get_channel_snapshot(t=0, channel="N2_sfc")  # (H, W)
t_idx = reader.iteration_to_index(iteration=1463616)

grid = GlobalGridZarrReader(bucket, folder, dataset_name="llc4320_grid.zarr", fs=fs)
lon, lat = grid.lon, grid.lat
land_mask = grid.land_mask
ds_grid = grid.to_dataset()
```

---

## Vector fields: orientation and history

**In current stores, `U`, `V`, `oceTAUX`, and `oceTAUY` are true
geographic components** — `U`/`oceTAUX` eastward, `V`/`oceTAUY` northward
— colocated with the tracers at cell centres. The pipelines interpolate
the staggered model components to tracer points and rotate them from the
model (x, y) basis to (east, north) using the grid `CS`/`SN` coefficients
before the face stitch (canonical explanation: the "Vector-handling
policy" in `dbof/utils/faces_to_latlon.py`; proof:
`tests/test_vector_rotation_equivalence.py`, which writes comparison
figures to `tests/output/`).

**Stores generated before July 2026 have known vector artifacts** and
should not be mixed with regenerated data:

- *SURF/OSN `native_fields` (U/V) and `surface_wind` (oceTAUX/Y), both
  pipelines:* the fields were interpolated to tracer points but then
  passed through xmitgcm's mate/vector face stitch, whose one-pixel
  shift-and-pad is only correct for staggered input. Result: the
  **northward** components (V, oceTAUY) are displaced one pixel (~2 km)
  relative to the land mask and all other channels over the rotated half
  of the map (source faces 7–12), plus zero-filled seam lines at the
  rotated facet edges. Eastward components are unaffected.
- *DEPTH `native_fields` (`U_*`, `V_*`):* worse — these channels were
  interpolated but **never rotated**, and were stitched as scalars. Over
  the rotated half of the map U and V are effectively swapped (with a
  sign), i.e. wrong at full amplitude.

Regenerating the affected subsets fixes both. Note the fix also changed
the **channel order** in `native_fields` (now `Theta, Salt, Eta, W, U, V`)
and `surface_wind` (DEPTH: `oceQnet, oceTAUX, oceTAUY, ...`), because
vector fields moved from model channels to computed channels — always
index by the store's `channel_names` attribute, not by position.

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
