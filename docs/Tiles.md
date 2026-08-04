# Single-Tile 3D Property Extraction

**Extract one 720×720 × 51-level tile of a single property from one LLC4320
snapshot, on the native grid, written to NetCDF.**

Where the [global maps](Global_Maps.md) pipeline stitches all 13 LLC4320 faces
into full `(12960, 17280)` 2D arrays, the **tiles** pipeline does the opposite:
it pulls a single depth-resolved `(k=51, j=720, i=720)` block — one native zarr
chunk in spatial extent, the full water column — for one property and one
timestamp. This is the natural unit for spot-checks, local analysis, and
training-cutout generation, and it costs exactly one S3 GET per variable.

The native grid geometry is preserved — no interpolation to a regular lat/lon
grid is performed. The 2D `XC`/`YC` longitude/latitude of the tile are carried
along as coordinates.

Code lives in `src/dbof/tiles/`:

| Module | Responsibility |
|--------|----------------|
| `dbof/tiles/tile_mapping.py` | Resolve a rectangular-grid `(i, j)` pixel → its enclosing LLC tile (face + face-local slices). |
| `dbof/tiles/field_registry.py` | The property registry: one `TileProperty` per subset channel, each pointing at the canonical compute function. |
| `dbof/tiles/tile_utils.py` | Data loading, geographic `(lon, lat)`→`(i, j)` resolution, tile-context assembly (local xgcm grid), compute scaffolding, output assembly, and the `run()` orchestrator. |
| `dbof/cli/generate_tile.py` | Thin CLI wrapper around `tile_utils.run()`; installed as the `generate-tile` console script. |

---

## Tile geometry

A **tile** is one 720×720 spatial block on the rectangular LLC4320 output grid,
spanning all 51 depth levels.

- Global rectangular grid: `H = 3·4320 = 12960` rows × `W = 4·4320 = 17280`
  columns.
- The rect grid divides into `18` tile-rows × `24` tile-cols = **432 tiles**,
  indexed flat in row-major order:

  ```
  tile_idx = tile_j_rect * 24 + tile_i_rect      # 0 .. 431
  tile_j_rect = j_rect // 720                     # 0 .. 17
  tile_i_rect = i_rect // 720                     # 0 .. 23
  ```

- **You may pass any pixel inside the desired tile** — the code floors the
  `(i, j)` you give to the enclosing 720×720 tile boundary. `(0, 0)` and
  `(719, 719)` both resolve to tile `0`; `(17279, 12959)` resolves to tile `431`.

Because rect tiles align to LLC face-chunk boundaries and each face is 6×6
tiles, a single rect tile always lives entirely on **one** LLC face — possibly
with a per-face rotation that swaps/reverses the face-local `j` and `i` axes
relative to the rect grid. `rect_ij_to_tile()` resolves that rotation
automatically (see below) and returns a `TileInfo`:

| Field | Meaning |
|-------|---------|
| `tile_idx` | flat row-major tile index, 0..431 |
| `tile_j_rect`, `tile_i_rect` | rect-tile row / column |
| `rect_j_slice`, `rect_i_slice` | rect-grid 720-wide ranges |
| `face_idx` | LLC face (0..12) holding the data |
| `j_face_slice`, `i_face_slice` | face-local 720-wide slices on that face |

### How the face mapping works

`tile_mapping` synthesises three small int arrays of shape
`(13, 4320, 4320)` — each pixel holds, respectively, its face index, its
face-local `j`, and its face-local `i` — then runs them through the **same**
`utils.faces_to_latlon.faces_dataset_to_latlon` stitching the real data uses.
Sampling the stitched arrays over the tile's rect slice reveals which face and
which face-local `(j, i)` range the tile occupies, transparently handling
rotations. The lookup arrays are deterministic and cached at module level
(built once per process; the stitch costs a few seconds).

---

## Properties

Properties are registered in `dbof/tiles/field_registry.py`
(`TILE_PROPERTIES`, re-exported by `tile_utils` for backward
compatibility). There is one entry per **subset channel** — every field
of every SURF and DEPTH subset in
`global_dataset_creation.subset_definitions` can be extracted as a tile
under its exact channel name (`density`, `N2`, `relative_vorticity`,
`frontogenesis_tendency`, `oceTAUX`, ...). The legacy names
`temperature` → `Theta` and `salinity` → `Salt` still work as aliases.

Each entry declares the S3 variables it needs, a compute callback
`compute(ds_merge, grid) -> lazy DataArray`, output metadata, and an
`edge_margin` (see below). **Physics never lives in the registry**:
every callback points at the single canonical implementation in
`preprocessing.calculate_fields` / `calculate_fields_at_depth`; only
passthroughs and trivial component extraction are local. The
authoritative field list, units, and equations are documented per
subset in `subset_definitions.py` and the field-validation notebooks —
`tests/test_tile_field_registry.py` asserts that every subset channel
has a registry entry, so the registry cannot silently fall behind.

### Tile context: a local xgcm grid

Fields with horizontal stencils (gradients, vorticity, frontogenesis,
...) need an xgcm grid, and two loading details make that work:

1. **Staggered dims.** `U` lives on `(k, face, j, i_g)`, `V` on
   `(k, face, j_g, i)`, and grid metrics use both (`dxC` on
   `(j, i_g)`, `rAz` on `(j_g, i_g)`, ...). One shared indexer
   (`tile_utils._tile_indexer`, ported from the `tiles_field` branch)
   slices `i_g`/`j_g` alongside `i`/`j` in **both** the tracer and
   grid loaders — chunk-aligned tiles take the *same* slice, so
   staggered variables come down at tile size rather than full-face.
2. **A local grid.** The tile is a **single face**, so
   `tile_utils._build_tile_context` merges the tile tracers + grid and
   builds a **local** grid with `use_connections=False` — no face
   connections exist for one tile. Vertical coordinate vars
   (`Z`/`Zl`/`Zu`/`Zp1`/`drF`) are dropped from what xgcm sees, same
   as `grid_setup.set_up_grid_depth` in the DEPTH pipeline; the merged
   dataset handed to compute callbacks keeps them. Consequence: horizontal stencils are
invalid in a rim of cells at the tile boundary. Each registry entry
declares that rim as `edge_margin` (0 = rim-free passthroughs /
verticals, 1 = staggered-point interpolation only, 3 = horizontal
derivative / Jacobian chains); `compute_tile_property` sets the rim to
NaN — the output keeps its full 720×720 shape, and the value is
recorded as an `edge_margin` attribute on the output variable.

### Vector rotation

Native `U`/`V`/`oceTAUX`/`oceTAUY` are staggered and aligned with the
**model** x/y axes, which on the rotated LLC faces point ~north, not
east. Vector entries are therefore **never passthroughs**: the `U`/`V`
and `oceTAUX`/`oceTAUY` entries interpolate to tracer points and rotate
to true east/north via the grid's `CS`/`SN` coefficients (the canonical
`geographic_velocity` / `geographic_wind_stress` functions). The
rotation is pointwise, so it works identically on a single tile as in
the global pipeline (acceptance-tested with a synthetic CS=0/SN=1
face in `tests/test_tile_field_registry.py`).

### 2D fields

Inherently-2D channels (`Eta`, `mixed_layer_depth`, `ml_heat_content`,
`KE`, the wind/ice channels, ...) produce `(j, i)` outputs with no `k`
dimension or `Z` coordinate — the output writer handles both cases.
Surface-subset channels are computed **at every depth level** where the
formula is depth-valid (e.g. `density`, `relative_vorticity`); wind/ice
channels are 2D by nature.

### Adding a new property

Append one entry to `TILE_PROPERTIES` in `field_registry.py` — no other
code changes are needed:

```python
TILE_PROPERTIES["my_field"] = TileProperty(
    name="my_field",
    vars_needed=("Theta", "Salt"),   # S3 variables to fetch
    out_name="my_field",             # variable name in the NetCDF
    units="...",
    long_name="...",
    filename_prefix="myfield",       # default-filename prefix
    compute=CF.my_field,             # (ds_merge, grid) -> lazy DataArray
    edge_margin=3,                   # invalid boundary rim (cells)
)
```

---

## CLI usage

Installed as the `generate-tile` console script (also runnable as
`python -m dbof.cli.generate_tile`). The tile location is given **either** as a
rect-grid pixel (`--i`/`--j`) **or** as a geographic coordinate (`--lon`/`--lat`,
resolved to the nearest rect pixel via the grid file) — supply exactly one pair:

```bash
# by rect-grid pixel
generate-tile \
    --i 9800 --j 9000 \
    --timestamp '2012-11-09 12:00:00' \
    --property density \
    [--output ./density_tile301_20121109T12.nc] \
    [--clobber] [--qa-plot] [--no-mask-land] [--s3-config some_override.yaml]

# by geographic location (resolved to i, j via grid.zarr)
generate-tile --lon -45.0 --lat 33.0 \
    --timestamp '2012-11-09 12:00:00' --property temperature

# any subset channel works, e.g. stratification / kinematics:
generate-tile --lon 145.0 --lat 35.0 \
    --timestamp '2012-11-09 12:00:00' --property N2
generate-tile --lon 145.0 --lat 35.0 \
    --timestamp '2012-11-09 12:00:00' --property relative_vorticity
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--i` / `--j` | one pair | rect-grid coords, `i∈0..17279`, `j∈0..12959` (any pixel inside the tile) |
| `--lon` / `--lat` | one pair | geographic degrees E / N; resolved to the nearest rect pixel via `grid.zarr` |
| `--timestamp` | yes | snapshot time, `'YYYY-MM-DD HH:MM:SS'` (UTC) |
| `--property` | no | any subset channel name (default `density`); legacy `temperature`/`salinity` aliases accepted |
| `--output` | no | output path; see below |
| `--clobber` | no | overwrite an existing output file instead of skipping |
| `--qa-plot` | no | also write a surface QA plot (PNG) next to the NetCDF |
| `--no-mask-land` | no | skip the default `hFacC == 0` land mask |
| `--s3-config` | no | optional legacy YAML override (see [Data source](#data-source)) |

Exactly one of the `--i`/`--j` or `--lon`/`--lat` pairs must be supplied;
passing neither or both is a CLI error.

**Output path rule** (`--output`):

- omitted → `./{prefix}_tile{tile_idx:03d}_{YYYYMMDDTHH}.nc` in the current
  directory, where `prefix` is `density` / `theta` / `salt`;
- a directory → the default name is placed inside it;
- a file path → used verbatim.

---

## Python API

```python
from dbof.tiles import tile_utils

out_path = tile_utils.run(
    i_rect=9800,              # OR pass lon=/lat= instead (exactly one pair)
    j_rect=9000,
    timestamp="2012-11-09 12:00:00",
    property="density",       # key into TILE_PROPERTIES
    output=None,              # see output-path rule above
    config_path=None,         # None → canonical LLC_DEPTH source
    clobber=False,            # skip if the output file already exists
    gen_qa_plot=False,        # also write a surface PNG next to the NetCDF
    mask_land=True,           # NaN land cells via hFacC==0
)

# ...or select the tile by geographic location:
out_path = tile_utils.run(
    lon=-45.0, lat=33.0,
    timestamp="2012-11-09 12:00:00",
    property="temperature",
)
```

`run()` returns the absolute `Path` of the written NetCDF. If the output
already existed and `clobber=False` the run is skipped, but the existing path
is still returned (the function always returns a `Path`).

To resolve a tile without doing any I/O:

```python
from dbof.tiles.tile_mapping import rect_ij_to_tile

tile = rect_ij_to_tile(9800, 9000)
print(tile.tile_idx, tile.face_idx, tile.j_face_slice, tile.i_face_slice)
```

---

## Output format

A single-variable NetCDF (`h5netcdf`, zlib level 4, float32):

- **Data variable** — named after the property's `out_name` (`sigma0`,
  `Theta`, `N2`, ...), dims `(k, j, i)` of size `(51, 720, 720)` for 3D
  fields or `(j, i)` of size `(720, 720)` for inherently-2D fields.
  `j`, `i` are **face-local** indices.
- **Coordinates**
  - `XC(j, i)`, `YC(j, i)` — 2D longitude / latitude (native grid).
  - `Z(k)` — 1D depth (3D fields only).
  - scalar provenance coords: `tile_index`, `face_index`, `rect_i_start`,
    `rect_j_start`.
- **Attributes** — `timestamp`, `iteration`, `tile_index`, `tile_j_rect`,
  `tile_i_rect`, `face_index`, `rect_i_user`, `rect_j_user`, `property`,
  `edge_margin`, `source_script`, `git_commit`.

**Masking.** Land cells are NaN'd via `hFacC == 0` by default
(`run(..., mask_land=False)` / `--no-mask-land` to disable) — a cheap
safety, since derivative-based fields are pathological over and next to
land. Fields with `edge_margin > 0` additionally have that many
boundary cells set to NaN in `j`/`i` (horizontal stencils are invalid
there on a single-face tile — see
[Tile context](#tile-context-a-local-xgcm-grid)). Values are otherwise
**raw** physical units — no log scaling, clipping, or sign
normalisation.

If `gen_qa_plot=True`, a surface (`k=0`) `pcolormesh` is written as a `.png`
next to the NetCDF, using `XC`/`YC` as the axes.

---

## Data source

By default the tile pipeline reads the canonical **LLC_DEPTH** source —
`dbof.global_dataset_creation.data_sources.get_data_source("DEPTH")` — the same
full-depth timestep stores the [DEPTH global pipeline](Global_Maps.md) uses:

| Key | Value |
|-----|-------|
| `s3_endpoint` | `https://s3-west.nrp-nautilus.io` |
| `bucket` | `dbof/` |
| `folder` | `LLC4320_RAW/DEPTH` (per-timestep `{YYYYMMDDTHH}.zarr` stores) |
| `grid_folder` | defaults to `folder` (`grid.zarr` in `LLC4320_RAW/DEPTH`) |

These stores must already have been transferred from MIT via
`transfer_llc4320.py` — see
[Global Maps → Preprocessing](Global_Maps.md#preprocessing-transfer_llc4320py).
There is no pre-flight existence check, so a missing transfer surfaces as an S3
read error at runtime.

`--s3-config` (or `config_path=`) accepts a legacy YAML with an `s3_source`
block as an override; when omitted the canonical source above is used.

### Shared building blocks

The tile code deliberately reuses existing repo machinery rather than
re-implementing it:

- `get_raw_data.get_llc_depth_gridfile` / `get_llc_timestep_data` — S3 readers
  (with the depth chunking and cache-disabled storage options).
- `preprocessing.preproc_llc_core_data.process_llc4320_3d_grid` — grid extraction.
- `preprocessing.calculate_fields.potential_density_anomaly` — the
  single canonical σ₀ routine shared with the global depth pipeline (wraps the
  JMD95 `potential_density` function and subtracts `RHO0_REFERENCE`).
- `utils.faces_to_latlon.faces_dataset_to_latlon` — face stitching (for both the
  tile→face lookup and the geographic `(lon, lat)`→`(i, j)` resolver).
- `global_dataset_creation.iterations.mit_date_to_iteration` / `DATE_FMT` —
  date↔iteration conversion (same conventions as the global pipeline; see
  [Iteration / date conventions](Global_Maps.md#iteration--date-conventions)).
- `global_dataset_creation.metadata._git_commit_hash` — git provenance (falls
  back to the old `logging` location before the branch is rebased).

---

## Testing

Tests live in `tests/test_generate_tile.py` (mapping + `run()`
round-trips with all S3 access monkeypatched) and
`tests/test_tile_field_registry.py` (every registry entry executed
end-to-end on a synthetic single-face tile: dims, dtype, NaN edge rim,
subset-channel coverage, aliases, and the CS/SN rotation acceptance
test). Offline tests run in seconds; one network test opens the real
`grid.zarr` to validate the `(i, j)` → tile mapping against true
coordinates.

```bash
# Full suite (needs network for the grid.zarr integration test):
conda run -n ocean14 python -m pytest \
    tests/test_generate_tile.py tests/test_tile_field_registry.py -v

# Offline only:
conda run -n ocean14 python -m pytest \
    tests/test_generate_tile.py tests/test_tile_field_registry.py \
    -k "not grid_zarr"
```

---

## Known issues / Notes

- The tile holds `~2 vars × 51 × 720² × 4 B ≈ 210 MB` lazily — trivial in
  memory. The default threaded Dask scheduler is sufficient; no
  `dask.distributed.Client` is needed.
- Materialisation happens in a single `.compute()` (wrapped in a `ProgressBar`),
  matching the one-compute pattern used by the global depth pipeline.
- **Compute unification.** Every property calculation has exactly one
  implementation, in `preprocessing.calculate_fields` /
  `calculate_fields_at_depth`; the registry's `compute` callbacks point
  straight at those canonical functions, and neither `tile_utils` nor
  `field_registry` holds field math of its own. The generalization to
  every subset channel (the former `tile_fields` follow-up) is done —
  see `prompts/tiles_fields.md` for the design decisions (tile context,
  edge rim, staggered-dim slicing, 2D outputs, vector rotation).

*Generated by JXP and Claude.*
