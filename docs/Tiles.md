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
| `tile_mapping.py` | Resolve a rectangular-grid `(i, j)` pixel → its enclosing LLC tile (face + face-local slices). |
| `tile_utils.py` | Data loading, the property registry, compute scaffolding, output assembly, and the `run()` orchestrator. |
| `generate_tile.py` | Thin CLI wrapper around `tile_utils.run()`. |

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

Properties are registered in `tile_utils.TILE_PROPERTIES`. Each entry declares
the tracer variables it needs, a compute callback returning a lazy
`(face, k, j, i)` `DataArray`, and output metadata.

| `--property` | Output var | Units | Inputs | Computation |
|--------------|-----------|-------|--------|-------------|
| `density` | `sigma0` | `kg m-3` | `Theta`, `Salt` | JMD95 potential density anomaly (`p=0`) − 1000 |
| `temperature` | `Theta` | `degC` | `Theta` | passthrough |
| `salinity` | `Salt` | `psu` | `Salt` | passthrough |

`density` reuses `utils.physical_calculations.density_of_field`, which calls the
JMD95 equation of state with `p = 0` at every level — i.e. **potential density
referenced to the surface** at every depth.

### Adding a new property

Append one entry to `TILE_PROPERTIES` — no other code changes are needed for any
field that fits the `(face, k, j, i)` shape:

```python
TILE_PROPERTIES["my_field"] = TileProperty(
    name="my_field",
    vars_needed=("Theta", "Salt"),          # tracers to fetch from S3
    out_name="my_field",                     # variable name in the NetCDF
    units="...",
    long_name="...",
    filename_prefix="myfield",               # default-filename prefix
    compute=lambda ds: ...,                  # ds -> lazy DataArray (face,k,j,i)
)
```

---

## CLI usage

The script is importable and runnable as a module:

```bash
python -m dbof.tiles.generate_tile \
    --i 9800 --j 9000 \
    --timestamp '2012-11-09 12:00:00' \
    --property density \
    [--output ./density_tile301_20121109T12.nc] \
    [--s3-config some_override.yaml]
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--i` | yes | rect-grid i coord, `0..17279` (any pixel inside the tile) |
| `--j` | yes | rect-grid j coord, `0..12959` (any pixel inside the tile) |
| `--timestamp` | yes | snapshot time, `'YYYY-MM-DD HH:MM:SS'` (UTC) |
| `--property` | no | `density` (default), `temperature`, or `salinity` |
| `--output` | no | output path; see below |
| `--s3-config` | no | optional legacy YAML override (see [Data source](#data-source)) |

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
    i_rect=9800,
    j_rect=9000,
    timestamp="2012-11-09 12:00:00",
    property="density",       # key into TILE_PROPERTIES
    output=None,              # see output-path rule above
    config_path=None,         # None → canonical LLC_DEPTH source
    clobber=False,            # skip if the output file already exists
    gen_qa_plot=False,        # also write a surface PNG next to the NetCDF
)
```

`run()` returns the absolute `Path` of the written NetCDF, or `None` if the
output already existed and `clobber=False`.

To resolve a tile without doing any I/O:

```python
from dbof.tiles.tile_mapping import rect_ij_to_tile

tile = rect_ij_to_tile(9800, 9000)
print(tile.tile_idx, tile.face_idx, tile.j_face_slice, tile.i_face_slice)
```

---

## Output format

A single-variable NetCDF (`h5netcdf`, zlib level 4, float32):

- **Data variable** — named after the property (`sigma0` / `Theta` / `Salt`),
  dims `(k, j, i)` of size `(51, 720, 720)`. `j`, `i` are **face-local**
  indices.
- **Coordinates**
  - `XC(j, i)`, `YC(j, i)` — 2D longitude / latitude (native grid).
  - `Z(k)` — 1D depth.
  - scalar provenance coords: `tile_index`, `face_index`, `rect_i_start`,
    `rect_j_start`.
- **Attributes** — `timestamp`, `iteration`, `tile_index`, `tile_j_rect`,
  `tile_i_rect`, `face_index`, `rect_i_user`, `rect_j_user`, `property`,
  `source_script`, `git_commit`.

No land masking is applied; cells over land carry whatever the model stored.

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
| `folder` | `LLC4320_v1` (per-timestep `{YYYYMMDDTHH}.zarr` stores) |
| `grid_folder` | `LLC4320` (`grid.zarr`) |

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
- `utils.physical_calculations.density_of_field` — JMD95 potential density.
- `utils.faces_to_latlon.faces_dataset_to_latlon` — face stitching (for the
  tile→face lookup).
- `global_dataset_creation.iterations.mit_date_to_iteration` / `DATE_FMT` —
  date↔iteration conversion (same conventions as the global pipeline; see
  [Iteration / date conventions](Global_Maps.md#iteration--date-conventions)).

---

## Testing

Tests live in `tests/test_generate_tile.py`. Offline tests stub all S3 access
with monkeypatched in-memory datasets and run in seconds; one network test opens
the real `grid.zarr` to validate the `(i, j)` → tile mapping against true
coordinates.

```bash
# Full suite (needs network for the grid.zarr integration test):
conda run -n ocean14 python -m pytest tests/test_generate_tile.py -v

# Offline only:
conda run -n ocean14 python -m pytest tests/test_generate_tile.py \
    --deselect tests/test_generate_tile.py::test_rect_ij_to_tile_against_grid_zarr
```

---

## Notes

- The tile holds `~2 vars × 51 × 720² × 4 B ≈ 210 MB` lazily — trivial in
  memory. The default threaded Dask scheduler is sufficient; no
  `dask.distributed.Client` is needed.
- Materialisation happens in a single `.compute()` (wrapped in a `ProgressBar`),
  matching the one-compute pattern used by the global depth pipeline.

*Generated by JXP and Claude.*
