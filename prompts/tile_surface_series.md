# tile-surface-series: hourly gradb2 on one tile, from surface-only data

**Repos**: `llc4320-native-grid-preprocessing` (this one) and `fronts`.
**Goal**: `gradb2` for tile 330 (California Current / Monterey Bay), surface
only, 504 consecutive hourly snapshots (3 weeks), then front finding on it.

**Status: IMPLEMENTED.** See the Logs section for what changed and how it was
verified.

## Question answered first: does the tile code work for surface-only data?

**Yes, and the physics needed nothing at all.**  `gradb2` is the best case in
the registry.  Its chain is

```
field_registry["gradb2"] -> calculate_fields.grad_b2 -> buoyancy_of_field
    -> native_gradient.calculate_grad_squared_tracer
```

and the operator at the bottom is four xgcm calls:

```python
ds_dx_M = grid.diff(ds_value, 'X') / ds_grid.dxC
ds_dy_M = grid.diff(ds_value, 'Y') / ds_grid.dyC
dx2_c = grid.interp(ds_dx_M ** 2, 'X', boundary='fill')
dy2_c = grid.interp(ds_dy_M ** 2, 'Y', boundary='fill')
```

No `face`, no `face_connections`, no `k`, and -- because the squares are taken
BEFORE the centre interpolation -- no `CS`/`SN` either (`docs/Gradients.md`
81-98: `|grad s|^2` is rotation-invariant).  The only inputs are `Theta`,
`Salt`, `dxC`, `dyC`, plus `hFacC` for the land mask.

`tests/test_calculate_fields.py` already asserts
`grad_b2(ds2d) == grad_b2(ds3d).isel(k=0)`, so "surface-only" is a tested
contract rather than a hope.  Everything downstream in `tile_utils` was
already `"k" in dims`-conditional (`_tile_indexer`, `compute_tile_property`,
`_build_output_dataset`, `_qa_plot`), so a 2D input produces a clean `(j, i)`
output with `XC`/`YC` and no `Z`.

The price of a tile is the boundary: the xgcm grid is built with
`use_connections=False`, so `boundary='fill'` contaminates the rim and
`gradb2` carries `edge_margin=3`.  Three cells on each side of 720 -- 0.4%.

## The real constraint: where 504 hourly Theta/Salt come from

| source | Theta/Salt | dates available |
|---|---|---|
| `LLC4320_RAW/DEPTH` | yes, 51 levels | **1** (`20121109T12`) |
| `LLC4320_RAW/CHUNKS/monterey_bay` | yes, 51 levels | **17** |
| OSN kerchunk `llc_surf` | yes, surface only | **hourly**, 2011-09-13 -> 2012-11-15 |

Only OSN has consecutive hours, and it is what the SURF global pipeline
already reads.  It is also *per face*: one kerchunk JSON per face per
iteration, so asking for face 10 opens ONE reference file instead of 13.

## What changed

### `llc4320-native-grid-preprocessing`

1. **`tiles/tile_utils.py` -- an OSN branch.**  `_resolve_s3_source` takes a
   `pipeline` argument and returns `{"kind": "OSN", ...}` for the kerchunk
   store.  A new `_open_full_grid` picks the right grid reader
   (`get_remote_gridfile` + `process_llc4320_grid` for OSN,
   `get_llc_depth_gridfile` + `process_llc4320_3d_grid` otherwise) and is
   shared by `_load_grid_for_tile` and `latlon_to_rect_ij`.
   `_load_tracers_for_tile` grows the matching branch:
   `get_remote_llc_data(endpoint, osn_date_to_iteration(ts), [face])`, which
   already returns `isel(time=0, k=0, k_l=0)`.  All of that is code the SURF
   pipeline was calling anyway; the tile loaders just learned to call it.

2. **`_ensure_comodo_attrs`.**  `_build_tile_context` builds xgcm from the
   grid dataset ALONE and raises if X/Y are undetectable.  Only
   `get_llc_depth_gridfile` stamps the comodo `axis` attrs; the OSN path goes
   through `process_llc4320_grid`, whose `reset_coords()` can drop them.  The
   helper stamps only dims that are missing an `axis` attr, so a store that
   declares its own `c_grid_axis_shift` keeps its own convention.

3. **`run_series(...)` -- the time loop.**  `run()` is one snapshot, and every
   call re-resolves the tile (rebuilding the 12960x17280 rect lookup, which
   `tile_mapping` deliberately does not cache), re-downloads the tile grid,
   and rebuilds the xgcm grid.  504 steps through `run()` would pay all three
   504 times.  `run_series` hoists them: per timestamp only the tracer read,
   the merge, the compute and the write remain.  `_build_tile_context` gained
   an optional `grid=` argument so the hoisted xgcm grid can be handed back
   in; `run()` is untouched.
   One NetCDF per timestamp -- the file-per-snapshot shape the front finder
   already expects -- not one file with a `time` dim.  `output_paths=` lets a
   consumer dictate the filenames, and `continue_on_error=` keeps a
   multi-week run alive through a bad hour.

4. **`cli/generate_tile.py`** -- `--pipeline {DEPTH,SURF,OSN}`,
   `--dates-config` (reads `data.date_iterations`, the same key the global and
   transfer configs use) and `--continue-on-error`.  `--dates-config`
   switches the CLI to `run_series`; `--timestamp` still drives a single
   snapshot and the default is still `DEPTH`, so nothing existing changes
   behaviour.

5. **`configs/tiles/tile330_gradb2_osn.yaml`** -- 504 hourly dates,
   2012-06-29 00:00 -> 2012-07-19 23:00 UTC.  Deliberately read by BOTH
   repos: `generate-tile --dates-config` takes `data.date_iterations`, and
   `fronts.properties.run.read_build_config` takes the whole file.  The window
   sits inside OSN coverage and straddles the 17 dates already in
   `LLC4320_RAW/CHUNKS/monterey_bay`, so those hours can be recomputed from
   the native chunk store as an independent cross-check.

### `fronts`

Nothing in `fronts/finding/` -- see the sibling note,
`fronts/prompts/fronts_tile_finding.md`.

## What did NOT change

`field_registry`, `calculate_fields`, `native_gradient`, `tile_mapping`,
`compute_tile_property`, `_build_output_dataset`, `run()`, and the whole
global pipeline.  The tile module gained plumbing only, never physics -- the
same rule `prompts/tiles_fields.md` set.

## Running it

```bash
# gradb2 only, straight to a directory
generate-tile --lon -121.9 --lat 36.8 --property gradb2 \
    --pipeline OSN --continue-on-error \
    --dates-config configs/tiles/tile330_gradb2_osn.yaml \
    --output /path/to/gradb2_tile330

# or the full fronts workflow (steps 1-3), same config file
python build_v5.py 1 <path>/configs/tiles/tile330_gradb2_osn.yaml
python build_v5.py 2 <path>/configs/tiles/tile330_gradb2_osn.yaml
python build_v5.py 3 <path>/configs/tiles/tile330_gradb2_osn.yaml
```

## Check these before a full 504-step run

1. **Tile 330 == the `monterey_bay` chunk?**  `330 = 13*24 + 18` -> rect
   `j 9360:10080`, `i 12960:13680`.  `transfer.chunk_selection.resolve_chunk`
   at (36.8, -121.9) returns face 10, `j 0:720`, `i 2880:3600`, and the two
   lon/lat boxes agree (-127.99..-113.0, 26.66..38.2).  Confirm once with
   `rect_ij_to_tile(13320, 9720)` -- it is the assumption the cross-check in
   (5) rests on.
2. **Comodo attrs survive the OSN grid.**  `set_up_grid_osn` builds an xgcm
   grid from the same `process_llc4320_grid` output, so they should be there;
   `_ensure_comodo_attrs` is the safety net.  Worth one interactive check that
   the net is not doing the work.
3. **`c_grid_axis_shift` sign.**  RESOLVED 2026-09-01: `-0.5` is right, as
   `fronts/llc/tiles.py` had it.  It is a signed direction (xgcm maps -0.5 ->
   'left'), MITgcm puts U on the west face, and xmitgcm and the OSN grid both
   stamp -0.5; measured directly from the corner geometry too.  The value now
   lives once, in `llc4320_ingestion.grid.COMODO_COORD_META` -- import it
   rather than writing it out.  See docs/Grid.md and
   `dev/verify_grid_stores.py`.
4. **One hour first.**  Run a single timestamp end to end, look at the QA plot
   (`--qa-plot`), confirm the 3-cell NaN rim and that land is masked, then let
   the 504 go.
5. **Cross-check against the chunk store.**  For an hour that exists in both,
   compute `gradb2` from `LLC4320_RAW/CHUNKS/monterey_bay` (via
   `fronts.llc.tiles.chunk_loader`) and from OSN, and compare the interiors.
   Different sources, different readers, same physics -- a real test of the
   new branch.

## Logs

### 2026-08-26 (Implemented the surface-only tile series and its OSN source)

Built everything described above.  Notes worth keeping:

- **The physics needed no edits, and that was verifiable up front.**  The
  dimension-agnosticism contract in `tests/test_calculate_fields.py` covers
  `grad_b2` explicitly, so the only question was plumbing.
- **`_build_tile_context` builds xgcm from `ds_grid_tile` alone** (rewritten
  at `01a53ae`).  Two test modules had not caught up and were failing before
  any of this work:
  `tests/test_tile_field_registry.py` passed `xr.Dataset()` as the grid, and
  `tests/test_generate_tile.py::_make_synthetic_grid_face` built a grid with
  no comodo attrs.  Both now hit
  `RuntimeError: xgcm grid for the tile is missing axes ['X', 'Y']`.
  Fixed minimally (pass the synthetic tile dataset as the grid; annotate the
  synthetic grid face) -- production was never affected, but there was no
  green baseline to verify against otherwise.
- **New `tests/test_tile_series_surface.py`** (6 tests, fully offline) pins
  the three things that actually matter: 2D in -> 2D out with no `Z`; the
  grid is loaded exactly ONCE for the whole series; and `run_series` agrees
  with `run` cell for cell on the same snapshot.  Plus the skip/clobber,
  explicit-output-paths and continue-on-error behaviours.
- **Provenance**: the `iteration` attr now records the iteration of the store
  the data actually came from -- OSN references carry the
  `FIRST_WIND_RECORD_OFFSET` shift, S3 stores carry raw MIT iterations.
  `generate_global` already logs it that way for SURF.
- **Verification**: `tests/test_generate_tile.py`,
  `test_tile_field_registry.py`, `test_calculate_fields.py`,
  `test_grad_squared_staggered.py` -> 157 passed (network-gated
  `test_rect_ij_to_tile_against_grid_zarr` deselected, offline);
  `test_tile_series_surface.py` -> 6 passed.  The OSN branch itself is NOT
  covered by an offline test -- it is a network seam, and the check in (5)
  above is the real test of it.
