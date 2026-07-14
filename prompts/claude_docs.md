# Doc prompts

## Readthedocs

## Please add the necessary files to expose the docs/ in llc4320-native-grid-preprocessing Repository on readthedocs.

## Mermaid diagram

1. Please generate a Mermaid diagram that shows the flow of data through the pipeline.  You may need to make separate ones for the the cutouts, global, and tiles.  Put these in the docs/ directory in a file named `mermaid_diagrams.md`.  If you have any questions, put them under the Q&A section.  Log your work.

2. I have answered the Q&A section.  Please review and continue on with the task.  Log your work.

## Q&A

- **Mermaid rendering in the docs build.** The diagrams live in
  `docs/mermaid_diagrams.md` as fenced ```mermaid blocks (GitHub renders
  these natively). For the Read the Docs / Sphinx build to render them, the
  build needs `sphinxcontrib-mermaid` (add to `docs/requirements.txt` and to
  `extensions` in `docs/conf.py`) or, if using MyST-Markdown, the `mermaid`
  fence directive enabled. Want me to wire that in as part of the Read the
  Docs setup item above? (Left out for now since that's a separate prompt.)
>A. Yes, add to requirements.txt and extensions in conf.py
- **Scope.** I produced three diagrams as suggested (global, cutouts,
  tiles) and added `mermaid_diagrams.md` to the `docs/` toctree
  (`docs/index.md`). The global diagram also folds in the one-time ingest
  (`transfer_llc4320`) and grid-generation setup, since those produce the
  raw inputs. Say the word if you'd prefer those split into a separate
  "ingest" diagram.
>A. Yes, split into ingest and grid-generation diagrams.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-14 (Mermaid data-flow diagrams for the three pipelines)

Executed the 1st item under **Mermaid diagram**: created
`docs/mermaid_diagrams.md` with three Mermaid `flowchart` diagrams — one
each for the **global**, **cutouts**, and **tiles** pipelines — and added
the page to the `docs/index.md` toctree.

**How I traced the flows.** Fanned out three parallel `Explore` subagents,
one per pipeline, to map entry points → S3/zarr I/O → processing →
outputs against the real code. For the **cutouts** pipeline the explorer
surfaced some pre-refactor code (the old `generate_front_training_data` /
`spatial_patches` / `native_grid_masks` path, ice-from-`Theta`), so I used
the current `cutouts_data_v2` flow instead (which I had reviewed in depth
for PR #19): global-store input via `global_input.load_snapshot_features`,
`SIarea`-based ice halo in `build_sampling_mask`, `static_masks` +
`spatial_cutouts`, and `gradb2`-driven `weighted_sample_on_grid`.

**Diagram contents (nodes map to real modules/functions):**
- *Global* — `generate_global.load_snapshot` (SURF/OSN/DEPTH variants),
  `variable_selection`, subset compute fns, `apply_depth_strategies`
  (DEPTH only), `stitch_and_mask` → `GlobalZarrDataset` →
  `{surface_fields or depth_fields}/{run_id}/{date_prefix}/{subset}.zarr`,
  plus optional `zarr_to_netcdf`; includes the one-time `transfer_llc4320`
  ingest and `generate_grid_global` setup.
- *Cutouts* — consumes the global stores + stitched grid;
  `processing.run` → land/ice halo `build_sampling_mask` →
  `sample_cutout_centers_with_loggradb` → `run_cutout_creation`
  (extents → batched dask image creation → downsample) →
  cutout zarr (images + image_ids) + parquet metadata → `access`.
- *Tiles* — `generate_tile` → `rect_ij_to_tile` (TileInfo) →
  eager grid + lazy tracer load sliced to one 720×720 face tile →
  `compute_tile_property` (e.g. `potential_density_anomaly_3d`) →
  NetCDF with provenance attrs (+ optional QA png).

**Validation.** No local Mermaid renderer (`mmdc`) available and no network
for `npx`, so I ran a structural check on the three fenced blocks: even
quote counts and balanced `[] {} ()` in all three, arrows/subgraphs
present. Hardened two node labels by replacing the literal `|` (which is
Mermaid's edge-label delimiter) with "or".

**Open items** (recorded under Q&A): the Sphinx/Read-the-Docs build needs
`sphinxcontrib-mermaid` (or MyST's `mermaid` fence) enabled to render the
diagrams — flagged there and in a note at the top of
`mermaid_diagrams.md`, but deferred since it belongs to the Read the Docs
setup item.

**What I learned.** The three pipelines are cleanly staged: Global writes
stitched `(C, 12960, 17280)` per-subset zarr stores keyed by
`{run_id}/{date_prefix}`; Cutouts is a pure *consumer* of those stores
(no raw-LLC access — it reads channels back through
`GlobalZarrDatasetReader`); Tiles is the odd one out, bypassing the global
store entirely to pull a single tile straight from the raw depth store —
which is why it reuses `calculated_fields_at_depth` directly rather than
the subset compute machinery.

### 2026-07-14 (Wired Mermaid into the docs build; split ingest/grid diagrams)

Executed the 2nd item under **Mermaid diagram**, acting on the two answered
Q&A items.

**Q&A #1 — render Mermaid in the Sphinx / Read the Docs build.**
- `docs/requirements.txt`: added `sphinxcontrib-mermaid>=0.9`.
- `docs/conf.py`: added `"sphinxcontrib.mermaid"` to `extensions`, and set
  `myst_fence_as_directive = ["mermaid"]` so the fenced ```mermaid blocks
  in `mermaid_diagrams.md` are routed to the mermaid directive (MyST
  otherwise renders them as literal code). Updated the rendering note at the
  top of `mermaid_diagrams.md` to say it's now wired in.

**Q&A #2 — split the one-time setup into its own diagrams.** Pulled the
`INGEST` subgraph out of the Global diagram and made two standalone
prerequisite diagrams:
- **§1 Ingest** — `transfer_llc4320` (MIT local zarr → S3
  `LLC4320_RAW/SURFACE` and `/DEPTH`) and the OSN kerchunk source, with
  arrows to their downstream consumers (Global variants, Tiles).
- **§2 Grid generation** — `generate_grid_global` → stitched `grid.zarr`,
  consumed read-only by Global (`grid_setup`), Cutouts
  (`GlobalGridZarrReader`), and Tiles (`_load_grid_for_tile`).

The Global diagram (now §3) starts from those raw stores + grid as
cylinder input nodes. Renumbered sections to 1–5 (Ingest, Grid generation,
Global, Cutouts, Tiles) and updated the intro list.

**Verification (this time I could actually build).** `ocean14` has both
`sphinx` (9.1) and `sphinxcontrib.mermaid`, so I ran a full
`sphinx-build -b html`:
- Build succeeded; `mermaid_diagrams.html` produced.
- All **5** fences rendered as `<pre class="mermaid">` blocks (i.e. the
  directive fired — not literal code), and `mermaid.js` is loaded +
  `mermaid.initialize` present.
- **No** warnings referencing `mermaid_diagrams.md`, `conf.py`, or
  `sphinxcontrib`. (The build does emit pre-existing, unrelated warnings —
  missing toctree titles for `Preprocess_LLC_Data`/`Halo_Masking` and a few
  `Tiles.md` xref targets — left untouched; they belong to the Read the Docs
  setup item.)
- Re-ran the structural check: 5 blocks, even quotes and balanced
  `[] {} ()` (including the new `[(...)]` cylinder nodes).

**What I learned.** MyST does **not** hand fenced ```mermaid blocks to
`sphinxcontrib-mermaid` on its own — the missing link is
`myst_fence_as_directive = ["mermaid"]`. With just the extension installed,
the diagrams would silently render as literal code blocks; both pieces are
required.