# Pipeline Data-Flow Diagrams

This page shows how data moves through the pipelines in this repository, as
[Mermaid](https://mermaid.js.org/) flowcharts:

1. **Ingest** — get raw LLC4320 snapshot data into S3 (one-time / as needed).
2. **Grid generation** — build the stitched global grid store (one-time).
3. **Global** — build stitched, rectangular global fields (surface or
   depth-resolved) from the raw data.
4. **Cutouts** — sample small, downsampled image cutouts from the global
   fields for ML training.
5. **Tiles** — extract a single 720×720 tile of one field from the raw
   LLC4320 depth store (a lightweight, on-demand path).

Diagrams 1–2 are one-time prerequisites that produce the raw stores and the
grid. The processing pipelines are then staged **Global → Cutouts**, while
**Tiles** is independent and reads the raw depth store directly.

> Rendering note: these are fenced ```mermaid blocks. GitHub renders them
> natively. The Sphinx / Read the Docs build renders them via
> `sphinxcontrib-mermaid` together with MyST's `myst_fence_as_directive =
> ["mermaid"]` — both wired into `docs/conf.py` and `docs/requirements.txt`.

Node labels reference the real modules/functions so the diagrams double as a
code map. S3 locations use the `dbof` bucket on the NRP/Nautilus endpoint
`https://s3-west.nrp-nautilus.io` unless noted.

---

## 1. Ingest (raw LLC4320 → S3)

How raw snapshot data becomes available to the pipelines. Two independent
sources: MIT's local zarr store transferred to S3 by
`dbof.cli.transfer_llc4320`, and the public OSN kerchunk endpoints read
directly at run time.

```mermaid
flowchart LR
    MIT["MIT local zarr store<br/>(LLC4320 native output)"]
    MIT -->|"dbof.cli.transfer_llc4320"| RAWS["S3 dbof/LLC4320_RAW/SURFACE/<br/>{date}.zarr (surface + forcing)"]
    MIT -->|"dbof.cli.transfer_llc4320"| RAWD["S3 dbof/LLC4320_RAW/DEPTH/<br/>{date}.zarr (51-level column)"]
    OSN["OSN kerchunk (public)<br/>mghp.osn.xsede.org<br/>llc_surf + llc_wind"]

    RAWS --> CG["→ Global (SURF forcing)"]
    RAWD --> CGD["→ Global (DEPTH) and Tiles"]
    OSN --> CO["→ Global (OSN / SURF core)"]
```

---

## 2. Grid generation (one-time)

The static stitched grid store, built once and reused read-only by all
three processing pipelines.

```mermaid
flowchart LR
    OSNG["OSN kerchunk grid<br/>(13 LLC faces)"]
    OSNG -->|"dbof.cli.generate_grid_global"| GRIDZ["S3 grid store<br/>grid.zarr / llc4320_grid.zarr<br/>stitched (12960 x 17280)<br/>XC, YC, dxC, dyC, hFacC, CS, SN, ..."]
    GRIDZ --> U1["→ Global: grid_setup (xgcm Grid)"]
    GRIDZ --> U2["→ Cutouts: GlobalGridZarrReader"]
    GRIDZ --> U3["→ Tiles: _load_grid_for_tile"]
```

---

## 3. Global pipeline

Entry point: `dbof.cli.generate_global:main` (batch driver:
`dbof.cli.run_all_subsets`). Three mutually exclusive pipeline variants —
**SURF**, **OSN**, **DEPTH** — differ only in where a snapshot's raw data is
read from and whether depth strategies run. Inputs (the raw stores and the
grid) come from the **Ingest** and **Grid generation** diagrams above.

```mermaid
flowchart TD
    RAW[("S3 LLC4320_RAW<br/>SURFACE or DEPTH /{date}.zarr")]
    OSN[("OSN kerchunk<br/>llc_surf + llc_wind")]
    GRIDZ[("S3 grid store<br/>grid.zarr")]

    subgraph LOAD["Per-snapshot load — generate_global.load_snapshot"]
        direction TB
        SEL["variable_selection.<br/>required_model_variables<br/>(channels → raw vars)"]
        DEPTHL["DEPTH: get_llc_timestep_data<br/>(S3 LLC4320_RAW/DEPTH)"]
        SURFL["SURF: OSN core + S3 forcing"]
        OSNL["OSN: get_remote_llc_data<br/>(+ wind kerchunk)"]
        SEL --> DEPTHL & SURFL & OSNL
    end

    RAW --> DEPTHL
    RAW --> SURFL
    OSN --> SURFL
    OSN --> OSNL
    GRIDZ --> GRIDSET["grid_setup:<br/>xgcm Grid (13 faces)"]

    DEPTHL & SURFL & OSNL --> MERGE["ds_merge<br/>(face, k, j, i) + grid"]
    GRIDSET --> MERGE

    MERGE --> COMPUTE["Compute subset fields<br/>subset_definitions.get_compute_fn →<br/>surface_subsets / depth_subsets<br/>compute_*(ds_merge, grid, channels)"]
    COMPUTE --> DEPTHSTRAT{"DEPTH<br/>pipeline?"}
    DEPTHSTRAT -->|yes| STRAT["apply_depth_strategies<br/>3D → 2D at sfc / z25m /<br/>mld / mld_mean"]
    DEPTHSTRAT -->|no| STITCH
    STRAT --> STITCH["faces_to_latlon.stitch_and_mask<br/>13 faces → (C, 12960, 17280)<br/>land→NaN, rotate U/V,τ to geographic"]

    STITCH --> WRITE["GlobalZarrDataset write"]
    WRITE --> OUT["S3 dbof/{surface_fields or depth_fields}/<br/>{run_id}/{date_prefix}/{subset}.zarr<br/>attrs: channel_names, iteration"]
    OUT -->|"optional, dbof.cli.zarr_to_netcdf"| NC["per-channel NetCDF<br/>LLC4320_{date}_{channel}_{run_id}.nc"]
```

**Subsets** (each written as its own `{subset}.zarr`): `native_fields`,
`surface_wind`, `icearea`, `frontal_structure`, `kinematic`,
`frontogenesis` (all pipelines) plus `stratification`, `vertical_shear`,
`mixing_parameters`, `ertel_pv`, `buoyancy_fluxes`, `energetics` (DEPTH
only). See `docs/Global_Maps.md` for the full channel lists.

---

## 4. Cutouts pipeline

Entry point: `dbof.cli.generate_cutout_dataset:main` →
`dbof.cutout_dataset_creation.processing.run`. **Consumes the global
pipeline's output** (the `{date_prefix}/{subset}.zarr` stores, which must
include `gradb2` for sampling and `SIarea` for the ice mask) plus the
stitched global grid store. Produces downsampled image cutouts + metadata.

```mermaid
flowchart TD
    subgraph IN["Inputs (global-pipeline output)"]
        GLOB["S3 global stores<br/>{folder}/{date_prefix}/{subset}.zarr<br/>channels incl. gradb2, SIarea"]
        GRID["S3 stitched grid<br/>llc4320_grid.zarr"]
    end

    CFG["config.load_config → JobConfig"] --> RUN["processing.run"]
    RUN --> DISC["global_input.resolve_date_prefixes<br/>(discover date folders)"]
    RUN --> VERIFY["global_input.verify_feature_channels /<br/>verify_required_channels"]
    RUN --> SETUP["set_up_grid_data_and_land_masks<br/>GlobalGridZarrReader +<br/>static_masks.generate_halo_land_mask"]
    GRID --> SETUP

    DISC --> LOOP{{"for each date_prefix<br/>(snapshot)"}}
    SETUP --> LOOP

    LOOP --> LOADS["global_input.load_snapshot_features<br/>open_feature_readers → per-subset<br/>GlobalZarrDatasetReader.get_channel_snapshot<br/>→ ds_merge (j, i) + grid"]
    GLOB --> LOADS

    LOADS --> MASK["build_sampling_mask<br/>ice halo (SIarea>0) AND land halo"]
    MASK --> SAMPLE["sample_cutout_centers_with_loggradb<br/>log10(gradb2) → weighted_coordinate_sampling.<br/>weighted_sample_on_grid → center (j,i) list"]

    SAMPLE --> CREATE["dask_pipeline.run_cutout_creation"]
    CREATE --> EXT["spatial_cutouts.get_lat_lon_extents_of_cutout<br/>(km → index extent per center)"]
    EXT --> IMG["create_image_cutouts_batch_as_tensors_dask<br/>slice channels → (C, H, W)"]
    IMG --> DOWN["downsample_and_write_cutout_lazy<br/>area-interp → (C, res, res)"]

    DOWN --> ZW["ZarrDataset.append (image + uuid)"]
    DOWN --> MW["metadata.MetadataWriter (parquet)"]

    ZW --> OUTZ["S3 {output}/{run_id}/{dataset}.zarr<br/>images (N,C,res,res) + image_ids"]
    MW --> OUTM["S3 {output}/{run_id}/metadata/*.parquet<br/>center lat/lon, km extents, log_grad_b, time"]

    OUTZ --> READ["cutout_dataset_access.access.<br/>load_cutout_dataset → CutoutDataset<br/>(images row-aligned to metadata)"]
    OUTM --> READ
```

---

## 5. Tiles pipeline

Entry point: `dbof.cli.generate_tile:main` → `dbof.tiles.tile_utils.run`.
Independent, on-demand path: extract **one 720×720 tile** of a single
property from the raw LLC4320 **depth** store and write it to NetCDF with
full provenance. No global store required.

```mermaid
flowchart TD
    subgraph IN["Inputs (raw LLC4320 depth store)"]
        TS["S3 dbof/LLC4320_RAW/DEPTH/<br/>{date}.zarr (Theta, Salt, ...)"]
        GRIDT["S3 dbof/LLC4320/grid.zarr"]
    end

    CLI["generate_tile.main<br/>(--i/--j or --lon/--lat, --timestamp, --property)"] --> RUN["tile_utils.run"]

    RUN --> LL{"lon/lat<br/>given?"}
    LL -->|yes| L2R["tile_mapping.latlon_to_rect_ij"]
    LL -->|no| RESOLVE
    L2R --> RESOLVE["tile_mapping.rect_ij_to_tile<br/>→ TileInfo (face_idx,<br/>j/i_face_slice, tile_idx)"]

    RESOLVE --> LOADG["_load_grid_for_tile (eager)<br/>XC, YC, Z sliced to tile"]
    RESOLVE --> LOADT["_load_tracers_for_tile (lazy)<br/>vars_needed sliced to face+tile"]
    GRIDT --> LOADG
    TS --> LOADT

    LOADT --> COMPUTE["compute_tile_property<br/>TILE_PROPERTIES[prop].compute(...)<br/>e.g. calculated_fields_at_depth.<br/>potential_density_anomaly_3d<br/>→ single .compute()"]

    COMPUTE --> BUILD["_build_output_dataset<br/>field + XC/YC/Z coords +<br/>provenance attrs (timestamp,<br/>iteration, tile_index, face_index,<br/>property, git_commit)"]
    LOADG --> BUILD

    BUILD --> WRITE["to_netcdf (h5netcdf, zlib, float32)<br/>{prefix}_tile{idx}_{date}.nc"]
    BUILD -->|"optional"| QA["_qa_plot → surface pcolormesh .png"]
```

**Registered properties** (`tile_utils.TILE_PROPERTIES`): `density`
(σ₀ via JMD95), `temperature` (Θ), `salinity` (S). Add a new property by
registering a `TileProperty` with its `vars_needed` and a `compute`
callback.
