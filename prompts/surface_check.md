# Surface check

## Goals

Out of curiousity alone, let us see how far we have progressed on DBOF since JXP's initial exploration

## Context

These are the main 3 repositories:


- code in `llc4320-native-grid-preprocessing/src/dbof` with emphasis on the surface calculations
- code in `Oceanography/python/wrangler`
- code in `fronts/fronts` with emphasis on `dbof` and `llc`

## Running Python

If you need to run Python, use the "ocean14" conda environment.

## Prompts

1.  Please examine all of the code in Context, so that it is in your context.  Use Opus, Log your work.  Eventually, I am going to ask you to assess how much farther along the DBOF project is (from the code in `wrangler` and `fronts`), and you will need to know the code in Context.

2.  Ok, now try to answer this question:  "When it comes to surface fields, how much farther along is the `llc4320-native-grid-preprocessing` project compared to the `wrangler` and `fronts` projects?  Use Fable, Log your work.  Consider:
    - Each set of codes ability to generate cutouts for machine learning
    - The range of properties that can be calculated 
    - The sampling of the LLC4320
    - The quality of the calculations
Write a report named `surface_report.md` that answers the question.  Put it in the `reports` directory.

3. Ok, a few additional questions/requests:
    - I can see in my input files in `Oceanography/python/fronts/fronts/runs/prototypes/sst_ssh_unet` that some of them use a "balance" parameter that samples by log10(Divb2).  That is, I don't think 
    the cutout generation was purely uniform in the "old DBOF".  Please check and modify the report 
    as need be.
    - Make a quantitative estimate of the improvement in the quality of the calculations.
Use Fable, Log your work.

4. Please check:
    - Check whether the old stack really mis-estimated the gradients.  The LLC4320 was first interpolated to a constant 2.25km grid and then the gradient was performed.  Why would this lead to such large errors?
Use Fable, Log your work.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-19 (Examined all code in the three Context repositories)

Executed Prompt 1: read all code in `llc4320-native-grid-preprocessing/src/dbof` (emphasis on surface calculations), `wrangler`, and `fronts/fronts` (emphasis on `dbof` and `llc`). Work was done with a combination of direct reading (all surface-calculation modules, `fronts/dbof/*`, `fronts/llc/*`, `wrangler/ogcm/*`, `wrangler/extract/ex_ogcm.py`, `wrangler/preproc/pp_ogcm.py`) and three exhaustive Opus subagent sweeps (one per repo, every module read). Totals: ~15.6k lines in `src/dbof`, ~3.3k in `wrangler`, ~16.2k in `fronts/fronts`.

**1. `llc4320-native-grid-preprocessing` (`dbof` package) — the new pipeline**

- Purpose: turn raw LLC4320 (13 faces, 4320², 51 levels, hourly) into ML-ready products *without interpolating to lat/lon*. Three product families: (a) global `(12960, 17280)` stitched maps of ~100 derived channels written as S3 zarr + NetCDF export; (b) fixed-physical-size cutouts sampled ∝ frontal activity (`log10 |∇b|²`, bias=1.3–2.0) with land/ice halo masks (scikit-fmm), downsampled to 64 px, zarr + parquet metadata; (c) single 720×720×51 native tiles.
- Surface calculations (the emphasis): `preprocessing/surface_subsets.py` dispatches six compute subsets — `native_fields` (U/V interp-to-tracer + CS/SN rotation to east/north), `surface_wind` (geographic τ, wind-stress curl, Ekman pumping/transport), `frontal_structure` (gradtheta2/gradsalt2/gradeta2/gradb2/gradrho2, Turner angle with shared gradients, density/buoyancy), `kinematic` (relative vorticity, strain n/s/mag, divergence, coriolis_f, Rossby number, Okubo-Weiss — all from one shared velocity Jacobian), `frontogenesis` (full/geostrophic/ageostrophic tendency, ug/vg from ∇η, single fused `dask.compute()` to avoid scheduler run_spec hazards), and `icearea` (raw passthrough).
- All gradients are done properly on the native C-grid via `utils/native_gradient.py` (xgcm `diff`/`interp` with `dxC`/`dyC` metrics, then CS/SN rotation; follows the ECCO v4 tutorial), with face connections wired in `llc4320_ingestion/grid.py` so seams are handled. Density/buoyancy use a vendored JMD95 EOS at p=0; project convention scales buoyancy ×1e3 (`b = 1e3·g_km·ρ/1025`), so `gradb2` is 1e6× physical — deliberately compensated for the newest fields (`R_ib`, `Wstar` use an unscaled `_buoyancy_gradient_phys_3d`).
- A parallel, much larger 3D world (`calculated_fields_at_depth.py`, 1566 lines; `depth_subsets.py`; `vertical_helpers.py`; `depth_strategies.py`) provides 12 depth subsets with per-channel depth strategies (`_sfc`, `_z25m`, `_mld`, `_mld_mean`): MLD (two definitions), N², Ri, Fr, Ro, Bu, R_ib, Ertel PV (vertical + tilt), buoyancy fluxes, ML heat content, KE, Wstar (Bachman 2021 modified Okubo-Weiss), plus 3D versions of all frontal/kinematic/frontogenesis fields.
- A critical correctness story: the vector-handling policy in `utils/faces_to_latlon.py` — pre-July-2026 stores mixed the staggered vector stitch with cell-centred data (V displaced 1 px in SURF/OSN; U/V effectively swapped in DEPTH native_fields). The fix is enforced structurally (`stitch_and_mask` strips `mate` attrs; processors reject staggered model channels) and proven by `tests/test_vector_rotation_equivalence.py`.
- Maturity: production-grade DEPTH pipeline, transfer layer (MIT→S3 with tile read-back verification), idempotent existence planning, per-run provenance metadata, 190 test functions (~7.2k lines) concentrated on the physics risk (Jacobian, tracer gradients, vertical helpers, vector rotation). Rough edges found: `Wstar` declared in `SURFACE_SUBSETS["frontogenesis"]` but unimplemented in `surface_subsets.py` (KeyError if requested for SURF/OSN); `zarr-to-netcdf --mode grid` calls nonexistent reader methods; `compute_energetics` crashes if only `KE_sfc/z25m` requested (mld=None); `defs.py` is a 4-field stub (the real dictionary is `plotting/field_cmaps.yaml`, ~180 channels, not yet updated for R_ib/Wstar); README stale. Per `docs/Data_Organization.md`: only one DEPTH date transferred (20121109T12), `depth_fields` V4 out of date, `surface_fields` global products have no dates generated yet.

**2. `wrangler` — JXP's earlier exploration (the engine under old DBOF)**

- All LLC machinery lives on the unmerged `llc_wrangling` branch (+4.8k lines over `main`, which is VIIRS/PODAAC-oriented). No LLC tests, no LLC docs.
- LLC access is entirely pre-staged local netCDF (`$OS_OGCM/LLC/data/ThetaUVWSaltEta/LLC4320_<iso>.nc`, surface slices of Theta/U/V/W/Salt/Eta) — files produced by `fronts/scripts/slurp_llc.py` from OSN kerchunk via `faces_dataset_to_latlon` (i.e. the face structure is destroyed at ingestion).
- `ogcm/llc.py::build_table` does healpix-uniform spatial sampling (astropy SkyCoord cross-match against a precomputed CC mask), date replication, and lat/lon/time-hash UIDs — this is exactly what `fronts/dbof/tables.py` wraps. `extract/ex_ogcm.py::llc_datetime` loads a whole timestep into RAM and slices cutouts by array index (fixed_km → pixel count from one meridional Δlat; no face-seam awareness).
- `preproc/pp_ogcm.py` is the old derived-field library and the direct ancestor of the new `calculate_additional_fields.py`: gradb2, Fs (frontogenesis), b, OW, vorticity, divergence, strain_rate, plus Cu (curvature number) and L (angular momentum) which have *no* counterpart in the new pipeline. All use `np.gradient` on the cutout index grid with a constant dx (2.25 km): no C-grid destaggering, no metric terms, no CS/SN rotation. Known unit inconsistencies (scalar-gradient family per-km vs current family per-m; velocity gradients in `calc_F_s` never divided by dx; `calc_curvatureradius` divides a length by dx).
- Notable latent bugs catalogued for the future comparison: lon-instead-of-lat cosine in `ogcm/utils.latlons_for_cutouts`; `resize is not None` truthiness in `preproc/field.py`; `dlat_km` unbound without `fixed_km`; success/pp_fields ordering assumption in `llc_datetime`; live `IPython.embed()` in error paths.

**3. `fronts/fronts` — front detection/characterization; two generations coexist**

- **Name collision**: `fronts/dbof/` (Generation 1, the *old* cutout-database DBOF built on wrangler) is unrelated to `import dbof` (Generation 2's upstream — this repo's new native-grid package). Both are called "DBOF".
- Generation 1 (`fronts/dbof` + `fronts/train` + `runs/dbof/dev`): JSON-driven cutout DB at `$DBOF_PATH/{name}/` — main parquet table (UID-keyed, one boolean column per field) + per-field HDF5 (groups by date, `gidx` index) + per-field meta parquet. `defs.py` catalogues 19 surface fields (SSTK, SSS/SSSs, SSH/SSHa/SSHs, Divb2, DivSST2, DivSSS2, b, Fs, OW, Cu, L, U, V, W, vorticity, divergence, strain_rate) all at dx=2.25 km, 64 px / 144 km, extracted via `wrangler.extract.ex_ogcm.llc_datetime`. `train/` builds balanced train/valid/test HDF5 sets (balance on log10(Divb2 p90)) for the SST/SSS/SSH→Divb2 U-Net ("Jake_test_set"). Working prototype; blocked on undeclared wrangler dependency; embed() calls left in.
- Generation 2 (current): `fronts` is a *consumer* of the new dbof package. `preproc/gradb2.py` calls `dbof.cli.generate_global.main(config, subset='frontal_structure', only_these_features=['gradb2'], run_id=...)`; `llc/io.py::zarr_to_nc` calls `dbof.cli.zarr_to_netcdf.main(...)` and reads `dbof.dataset_creation.config.load_config` — pinning the expected config shape (`run.run_id`, `output.{s3_endpoint,bucket,folder,dataset_name}`, `data.date_iterations`, YAML `subsets:`). Downstream everything assumes one global 2-D lat-lon array per (timestamp, channel, version) named `LLC4320_{ts}_{channel}_v{ver}.nc` under `$OS_OGCM/LLC/Fronts/V{ver}/YYYYMMDD_HHMMSS/`.
- The Gen-2 chain: gradb2 → `finding/` (pyBOA local-percentile threshold with 4 backends, gradb2-ridge-following priority-queue sharpening with a topology-preserving simple-point LUT, skan-based spur pruning; configs A–D,Z) → binary global .npy → `properties/` (labeling, per-front geometry: haversine length, orientation, curvature, branch points; EDT-dilated colocation stats against ~22 property channels; PCA and Balwada-style vorticity–strain JPDFs — the two most polished modules) → viewers (PyQt6 global viewer, Bokeh per-front hover). Current canonical driver: `runs/prototypes/one_full/build_v3.py` (V3, config D, adds turner_angle/oceTAUX/oceTAUY/SIarea).
- `dev/dbof_version_comparison/` already bridges the generations: v0 = Jake_test_set (channels SSS/SSSs/SSTK/SSH → Divb2), v1 = the new pipeline's cutout zarr on s3://dbof (`run_id=year_4x150_20260203_201716`, channels Eta/Salt/Theta/U/V/W/log_gradb); matching by nearest lat/lon since UIDs/sampling differ. `dev/mld/gen_all.py` is the only depth/tile-aware fronts code and delegates entirely to `dbof.tiles.tile_utils`.
- A Cornillon SST-front lineage (`finding/thin_cc.py` port + original FORTRAN in `finding/CC/`) sits dormant as a reference implementation; `preproc/bouali_destriping.py` is complete but aimed at future satellite work.

**Key cross-repo takeaways for the coming progress assessment**

1. Lineage is clear: wrangler's flat-grid, cutout-local `np.gradient` diagnostics (gradb2, Fs, OW, vorticity, strain, divergence) → reimplemented in the new dbof package as native-C-grid, face-seam-aware, globally-stitched, dask-lazy, unit-tested fields — plus a large 3D/depth world (MLD, N², Ri, R_ib, Ertel PV, Wstar, fluxes) with no precedent in the old code. Old-only fields: Cu, L, SSHa/SSHs/SSSs smoothing variants, DivSST2/DivSSS2.
2. The old stack (wrangler + fronts/dbof + fronts/train) is a working single-user prototype with catalogued flat-grid approximations and latent bugs; the new stack replaces its production role, and the fronts repo has already re-pointed its front-finding/properties pipeline at the new package's zarr output.
3. Naming/convention translations needed for any quantitative comparison: SSTK↔Theta, Divb2↔gradb2/log_gradb (with the ×1e6 buoyancy-scaling convention), Fs↔frontogenesis_tendency, OW↔okubo_weiss, strain_rate↔strain_mag; old units are inconsistent (per-km vs per-m) while new units are SI with documented conventions.
4. Remaining integration gaps: surface_fields global products not yet generated as data; Wstar unimplemented for the surface pipeline; fronts' `ocean_field_defs` registry drift (missing gradrho2/turner_angle/oceTAUX/oceTAUY/SIarea); both wrangler and dbof are undeclared dependencies of fronts.

### 2026-08-24 (Wrote the surface-fields progress report — Prompt 2)

Executed Prompt 2: answered "how much farther along is `llc4320-native-grid-preprocessing` compared to `wrangler` and `fronts` for surface fields?" using the codebase knowledge assembled under Prompt 1 (no re-reading needed; verified one production config, `configs/cutouts/run/full_cutouts_dataset_v2_1.yaml`, to cite real cutout parameters: 21 feature channels, 150 samples/snapshot, 150 km / 64 px, gradient bias 1.3). Wrote the report to `reports/surface_report.md`.

Report structure and conclusions:

- **Verdict**: generational replacement, not incremental improvement. Summary table scores the new project ahead on all four requested axes, categorically so on sampling and calculation quality.
- **Cutout generation**: old = single-field-per-pass, RAM-bound, index-sliced 64 px/144 km cutouts with one-sided km→pixel conversion and fragile UID ordering, per-field HDF5+parquet; new = all-channels-at-once lazy dask extraction from S3 zarr, exact two-axis physical extents from cumulative dxC/dyC, gradient-weighted sampling with land+ice fast-marching halo masks, single (N,C,H,W) zarr with provenance metadata and hole-tolerant reader. Old retains two unported conveniences: target-balanced train/valid/test splitting and satellite-emulation preprocessing (SSHs/SSSs smoothing, SSHa demeaning, inpainting, noise).
- **Property range**: new computes ~29 surface channels in 6 subsets — a strict superset of the old physics except Cu and L — adding wind_stress_curl, Ekman pumping/transport, gradrho2, turner_angle, strain components, rossby_number, coriolis_f, and the geo/ageo frontogenesis decomposition; plus the ~100-channel depth world with no old precedent. Noted the declared-but-unimplemented surface `Wstar`.
- **LLC4320 sampling**: old = 6 bimonthly snapshots, healpix-uniform, lat ≤ +57°, pre-staged flat netCDF (faces destroyed at ingestion); new = full 13-face native access (OSN kerchunk + verified S3 raw stores), any hour addressable via the iteration calendar, global coverage including the Arctic cap, frontal-activity-weighted sampling, timestamp/corruption validation. Capability is categorical; the limitation is data inventory (surface_fields globals not yet generated; one DEPTH date).
- **Calculation quality**: the widest gap. Old = np.gradient on the index grid at constant dx=2.25 km, no destaggering/rotation/metrics, documented unit inconsistencies (Fs dimensionally wrong by a factor of dx; per-km vs per-m families) and bugs (lon-for-lat cosine), zero tests. New = canonical native-grid operators (rotate, Jacobian, tracer gradient) with real metrics and face connections, JMD95 EOS, documented conventions (×1e3 buoyancy scaling, compensated for R_ib/Wstar), NaN-guarded singularities, ~7.2k lines of tests concentrated on the physics, and the found-fixed-enforced-documented July-2026 vector-orientation bug as evidence the test infrastructure catches subtle errors the old stack could not.
- **Fairness section**: four things the old stack still has — delivered training-data volume, balanced splitting, observation-emulation preprocessing, and Cu/L — all downstream-of-extraction concerns portable onto the new store.
- **Bottom line**: old stack = successful proof of concept; new stack = production instrument. Remaining distance is data generation and porting conveniences, not capability; the fronts repo has already re-pointed its pipeline at the new package.

### 2026-08-24 (Prompt 3: verified the "balance" sampling claim; added a quantitative error budget)

Executed Prompt 3 (Fable). Two tasks: (1) check the `balance` parameter in the `sst_ssh_unet` input files and correct the report's "uniform sampling" characterization of old DBOF; (2) add a quantitative estimate of the calculation-quality improvement.

**1. The balance check — the user's recollection is correct; old DBOF training data was NOT uniformly sampled.**

- All four `runs/prototypes/sst_ssh_unet/llc4320_sst144_sss40_tvfile{A,B,C,D}.json` carry `"balance": {"metric": "logDivb290", "nbins": 20}`. Traced the full chain: `llc4320_sst_sss_proto.py::gen_trainvalid` → the *historical* `fronts/train/tables.py::gen_tvt` (recovered from git, commit `1cd4133`; the current file only has the later `dbof_gen_tvt`), which bins log₁₀ of the table column `Divb290` into 20 bins, takes **all** members of sparse bins and equal random draws from full bins — i.e., it flattens the target histogram. The `Divb290` column is written by `preproc_super` as `field + meta_key`, where the meta keys come from the historical `fronts/utils/stats.py::meta_stats` (`'90'` = per-cutout 90th percentile); the current wrangler equivalent is `preproc/meta.py::stats` with key `p90`.
- The later Generation-1 path did the same: `DBOF_train_config_jake_test.json` uses `"sampling": {"type": "balance", "field": "Divb2", "metric": "p90", "log_metric": true, "nbins": 10}` via `dbof_gen_tvt` (so the report's earlier "10 bins" figure was the Jake_test_set value; the sst_ssh_unet prototypes used 20).
- Important nuance preserved in the report: cutout *placement* into the database WAS uniform (`llc4320_dbof_dev.json`: `"sampling": "uniform"`, healpix); the front-enrichment happened at train/valid/test assembly. Old = post-hoc histogram *flattening* of a uniform pool (tail bins capped by what uniform placement collected — the code takes "all" of a sparse bin and cannot go back for more); new = exponential tilt `∝ exp(bias·log10 gradb2)` at extraction, with the strength a recorded config parameter. Report modified in five places: revision note in header, exec-summary sampling row, §1 (new bullet + assessment), §3 (new bullet + assessment), bottom line.

**2. Quantitative error budget (new §4.3 of the report).** Method: analytic error propagation from the verified code, plus a synthetic numerical experiment (512² doubly-periodic flow, k⁻² velocity spectrum, spectral-derivative truth, C-grid sampling via spectral half-cell shifts; both pipelines' exact stencils emulated — script `quant_error_check{,2}.py` in the session scratchpad, run in the `dbof` conda env). Key verified code facts: old derived fields were computed *before* the 64-px resize, on lat-lon-sector cutouts of true spacing ≈2.31·cos(lat) km, with constant dx=2.25 km (`ex_ogcm.llc_datetime` + `pp_ogcm` wrappers); `calc_grad2` divides by dx in km (per-km² family); `calc_vorticity/div/lateral_strain_rate` divide by dx·10³ (per-m family); `calc_F_s` never divides its velocity gradients by dx.

Findings, as written into §4.3:

- **Metric bias**: every old gradient amplitude is multiplied by 2.31·cos(lat)/2.25 (squared for |∇·|² fields): +3%/+5% at the equator, −27%/−47% at 45°, −44%/−69% at 57°N, −57%/−81% at 65°S. Area-weighted over the sampled −70°..+57° domain: −14% (first derivatives), −23% (squared). Systematic and latitude-correlated — an ML model regressing old-Divb2 amplitude inherits a spurious cos²(lat) suppression.
- **Dimensional errors**: old Fs is not a physical quantity in any unit system (velocity gradients in per-index units → uniform factor ~2.3×10³ off), with a spurious cos³(lat) cross-latitude modulation (×6–8 suppression at 57–60° vs equator); per-km² vs per-m families differ by 10⁶ in squared-gradient units.
- **Stencils — honest wash**: for tracer gradients the old and new stencils are algebraically identical on a uniform grid (np.gradient centered difference ≡ xgcm diff+interp), so the whole Divb2-family gap is metrics+EOS+seams; shared truncation floor ~1% RMS for resolved fields. For velocity-derived fields the synthetic test gave old (staggered-as-collocated) 21% RMS vorticity / 61% divergence at a 4·dx cutoff, scale-concentrated (32%/86% at 4–6·dx vs 4%/16% at 20–64·dx), dominated by ~0.7·dx (≈1.6 km) feature misregistration; the new interp–diff–interp chain measured 36%/105% — slightly *more* diffusive at grid scale (two extra 2-pt interps, cos² damping each). Reported honestly: the new pipeline does not win on raw grid-scale stencil accuracy; it wins on correct metrics/orientation/seams everywhere and unbiased (smoothing-only) error character.
- **Geolocation**: the cos(lon)-for-cos(lat) bug makes old pixel longitudes wrong by cos(lat)/cos(lon), sign-inverted for |lon|>90°.
- **Headline number**: systematic error on the core frontal diagnostics reduced from 20–80% (latitude-correlated) plus factors of 10³–10⁶ (Fs, cross-family) to the few-percent unbiased truncation floor — conservatively a one-to-two order-of-magnitude reduction, and "not a physical quantity → correct" for Fs. Added to the exec-summary quality row and the bottom line.

Lesson recorded: my first synthetic experiment had two bugs worth remembering — a forgotten DC-zeroing (K[0,0]=1e-9 with K⁻²·⁵ makes the "divergent" test field a near-constant, producing absurd 10¹³% relative errors) and an unfair comparison of corner-located native C-grid vorticity against center-located truth; the corrected run emulates dbof's actual interp-to-centre chain from `calculate_jacobian`.

### 2026-08-24 (Prompt 4: verified the gradient mis-estimate claim against the actual ingestion code and data — the "interpolated to constant 2.25 km" premise is a misrecollection)

Executed Prompt 4 (Fable): checked whether the old stack really mis-estimated the gradients, against the hypothesis that LLC4320 was first interpolated to a constant 2.25 km grid (which would make dividing by 2.25 correct). **Verdict: the mis-estimate is real; no interpolation to a constant grid ever happened.** Evidence chain, all checked directly:

1. `fronts/scripts/slurp_llc.py` (the script that produced every pre-staged file) calls `xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon(ds, metric_vector_pairs=[])` — despite the name, this only *rearranges* the 13 faces into a rectangle (concatenate sector faces, rotate faces 7–12, drop the Arctic cap). No resampling of any kind.
2. Opened the actual pre-staged file `$OS_OGCM/LLC/data/ThetaUVWSaltEta/LLC4320_2011-09-13T00_00_00.nc`: shape (12960, 17280) = 3×4320 by 4×4320 native cells (the very rectangle the new pipeline stitches), bare integer i/j indices, **no lat/lon coordinates**, and U/V still on staggered `i_g`/`j_g` dims — impossible for an interpolated product.
3. Measured the grid in `$OS_OGCM/LLC/data/CC/LLC_coords.nc` (the coords lookup `ex_ogcm.llc_datetime` actually used): meridional spacing 2.16 km at the equator → 1.85 km at 30° → 1.38 km at 51° → 0.97 km at 65°S; zonal 2.32·cos(lat). Never constant, never 2.25.
4. The extraction code itself presumes varying spacing: `dr = round(144/dlat_km)` recomputes the native pixel count per cutout from the *local* spacing — pointless on a uniform grid (would always be 64).
5. Located where 2.25 comes from and where it is legitimate: 144/64, the *post-resize* spacing (`mk_json.py` writes `'dx': 144./64`); gradients on already-resized images (e.g. `gallery()`'s `calc_gradb`) are fine. But the DB's derived fields set `resize: True` in `fronts/dbof/defs.py` and every `pp_ogcm` wrapper computes the gradient on the native-resolution cutout FIRST and resizes the result — the output grid's spacing was applied to the input grid.

Report changes: (a) added §4.4 answering the question with the five-point evidence chain and the corrected causal story ("resize-then-differentiate would have justified 2.25; the pipeline differentiated-then-resized"); (b) upgraded the §4.3-A bias table from the Mercator idealization (2.31·cos lat) to spacing *measured* from `LLC_coords.nc` — the measured grid is 4–7% tighter at low-mid latitudes, so the biases are slightly larger than first reported: ≈0% at the equator (2.25 luckily sits between dy=2.16 and dx=2.32), −26% at 30°, −50% at 45°, −71% at 57°N, −81% at 65°S for squared-gradient fields; area-weighted over the sampled −70°..+57° domain: **−20% (first derivatives), −33% (squared)**. The bias is monotone poleward — largest exactly where LLC4320's fronts are strongest (Southern Ocean).

What I learned beyond the verdict: the equatorial near-cancellation means low-latitude-only comparisons would (misleadingly) validate the old fields; and LLC4320's meridional spacing at the equator (2.16 km) is measurably tighter than the zonal 1/48° (2.32 km) — the grid is mildly anisotropic there, so the report's per-axis table now lists dy and dx separately.

**Follow-up (same day): full line-by-line read of `wrangler/preproc/pp_ogcm.py` and `wrangler/preproc/field.py`** (at JXP's request, to be sure), plus the historical fronts twins they shadow (`fronts/preproc/process.py`, `fronts/llc/extract.py`, `fronts/po/fronts.py` at commit 94ef7fb — all three since deleted from the repo). This confirmed the §4.3/§4.4 claim where it matters and surfaced one genuinely new nuance, now written into §4.4 as a "Route audit":

- **Route 1 (biased)** — the DBOF database's derived fields: every `pp_ogcm` wrapper computes the gradient on the native-resolution cutout with constant dx=2.25 and resizes the *result* (verified per-line: `gradb2_cutout` 95→100, `gradfield2_cutout` 57→61, `Fs_cutout` 134→138, `current_cutout` 218–228→234). `fronts/dbof/fields.py:66` injects `cutout_size: 64` into the defs.py pdicts, so `resize=True` executes as intended.
- **Route 2 (biased)** — the `sst_ssh_unet` U-Net *target*: historical `po_fronts.gradb2_cutout` has the same compute-then-resize order (calc at 38, resize at 43). The Divb2 target carries the full latitude bias.
- **Route 3 (NOT biased)** — the generic preprocessor (`preproc_field`/`main`) resizes *early* (to 64 px ≡ 2.25 km/px) and applies its `div2` gradient *last*, so the `sst_ssh_unet` DivSST2 training *inputs* (pdict `"div2": 2.25`) were differentiated on the resized grid where 2.25 is correct.

New insight from the audit: within the `sst_ssh_unet` training pairs, the input (route 3, correct scale) and target (route 2, biased scale) use inconsistent gradient conventions, so the input→target amplitude mapping the network had to learn drifts with latitude by the §4.3-A factors (up to ~3× at the poleward edge). Also re-confirmed two latent wrangler bugs with exact locations: `field.py:156` `resize is not None` truthiness (resize unconditional; defanged by the injected cutout_size) and `field.py:217` `div2` branch referencing `po_utils` without an import (NameError if ever hit — the div2 route only worked via the fronts copy, which imports it). Full reads also re-verified the §4.3-B unit facts at exact lines: `calc_F_s` velocity gradients per-index (296–300) vs buoyancy gradients per-km (304–309); `calc_grad2` per-km both axes; kinematics /(dx·10³); OW /(dx·10³)²; `calc_curvatureradius` divides a length by dx·10³ (478).
