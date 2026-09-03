# Surface Fields: How Much Farther Along Is `llc4320-native-grid-preprocessing`?

**Date:** 2026-08-24
**Revised:** 2026-08-24 — (1) corrected the characterization of old-DBOF sampling: training sets were *not* purely uniform — they were rebalanced on log₁₀(Divb2 p90), verified in the `sst_ssh_unet` prototype configs and both generations of `fronts/train/tables.py`; (2) added a quantitative error budget for the calculation-quality comparison (§4.3).
**Question:** When it comes to surface fields, how much farther along is the `llc4320-native-grid-preprocessing` project (the new `dbof` package) compared to the `wrangler` and `fronts` projects (JXP's initial exploration)?
**Method:** Full read of all three codebases (2026-08-19, logged in `prompts/surface_check.md`): ~15.6k lines in `src/dbof`, ~3.3k lines in `wrangler` (branch `llc_wrangling`), ~16.2k lines in `fronts/fronts`. This report compares along the four requested axes: cutout generation for ML, range of properties, LLC4320 sampling, and calculation quality.

---

## Executive summary

The new project is not an incremental improvement — it is a **generational replacement**. On every axis examined, `llc4320-native-grid-preprocessing` is substantially ahead of the wrangler/fronts stack, and in the two axes that matter most for scientific credibility (quality of the calculations and sampling of the model), it is ahead in *kind*, not just degree:

| Axis | wrangler + fronts (old DBOF) | llc4320-native-grid-preprocessing | Verdict |
|---|---|---|---|
| **Cutout generation** | Working prototype: 64 px / 144 km cutouts, per-field HDF5 + parquet, single-machine, RAM-bound, ordering-fragile | Production pipeline: 150 km / 64 px cutouts, any channel combination in one store, lazy dask over S3, physically exact extents, gradient-weighted sampling, resumable, hole-tolerant reader | New is far ahead |
| **Range of properties** | 19 surface fields (some redundant variants), 2 unique fields (Cu, L) | ~29 surface channels across 6 subsets + a 3D/depth world (~100 channels) with no old counterpart | New is ahead; 2 legacy fields not yet ported |
| **LLC4320 sampling** | 6 bimonthly snapshots, uniform-healpix cutout placement (lat ≤ +57°, pre-staged local netCDF, no faces), training sets then rebalanced on log₁₀(Divb2 p90) | Full native access (13 faces, OSN kerchunk + S3 raw stores), hourly-capable calendar, global coverage incl. rotated faces, frontal-activity-weighted sampling at extraction | New is categorically ahead |
| **Quality of calculations** | Flat-grid `np.gradient` at constant dx=2.25 km, no C-grid destaggering, no vector rotation, unit inconsistencies, zero tests | Native C-grid xgcm operators with metric terms and CS/SN rotation, face-seam handling, JMD95 EOS, ~7.2k lines of tests targeting exactly the physics risk | New is categorically ahead: systematic errors of 20–80% (latitude-correlated) plus factors of 10³–10⁶ on two fields reduced to a few-% unbiased truncation floor (§4.3) |

The one honest caveat: the old stack has actually **produced and consumed** large training datasets end-to-end (the Nenya-era 44 GB preproc HDF5, the `Jake_test_set` U-Net data), while the new pipeline's global *surface* products were, at the time of the code read, still ahead of the data (`surface_fields` had no generated dates; one DEPTH date transferred). The new code is farther along; the new *data inventory* for surface fields is still catching up — though the existing `cutouts_dataset_v2` config and the `year_4x150_20260203_201716` run show cutout production is already real.

---

## 1. Ability to generate cutouts for machine learning

### Old stack (wrangler + `fronts/dbof` + `fronts/train`)

The Generation-1 pipeline works and was used in anger, but it is a research prototype throughout:

- **Extraction** (`wrangler/extract/ex_ogcm.py::llc_datetime`): loads an entire global timestep into RAM as numpy (`ds.Theta.values` etc.), slices cutouts by raw array index, and runs per-cutout derived-field functions in a `ProcessPoolExecutor`. One field per pass — building a 19-field dataset means 19 full passes over every timestep.
- **Physical size** is approximated: `fixed_km=144` is converted to a pixel count using a *single meridional* Δlat at the cutout corner (`dr = round(fixed_km / dlat_km)`), with the zonal extent assumed equal. Cutouts are then `resize_local_mean`'d to 64 px.
- **Storage contract** (`fronts/dbof`): one HDF5 per field with date-named groups plus a `_meta.parquet` carrying `(UID, group, gidx)`; a main parquet table with one boolean column per field. Assembling training tensors (`fronts/train/cutouts.py`) requires reading whole date groups and scattering by `gidx`.
- **Training sets were target-balanced, not uniform** (verified 2026-08-24 in the `runs/prototypes/sst_ssh_unet` configs): although cutout *placement* was uniform (healpix, `"sampling": "uniform"` in `llc4320_dbof_dev.json`), the train/valid/test assembly rebalanced on the per-cutout 90th percentile of Divb2. Two generations of the mechanism exist: (a) the `sst_ssh_unet` prototypes (tvfiles A–D) pass `"balance": {"metric": "logDivb290", "nbins": 20}` to the original `gen_tvt`, which bins log₁₀(Divb2 p90) into 20 bins, keeps *all* samples in sparse bins and draws equally from full ones — flattening the target histogram; (b) the later `dbof_gen_tvt` (Jake_test_set, `DBOF_train_config_jake_test.json`) does the same with `nbins: 10` against the per-field meta table. The `Divb290` column comes from `wrangler`'s per-cutout `meta.stats` (p90 of each extracted Divb2 cutout) stored into the super table at preprocessing time.
- **Fragility observed in code**: success/`pp_fields` ordering in `llc_datetime` is only correct when UID order happens to be identity; out-of-bounds checks are inconsistent with the smoothing halo; `dlat_km` is unbound if `smooth_km` is set without `fixed_km`; live `IPython.embed()` in error paths; no resume of partially-built fields; no tests for any of it.

### New stack (`cutout_dataset_creation` + `cutout_dataset_access`)

- **Extraction**: consumes the generate-global zarr stores lazily over S3 (dask), never holding the globe in RAM. One run extracts **all requested channels at once** — the production config `configs/cutouts/run/full_cutouts_dataset_v2_1.yaml` pulls 21 channels (Eta, Salt, Theta, U, V, W, gradb2, oceTAUX/Y, gradrho2, turner_angle, strain n/s/mag, divergence, relative_vorticity, coriolis_f, oceQnet, ekman_pumping, wind_stress_curl, SIarea) drawn transparently from whichever subset store holds each one.
- **Physical size is exact, both axes**: `spatial_cutouts.py` cumulative-sums the actual native `dxC`/`dyC` rows outward from the centre and `searchsorted`s the target half-width independently in i and j, recording the real km width/height and the pre-interpolation resolution in the metadata. Downsampling to 64 px uses area interpolation for fields and nearest for coordinates (so lon/lat are never averaged across seams).
- **Sampling is front-weighted at extraction**: cutout centres are drawn with probability `∝ exp(bias · log10|∇b|²)` (bias 1.3 in production), restricted to a mask that excludes land *and ice* with a fast-marching **halo buffer** of one cutout-width (scikit-fmm), so no cutout grazes a coastline or ice edge. This differs from the old stack's approach in *where* the front-enrichment happens (see §3): the old stack extracted uniformly and rebalanced afterward, so its front-rich tail was capped by whatever a uniform draw happened to collect; the new stack biases the draw itself, so the stored dataset is front-enriched at any size.
- **Storage contract**: a single `(N, C, H, W)` float32 zarr chunked one-image-per-chunk, uuid image ids, `channel_names`/`target_km_res`/`down_sample_res` as attrs, parquet metadata parts with 12 provenance columns (centre lat/lon, grid j/i, real km extents, `log_grad_b_2_center`, timestamp). The reader (`load_cutout_dataset`) reconciles zarr rows with metadata, tolerating holes from failed runs, and returns aligned lazy images + DataFrame.
- **Operational maturity**: thread-safe resumable appends (`zarr_global_index`), config-driven runs with a shared `run_id`/logging scheme, and an existing produced dataset (`year_4x150_20260203_201716`) that `fronts/dev/dbof_version_comparison/` already compares against the old `Jake_test_set`.

**Assessment:** the new pipeline supersedes the old one on correctness of cutout geometry, channel throughput, memory behavior, masking, provenance, and resumability. The only old capabilities not yet reproduced are (a) explicit histogram-flattening train/valid/test splitting on the target (old `fronts/train/tables.py`, both generations) — though the new gradient-biased extraction serves much of the same purpose upstream, exponentially tilting the stored distribution toward frontal activity rather than flattening a uniform pool after the fact — and (b) the per-field preprocessing menu (smoothing to satellite resolution — SSHs/SSSs — demeaning, inpainting, noise injection) that made the old cutouts directly comparable to satellite products. Those live downstream of extraction and could be layered on the new store, but today they exist only in the old code.

---

## 2. Range of properties that can be calculated

### Old stack: 19 surface fields (`fronts/dbof/defs.py` + `wrangler/preproc/pp_ogcm.py`)

Raw/preprocessed: SSTK, SSS, SSSs (40 km smoothed), SSH, SSHa (demeaned), SSHs (15 km smoothed, MEaSUREs-matched), U, V, W.
Derived: Divb2 (|∇b|²), b, Fs (frontogenesis tendency), OW, vorticity, divergence, strain_rate, DivSST2, DivSSS2, **Cu (curvature number), L (angular momentum)**.

### New stack: ~29 surface channels across 6 subsets (`surface_subsets.py` + `subset_definitions.py`)

- `native_fields`: Theta, Salt, Eta, W (raw) + U, V (properly rotated to true east/north).
- `surface_wind`: oceTAUX/oceTAUY (rotated), **wind_stress_curl, ekman_pumping, u_ekman, v_ekman** (+ oceQnet on the SURF pipeline).
- `frontal_structure`: gradb2, gradtheta2, gradsalt2, gradeta2, **gradrho2, turner_angle, density, buoyancy**.
- `kinematic`: relative_vorticity, **strain_n, strain_s**, strain_mag, divergence, **coriolis_f, rossby_number**, okubo_weiss.
- `frontogenesis`: frontogenesis_tendency, **frontogenesis_geo, frontogenesis_ageo, ug, vg** (geostrophic decomposition — new capability).
- `icearea`: SIarea.

Bold = no old-stack counterpart. Beyond the strict surface scope, the new package adds an entire depth-resolved family (~100 channels: MLD ×2 definitions, N², Ri, Fr, Ro, Bu, R_ib, Ertel PV terms, buoyancy fluxes, ML heat content, Wstar) with **zero** precedent in wrangler/fronts — relevant here because several "surface" diagnostics of frontal dynamics (R_ib, Wstar) are only meaningful with vertical information the old stack never had.

**Not yet ported from the old stack:** Cu (curvature number) and L (angular momentum); the smoothed/demeaned observation-matched variants (SSHs, SSHa, SSSs); DivSST2/DivSSS2 as standalone channels (gradtheta2/gradsalt2 are the physical equivalents, in SI units). Also note one declared-but-unimplemented new channel: `Wstar` appears in `SURFACE_SUBSETS["frontogenesis"]` but `compute_frontogenesis` (2D) has no branch for it — requesting it on the SURF/OSN pipeline raises `KeyError` (the DEPTH implementation is complete).

**Assessment:** the new pipeline computes a strict superset of the old *physics* (every old derived field except Cu and L has a corrected successor), adds ~10 genuinely new surface diagnostics, and organizes them with shared-intermediate reuse (one velocity Jacobian per subset, forwarded gradients for the Turner angle). The old stack retains only the observation-emulation preprocessing variants.

---

## 3. Sampling of the LLC4320

### Old stack

- **Temporal**: 6 snapshots at 2-month spacing from 2011-09-13 (`llc4320_dbof_dev.json`) — the model is hourly for ~14 months; the old pipeline touched ~0.06% of available timesteps, and only what had been pre-staged as `LLC4320_<iso>.nc` files by `fronts/scripts/slurp_llc.py`.
- **Spatial placement**: healpix-uniform 0.5° sampling cross-matched (via an astronomy library, `astropy.match_coordinates_sky` in a "galactic" frame) against a precomputed 3 GB clear-cutout mask; **latitude capped at +57°** because the interpolated lat-lon grid ends there — the Arctic face is simply absent. Cutouts that span face seams are undetectable because the faces were destroyed at ingestion.
- **But training-set selection was front-balanced, not uniform** (correction, 2026-08-24): the cutout *database* was populated uniformly, and then `gen_tvt`/`dbof_gen_tvt` selected train/valid/test members by flattening the histogram of log₁₀(Divb2 p90) — 20 bins in the `sst_ssh_unet` prototypes, 10 in the Jake_test_set. So the datasets the old U-Nets actually saw were strongly enriched in frontal activity relative to the ocean at large. The mechanisms still differ meaningfully from the new pipeline: old = post-hoc *flattening* of a uniform pool (the extreme-front bins can never hold more cutouts than uniform placement happened to capture — the code takes "all" of a sparse bin and cannot go back for more); new = *exponential tilt* (`∝ exp(1.3·log10 gradb2)`) applied at extraction, so front-rich samples are abundant in the store itself and the enrichment strength is a tunable, recorded parameter rather than a property of one training-set build.
- **Access**: whole-globe surface slices only (k=0 files); one netCDF per timestep on local disk; the sampling grid generation code (CC mask, coords file) lives outside all three repos.

### New stack

- **Temporal**: a first-class iteration calendar (`llc4320_ingestion/date_iterations.py`) with the 25 s timestep, MIT vs OSN iteration offsets, and validation against the model span — any hour of the ~9,000-hour run is addressable. Ingestion is dual-path: public OSN kerchunk references (surface + wind/ice variables) and credentialed S3 raw stores transferred from MIT with tile-by-tile read-back verification and a corrupt-date guard.
- **Spatial**: the full native 13-face domain, including the rotated faces 7–12 and the Arctic cap, stitched to the `(12960, 17280)` rectangle only at output time with documented vector-rotation policy. Cutout sampling covers the true global ocean, is **weighted toward frontal activity** (`log10 gradb2`, tunable bias) rather than uniform, and excludes land *and per-snapshot ice* with physically-sized halo buffers.
- **Verification**: timestamp integrity is cross-checked between OSN and S3 stores (`verify_osn_llc_surf_timestamp`); a validity guard rejects corrupt snapshots before writing.

**Assessment:** categorical advance — though the gap on *front-awareness* is narrower than a naive "uniform vs weighted" reading suggests, since the old stack did enrich its training sets on the target's frontal statistic (just later in the chain, and bounded by its uniform pool). The unqualified advances stand: the old sampling was constrained to a small pre-staged subset of the model, geographically truncated, and blind to face topology; the new sampling can address the entire model in space and time, biases the store itself toward fronts with a recorded tunable strength, and validates what it reads. The main practical limitation is inventory, not capability: per `docs/Data_Organization.md`, 23 SURFACE dates are transferred to S3 but global `surface_fields` products had not yet been generated, and only one DEPTH date exists.

---

## 4. Quality of the calculations

This is the widest gap.

### Old stack (`wrangler/preproc/pp_ogcm.py`)

Every derived field uses `np.gradient` on the cutout's index grid with a constant nominal spacing (dx = 2.25 km everywhere on a model whose native spacing varies from ~0.75 km at high latitude to ~2.3 km at the equator):

- **No C-grid awareness**: LLC U and V live on staggered west/south faces; the old code differences them as if collocated with Theta.
- **No vector rotation**: on the lat-lon-interpolated grid this mostly cancels, but U/V are treated as east/north without ever applying the CS/SN rotation — and because the extraction happened after `faces_dataset_to_latlon`, the fields inherit that tool's seam artifacts with no downstream handling (the `fronts` repo carries `inpaint_edges.py` specifically to patch `-999` and near-zero seam pixels after the fact).
- **Unit inconsistencies**, documented in the code read: the scalar-gradient family (Divb2, DivSST2, calc_F_s buoyancy terms) divides by dx in km while the kinematic family divides by dx·10³ in m; `calc_F_s` divides buoyancy gradients by dx but *not* the velocity gradients, so Fs is dimensionally inconsistent by a factor of dx; `calc_curvatureradius` divides a length by dx (dimensionally suspect); several docstrings are copy-paste stale.
- **Bugs**: longitude-instead-of-latitude in the meridian-convergence factor of `latlons_for_cutouts` (wrong lon spacing for every consumer away from the equator); `resize is not None` truthiness making resize unconditional; the `div2` branch referencing an unimported module.
- **Zero test coverage** of any LLC/derived-field code; the only tests in wrangler cover the VIIRS satellite path.

None of this necessarily invalidated the old exploration — for 64 px patches used as ML inputs, consistent-if-approximate fields can suffice — but the fields carry uncontrolled metric errors of order the dx variation (tens of percent at high latitude), and Fs in particular is not the quantity its name claims.

### New stack (`utils/native_gradient.py` + `preprocessing/`)

- **Correct native-grid operators**: every gradient goes through one of three canonical functions — `rotate_vector_to_geographic` (interp staggered → tracer, then CS/SN rotation), `calculate_jacobian` (the ECCO-v4-tutorial velocity gradient tensor with true `dxC`/`dyC` metrics and rotation of the gradient vectors), and `calculate_native_gradient_tracer`. Face connections are wired into xgcm so differencing works across seams.
- **Real equation of state**: JMD95 (vendored MITgcm implementation) at p=0 for density/buoyancy, replacing the old GSW-at-surface with linear-EOS shortcuts; the Turner angle's linear-EOS α/β usage is explicit and documented.
- **Convention discipline**: physical constants centralized in `physical_constants.py`; the ×1e3 buoyancy scaling convention is documented and deliberately compensated where it would bias new diagnostics (`R_ib`, `Wstar` use an unscaled buoyancy gradient); known divergences (Ro and ug/vg at the equator) are documented as physical rather than silently masked, while true divide-by-zeros (Turner angle, Ekman at f=0) are NaN-guarded.
- **The vector-orientation bug was found, fixed, enforced, and documented**: pre-July-2026 stores had V displaced 1 px (SURF/OSN) and U/V effectively swapped (DEPTH) from mixing the staggered stitch with cell-centred data. The fix is structural (`stitch_and_mask` strips `mate` attrs; processors reject staggered model channels) and proven by a dedicated 570-line equivalence test with committed comparison figures. This episode is itself evidence of maturity: the project has the test infrastructure to *catch* subtle physics errors the old stack could never have detected.
- **Test mass where the risk is**: ~7.2k lines / 190 test functions, concentrated on the Jacobian (830 lines), tracer gradients (620), vertical helpers (529), vector rotation equivalence (570), plus dask-graph hygiene (single fused `dask.compute()` per subset, documented run_spec hazard).

### Quantitative error budget (added 2026-08-24)

How much better, in numbers? The estimate below combines analytic error propagation, **measured grid spacing from the actual old-pipeline coordinate file** (`$OS_OGCM/LLC/data/CC/LLC_coords.nc`, verified 2026-08-24), and a synthetic numerical experiment (512² C-grid-sampled flow with a k⁻² velocity spectrum, spectral truth; both pipelines' exact stencils emulated). Setup facts, verified in code and data: the old pipeline extracted cutouts on the native LLC4320 rectangle — **never regridded** (see §4.4) — whose measured spacing runs from ~2.16 km (meridional) / 2.32 km (zonal) at the equator down to ~0.97 km at 65°S, computed every derived field *before* the 64-px resize with `np.gradient` (`defs.py` pdicts: `resize: True`; `gradb2_cutout` computes, then resizes), and divided by a **constant 2.25 km** regardless of latitude.

**A. Metric bias (all old gradient-based fields).** Every old gradient amplitude carries a multiplicative bias of (true local spacing)/2.25, squared for the `|∇·|²` family. Using spacing measured directly from `LLC_coords.nc` (mid-ocean column, both axes; the effective bias for an isotropic field averages the two axes):

| lat | measured dy / dx (km) | 1st-derivative fields (vorticity, divergence, strain) | squared-gradient fields (Divb2, DivSST2, OW) |
|---|---|---|---|
| 0° | 2.16 / 2.32 | −1% | −1% |
| 30° | 1.85 / 2.01 | −14% | −26% |
| 45° | 1.53 / 1.64 | −30% | −50% |
| 57° | 1.16 / 1.26 | −46% | −71% |
| 60°S | 1.14 / 1.16 | −49% | −74% |
| 65°S | 0.97 / 0.98 | −57% | −81% |

(The report's first revision used the Mercator estimate 2.31·cos(lat); the measured spacing is 4–7% tighter at low-mid latitudes, so the measured biases above are slightly *larger*. Near the equator 2.25 km happens to sit between the meridional and zonal spacings, so the bias is negligible there — and grows monotonically poleward.)

Area-weighted over the old sampled domain (−70° to +57°): **−20% mean bias for first derivatives, −33% for squared gradients**, rising past −70% in the Southern Ocean — exactly where LLC4320's frontal activity is strongest. Crucially this error is *systematic and latitude-correlated*, not noise: a model trained to predict old-Divb2 amplitude learns a spurious cos²(lat)-shaped suppression of tens of percent baked into its targets.

**B. Dimensional errors (order 10³–10⁶).** `calc_F_s` divides its buoyancy gradients by dx (in km) but never divides the velocity gradients by anything, so old Fs is not the frontogenesis tendency in *any* unit system — it is off by a uniform factor of order the grid spacing in metres (~2.3×10³) and, because three gradient powers each miss their local metric, its cross-latitude amplitude is spuriously modulated by cos³(lat): a **factor 6–8 suppression at 57–60°** relative to the equator. Separately, the scalar-gradient family (per-km²) and the kinematic family (per-m) differ by 10⁶ in squared-gradient units — harmless within one field, fatal for any cross-field physics (e.g., comparing Divb2 to OW).

**C. Stencil/discretization — where the honest comparison is a wash.** For **tracer gradients** the two pipelines' difference stencils are algebraically identical on a uniform grid (`np.gradient`'s centered difference ≡ xgcm diff-to-face + interp-to-centre), so for Divb2/DivSST2 the *entire* quality gap is items A–B plus the EOS and face seams; the residual shared truncation floor measured ~1% RMS for well-resolved fields, a few % for variance near the model's effective resolution. For **velocity-derived fields** the synthetic experiment shows the old staggered-as-collocated shortcut costs 21% RMS (vorticity) / 61% (divergence) against spectral truth at a 4·dx spectral cutoff — concentrated at small scales (32%/86% in the 4–6·dx band vs 4%/16% at 20–64·dx) and dominated by a ~0.7·dx (~1.6 km) misregistration of features. The new destagger-then-difference chain measured 36%/105% on the same test — slightly *more* diffusive near the grid scale (two extra 2-pt interpolations each damp by cos²). So the new pipeline does not win on raw stencil accuracy at grid scale; it wins because its error is unbiased smoothing with correct metrics and orientation everywhere on the globe, while the old error is position/phase corruption stacked on the systematic biases of A–B.

**D. Geolocation.** The old `latlons_for_cutouts` uses cos(longitude) where cos(latitude) belongs: pixel longitudes are wrong by a factor cos(lat)/cos(lon), sign-inverted wherever |lon| > 90° (half the ocean), unbounded near ±90°. This corrupts coordinate metadata rather than field values, but poisons any consumer that maps old cutout pixels to positions.

**E. Bottom line, quantified.** For every gradient-based surface field, the old stack's *systematic* error budget was: 20–80% latitude-correlated amplitude bias (A), plus factors of 10³–10⁶ on Fs and on cross-family comparisons (B), plus grid-scale misregistration (C), with zero tests guarding any of it. The new stack's systematic error budget for the same fields is the few-percent unbiased truncation floor of finite differences at native resolution, with correct SI units throughout and 190 tests over the operators. **Conservatively, that is a one-to-two order-of-magnitude reduction in systematic error for the core frontal diagnostics (Divb2, vorticity, strain, OW), and the difference between "not a physical quantity" and "correct" for the frontogenesis tendency.** The one quality dimension where the improvement is *not* large is near-grid-scale discretization RMS, which is comparable (and slightly smoother) by construction in both.

### Was LLC4320 first interpolated to a constant 2.25 km grid? No — verified 2026-08-24

A natural objection to §4.3-A: *if the model was first interpolated onto a constant 2.25 km grid and the gradient taken afterward, dividing by 2.25 km would be correct.* That would indeed make the metric bias vanish — but the interpolation step never existed. Five independent pieces of evidence, checked against the actual code and the actual pre-staged data on disk:

1. **The ingestion code doesn't regrid.** `fronts/scripts/slurp_llc.py` builds each snapshot from the OSN kerchunk faces and calls `xmitgcm.llcreader.llcmodel.faces_dataset_to_latlon(ds, metric_vector_pairs=[])`. Despite its name, that function only *rearranges* the 13 faces into one rectangle — it concatenates the lat-lon-sector faces, rotates faces 7–12 by 90°, and drops the Arctic cap. No interpolation, no target grid, no resampling of any kind. Every stored value is an untouched native cell.
2. **The stored files prove it.** `LLC4320_2011-09-13T00_00_00.nc` (opened directly) has shape (12960, 17280) — exactly 3×4320 by 4×4320 native cells, the same rectangle the *new* pipeline stitches — with bare integer `i`/`j` indices, **no lat/lon coordinates at all**, and U/V still on the staggered `i_g`/`j_g` dimensions. An interpolated 2.25 km product would carry coordinate axes and collocated vectors; this is raw model output rearranged.
3. **The coordinate file shows native, varying spacing.** `LLC_coords.nc` (the lat/lon lookup the old extraction used): measured meridional spacing is 2.16 km at the equator, 1.85 km at 30°, 1.38 km at 51°, 1.04 km at 62°, 0.84 km at 69°S. It is never 2.25 km-constant anywhere — it is the LLC4320 Mercator-like native grid.
4. **The extraction code itself assumes varying spacing.** `ex_ogcm.llc_datetime` computes `dr = round(144 / dlat_km)` with `dlat_km` read from `LLC_coords.nc` *at each cutout's location* — the number of native pixels needed to span 144 km is recomputed per cutout. On a uniform 2.25 km grid this would be pointless (always 64). The cutouts therefore range from ~64 native pixels at the equator to ~150 near 65°S, and are `resize_local_mean`'d down to 64 afterward.
5. **Where 2.25 km comes from, and where it *is* correct.** 2.25 = 144 km / 64 px — the spacing of the cutout *after* the resize (`mk_json.py` writes it as `'dx': 144./64`). Gradients taken on already-resized 64-px images with dx = 2.25 are correct — e.g. the `gallery()` sanity plot's `calc_gradb(..., dx=144/64)`. But the database's derived fields were not computed that way: the `defs.py` pdicts set `resize: True` with `dx: 2.25` (and `fronts/dbof/fields.py:66` injects `cutout_size: 64`), and every `pp_ogcm` wrapper computes the gradient on the **native-resolution cutout first** and resizes the *result*. The 2.25 that describes the output grid was applied to the input grid.

**Route audit** (full line-by-line read of `wrangler/preproc/pp_ogcm.py` and `wrangler/preproc/field.py`, plus their historical fronts twins, 2026-08-24). The old stack had **three distinct gradient routes**, and they do not all share the bias:

- **Route 1 — biased (the DBOF database's derived fields).** `wrangler/extract/ex_ogcm.llc_datetime` → the `pp_ogcm` wrappers. Verified in the full read: `gradb2_cutout` computes at line 95, resizes at line 100; `gradfield2_cutout` computes (57) then resizes (61); `Fs_cutout` (134→138); `current_cutout` — OW/strain/divergence/vorticity/Cu/L — (218–228→234). Gradient first on native spacing with constant dx = 2.25, resize after. This built Divb2, DivSST2, DivSSS2, Fs, OW, vorticity, divergence, strain_rate, Cu, L in the database. **§4.3-A applies in full.**
- **Route 2 — biased (the `sst_ssh_unet` U-Net *target*).** The historical `fronts/llc/extract.preproc_field` sent Divb2 through `po_fronts.gradb2_cutout`: same order (`calc_gradb` with `po_utils.calc_grad2(b, dx)` at its line 38, resize at 43). The Divb2 the U-Nets were trained to predict carries the full latitude bias.
- **Route 3 — *not* biased (gradient-of-preprocessed-field inputs).** The generic field preprocessor (`preproc_field`/`main` in the fronts copy; same structure in `wrangler/preproc/field.py`) runs resize (to 64 px ≡ 2.25 km/px) *early* and applies its `div2` gradient option *last* — so the `sst_ssh_unet` training *inputs* built with `"div2": 2.25` (DivSST2-as-input) were differentiated on the resized grid, where 2.25 km is correct to within the ~1–3% cutout-extent estimate.

Two consequences of the route audit worth stating. First, the bias claim survives full-code scrutiny exactly where it matters — every derived field stored in the old DBOF database, and the Divb2 regression target — while one product family (div2-route training inputs) escapes it. Second, and more insidious: within the `sst_ssh_unet` training pairs, the **input** DivSST2 (route 3, correct scale) and the **target** Divb2 (route 2, biased scale) use *inconsistent* gradient conventions, so the input-to-target amplitude relationship the network had to learn drifts with latitude by the §4.3-A factors — up to ~3× at the poleward edge. (Incidentally the full read also confirmed two latent bugs in the wrangler copies: `field.py:156`'s `resize is not None` truthiness makes resize unconditional — defanged in practice by the injected `cutout_size` — and `field.py:217`'s `div2` branch references `po_utils` without importing it, so the div2 route only ever worked through the fronts copy, which imports it correctly.)

So the answer to "why would this lead to such large errors?" is that the premise inverts the actual order of operations on the routes that produced the stored derived fields. The mis-estimate is exactly the ratio (local native spacing)/2.25: the constant happens to be nearly right at the equator (spacing 2.16–2.32 km) and drifts to a ~2× underestimate of gradient amplitudes by 60°S, doubled in log-space for the squared fields — the numbers in the §4.3-A table, now taken from the measured coordinates rather than a Mercator idealization. Had the pipeline resized *first* and differentiated *second* everywhere — as its own div2 route did — dx = 2.25 would have been legitimate (at the cost of computing gradients on area-averaged fields); the new pipeline instead differentiates on native cells with the true `dxC`/`dyC` metrics, which is strictly better than both.

**Assessment:** the old calculations were flat-grid approximations with real dimensional errors and no tests; the new calculations are metric-aware, seam-aware, EOS-correct, convention-documented, and heavily tested. For any quantitative use of gradient-based fields (which is the entire point of a fronts project), only the new pipeline's outputs are defensible.

---

## Where the old stack is still ahead

For fairness, four things the wrangler/fronts era has that the new project does not (yet):

1. **Delivered data volume**: the old pipeline produced and served multi-year, multi-field training sets that downstream projects (Nenya, the Divb2 U-Net) actually trained on. The new pipeline's surface global products were still unpopulated at the code read; cutouts exist for one production run.
2. **Target-balanced dataset splitting** (`fronts/train/tables.py`): histogram-balanced sampling on the target field's p90 — useful and unported.
3. **Observation-emulation preprocessing**: smoothing to satellite resolution (SSHs 15 km, SSSs 40 km), demeaning (SSHa), inpainting, noise injection — the machinery for making model cutouts look like satellite data.
4. **Two legacy diagnostics**: curvature number (Cu) and angular momentum (L).

All four are downstream-of-extraction concerns that could be built on the new store; none argues for keeping the old extraction path.

---

## Bottom line

Measured against JXP's initial exploration, the DBOF project has moved from *"approximate surface cutouts from a pre-staged, truncated, flat-grid copy of LLC4320, computed with unvalidated finite differences and assembled by a fragile single-user pipeline"* to *"a tested, documented, provenance-tracked system that computes a superset of the old surface physics correctly on the native grid, samples the full model where fronts actually are, and serves ML-ready cutouts and global maps from cloud storage."*

In numbers (§4.3): the move retired systematic, latitude-correlated amplitude errors of 20–80% on the core frontal diagnostics — and factors of 10³–10⁶ on Fs and cross-family comparisons — down to the few-percent truncation floor any finite-difference method carries, while the old stack's genuinely good idea (enriching training data toward frontal activity, via post-hoc log₁₀(Divb2 p90) histogram flattening) survives in stronger form as gradient-biased sampling at extraction.

On a rough maturity scale: the old stack was a successful proof of concept (it answered "can we train on LLC4320 cutouts?" — yes); the new stack is a production instrument. The remaining distance is mostly **data generation and porting of conveniences**, not capability: populate `surface_fields` for the target date list, port balanced splitting and the satellite-emulation preprocessing if still wanted, implement the surface `Wstar`, and decide whether Cu/L are worth carrying forward. The `fronts` repo has already voted with its imports — its current front-finding and properties pipeline consumes the new package exclusively.
