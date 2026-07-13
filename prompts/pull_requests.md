# Pull Requests

This prompt doc will be used to review pull requests for the llc4320-native-grid-preprocessing repository.

## cutouts_data_v2

1. I have just merged in `main` and there are now conflicts in conftest.py.  Please resolve them.  Log your work in the Logs section.

2. Please perform a code review of the pull request on GitHub: `https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing/pull/19`.   Include inline code comments. Log your work in the Logs section.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-12 (Resolved conftest.py merge conflict on cutouts_data_v2)

Executed the 1st item under **cutouts_data_v2**: resolved the Git merge
conflict in `tests/conftest.py` after `main` was merged into the
`cutouts_data_v2` branch.

**The conflict.** A single conflicted region in the session-scoped
`zarr_reader` fixture (the S3 filesystem setup above it was unchanged and
outside the markers):

- `HEAD` (`cutouts_data_v2`): constructs the `ZarrDatasetReader` inside a
  `try/except` that calls `pytest.skip(...)` with the store URI when the
  cutout store is unreachable — so the reader-backed tests skip gracefully
  offline / when `s3://dbof/cutouts_dataset_v2_TESTING/...` is down.
- `origin/main`: the plain reader construction with no guard.

**Resolution.** Kept the `HEAD` version. It is strictly a superset — the
reader is built with the same `bucket`/`folder`/`run_id`/`dataset_name`/`fs`
arguments as `main`, just wrapped in the skip-guard — so nothing from
`main` is lost and the branch's graceful-skip behaviour is preserved.
Removed all conflict markers.

**Verification.**
- No conflict markers remain in `tests/conftest.py`.
- `python -c "ast.parse(...)"` → the file parses.
- `pytest tests/ --collect-only` collects all **336 tests** with no
  collection errors (the fixture imports cleanly).
- `git add tests/conftest.py` cleared the `UU` (both-modified) state → `M`;
  `git diff --diff-filter=U` reports no remaining unmerged files.
  Staged only — not committed (left for the user).

**What I learned.** The two sides made the *same* functional change (build
the reader) and differed only in error handling; the branch side added a
`pytest.skip` fallback so the cutout-store integration tests degrade to
"skipped" rather than "errored" when the NRP/Nautilus S3 endpoint is
unavailable. Choosing that side keeps the offline-friendly test posture the
rest of `conftest.py` already follows (the `--run-integration` gating).

### 2026-07-12 (Code review of PR #19 "Cutouts data v2" with inline comments)

Executed the 2nd item under **cutouts_data_v2**: a high-effort,
recall-biased code review of PR #19
(`Sea-Meets-the-Stars/llc4320-native-grid-preprocessing#19`, head
`466c749`, +3618/−1429 across 60 files) and posted the findings as inline
PR comments.

**Method.** Scoped the review to the Python diff (`src/` + `tests/`,
~3.5k lines; docs/notebooks/configs skimmed for context only). Fanned out
**7 independent finder angles** as subagents — line-by-line, removed-
behavior, cross-file tracer, reuse, simplification, efficiency, altitude —
each returning up to 6 candidates. Deduped, then **verified every surviving
candidate by reading the current source** (also to get exact line numbers,
since finders reported some diff-offset lines). Dropped weak/defensible
candidates (e.g. empty-list `feature_channels` erroring is arguably intended;
the dropped face-perimeter mask is documented as intentional for the
stitched grid — folded that into the related finding as a "confirm intent"
note).

**Posted 10 inline comments** (one COMMENT review, id 4679967246 —
not approve/request-changes):
- *Correctness (6):* (1) `stitched_halo_mask` uses a single global-mean
  grid spacing for fast-marching, vs the old per-face mean — km halos
  misplaced at high/low latitude; (2) snapshot time via
  `strptime(folder_name)` + `resolve_date_prefixes` accepting any
  subdirectory → run-aborting `ValueError` on a stray dir; (3)
  `date_prefixes[0]` → `IndexError` on empty discovery; (4) guardless
  `weighted_sample_on_grid` (crashes when valid cells < sample count, or
  `vals.min()` on an empty set); (5) `static_masks.py` calls the renamed
  `halo_mask.llc_halo_mask` → latent `AttributeError` in dead-but-shipped
  code (also flags the now-unapplied face-perimeter mask); (6)
  `access.meta.loc[keep_ids]` breaks image/metadata row-alignment if
  `image_id` is ever duplicated.
- *Efficiency (3):* per-snapshot re-`persist()` of static `dxC`/`dyC` in
  `generate_halo_ice_mask`; `gradb2` read twice per snapshot (their own
  TODO); `_open_subset_readers` (S3 `ls` + reader opens) re-run 3+ times
  per snapshot.
- *Cleanup (1):* a ~30-line dead commented-out block in `static_masks.py`
  referencing pre-rename names.

Review URL:
`https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing/pull/19#pullrequestreview-4679967246`
(verified: 10 inline comments attached).

**What I learned.** The PR is a solid refactor — most deleted behavior
(U/V tracer interpolation moved upstream, `log10(gradb2)` weighting, ice
polarity, grid-edge rejection) was correctly re-established, and the
cross-file tracer confirmed callers/imports resolve. The real theme of the
findings is the **native→stitched grid migration**: the halo fast-marching
lost its per-face spacing approximation, and a few native-grid helpers were
carried over with a stale callee name / no longer wired in. The other
cluster is **robustness of the new discovery path** (folder-name parsing
and empty/degenerate inputs that the old data-driven path never hit).