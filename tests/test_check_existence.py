"""
test_check_existence.py
-----------------------
Hermetic unit tests for the zarr / NetCDF existence planners in
``dbof.global_dataset_creation.check_existence``.

These reproduce, at the *decision* level, the manual S3 test that seeded one
FULL, one INCOMPLETE, and one MISSING zarr store and confirmed
``generate_global`` did SKIP / error / GENERATE respectively.

Why this can be data-free
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The store-state checks are **metadata-only**: they read just the store's own
``zarr.json`` (root group attrs -> ``channel_names``) and ``data/zarr.json``
(``shape[0]`` -> timestep count).  They never touch chunk data, and the
planners take ``fs`` as a parameter.  So every state is fabricated by writing a
couple of tiny JSON objects onto an in-memory fsspec filesystem -- no S3, no
LLC4320 data, no dask, no grid.

How generate_global dispatches on these states (the behaviour under test):

    plan_zarr -> ZARR_FULL        => generate_global SKIPs the date
    plan_zarr -> ZARR_INCOMPLETE  => generate_global raises "delete & rerun"
    plan_zarr -> ZARR_MISSING     => generate_global GENERATEs the date

We assert the classification (``plan_zarr`` / ``plan_subset_date`` output)
only; the heavy generation path is exercised by the end-to-end test below.

Two groups of tests live here:

* The hermetic unit tests (top of file) -- always run, no S3, no data.
* ``TestGenerateGlobalEndToEnd`` -- a real-S3 integration test that walks a
  **single** subset through all three states using real generation under a
  throwaway ``run_id``::

      MISSING     -> generate with ONE depth suffix          (GENERATE works)
      INCOMPLETE  -> the same store, judged against BOTH      (raise & rerun)
                     suffixes, is incomplete
      FULL        -> delete + regenerate with both suffixes,  (SKIP on re-run)
                     then re-run

  The INCOMPLETE store is therefore *genuinely* incomplete (a real one-suffix
  build), not fabricated.  It hits S3, needs credentials, and generates one
  subset twice from real data, so it is **skipped unless ``DBOF_E2E=1``** is
  set.  See the class docstring for the run command.

CLI usage
---------
unit tests
    pip install pytest        # 'test' extra isn't installed in fronts
    pytest tests/test_check_existence.py -v
end-to-end test
    DBOF_E2E=1 pytest tests/test_check_existence.py::TestGenerateGlobalEndToEnd -v -s
"""

import json
import os

import fsspec
import pytest

import dbof.global_dataset_creation.check_existence as ce


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CHANNELS = ["Theta_sfc", "Theta_25m", "N2_sfc", "N2_25m"]
DATE_PREFIX = "20121109_120000"
RUN_ID = "test_run"


@pytest.fixture
def mem_fs():
    """A fresh in-memory fsspec filesystem (cleared between tests).

    The memory backend keeps its objects in process-global dicts, so we clear
    them per test to keep each case isolated.
    """
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def seed_store(fs, store_path, channels=None, written=None):
    """Fabricate a zarr store's metadata on *fs*.

    - ``channels`` writes ``{store}/zarr.json`` with that ``channel_names``
      attribute.  Pass ``None`` to omit the root metadata entirely (a MISSING
      store), or write it with no ``channel_names`` to make an INCOMPLETE store
      that exists but has no channel attribute.
    - ``written=True`` adds the ``iteration`` completion marker to the root
      attributes (the writer sets it last, so its presence == fully written);
      ``written`` falsy omits it (a created-but-never-finished store).
    - When ``channels`` is given, also writes ``{store}/data/zarr.json`` with a
      ``(C, 3, 3)`` shape (3-D: one snapshot, no time axis).
    """
    key = ce._store_key(store_path)
    if channels is not None:
        attrs = {"channel_names": channels}
        if written:
            attrs["iteration"] = 0
        with fs.open(f"{key}/zarr.json", "w") as f:
            json.dump({"attributes": attrs}, f)
        with fs.open(f"{key}/data/zarr.json", "w") as f:
            json.dump({"shape": [len(channels), 3, 3]}, f)


def write_netcdfs(output_dir, channels, date_prefix=DATE_PREFIX, run_id=RUN_ID):
    """Create empty exported ``.nc`` files for *channels* under *output_dir*."""
    for ch in channels:
        (output_dir / ce.netcdf_filename(date_prefix, ch, run_id)).write_text("")


# ---------------------------------------------------------------------------
# plan_zarr: the three states from the manual S3 test (+ edge cases)
# ---------------------------------------------------------------------------

def test_plan_zarr_full(mem_fs):
    """All expected channels + completion marker -> FULL (generate_global SKIPs)."""
    seed_store(mem_fs, "memory://b/full.zarr", channels=CHANNELS, written=True)
    assert ce.plan_zarr(mem_fs, "memory://b/full.zarr", CHANNELS) == ce.ZARR_FULL


def test_plan_zarr_incomplete_missing_channel(mem_fs):
    """Store built with only the 'sfc' suffix -> INCOMPLETE (raise & rerun).

    Mirrors the manual 'incomplete' subset: present but missing the 25m
    channels the run expects.
    """
    seed_store(mem_fs, "memory://b/inc.zarr",
               channels=["Theta_sfc", "N2_sfc"], written=True)
    assert ce.plan_zarr(mem_fs, "memory://b/inc.zarr", CHANNELS) == ce.ZARR_INCOMPLETE


def test_plan_zarr_incomplete_not_written(mem_fs):
    """All channels present but no 'iteration' marker (crashed run) -> INCOMPLETE."""
    seed_store(mem_fs, "memory://b/empty.zarr", channels=CHANNELS, written=False)
    assert ce.plan_zarr(mem_fs, "memory://b/empty.zarr", CHANNELS) == ce.ZARR_INCOMPLETE


def test_plan_zarr_incomplete_no_channel_names(mem_fs):
    """Root metadata exists but carries no channel_names attr -> INCOMPLETE."""
    key = ce._store_key("memory://b/nochan.zarr")
    with mem_fs.open(f"{key}/zarr.json", "w") as f:
        json.dump({"attributes": {}}, f)
    assert ce.plan_zarr(mem_fs, "memory://b/nochan.zarr", CHANNELS) == ce.ZARR_INCOMPLETE


def test_plan_zarr_missing(mem_fs):
    """No store on the filesystem at all -> MISSING (generate_global GENERATEs)."""
    assert ce.plan_zarr(mem_fs, "memory://b/missing.zarr", CHANNELS) == ce.ZARR_MISSING


# ---------------------------------------------------------------------------
# plan_subset_date: the run_all_subsets ".nc-first" combined planner
# ---------------------------------------------------------------------------

def _plan(mem_fs, store_path, output_dir):
    return ce.plan_subset_date(
        mem_fs, store_path, str(output_dir), DATE_PREFIX, RUN_ID, CHANNELS)


def test_plan_subset_date_skip_when_all_netcdfs_exist(mem_fs, tmp_path):
    """Every channel's .nc on disk -> SKIP (zarr never consulted)."""
    write_netcdfs(tmp_path, CHANNELS)
    # Intentionally do NOT seed the store: SKIP must not depend on it.
    action, export = _plan(mem_fs, "memory://b/strat.zarr", tmp_path)
    assert action == ce.SKIP
    assert export == []


def test_plan_subset_date_export_all_when_partial_nc_and_full_store(mem_fs, tmp_path):
    """Some .nc missing + FULL store -> EXPORT *all* channels (overwrite)."""
    write_netcdfs(tmp_path, ["Theta_sfc"])  # only one .nc present
    seed_store(mem_fs, "memory://b/strat.zarr", channels=CHANNELS, written=True)
    action, export = _plan(mem_fs, "memory://b/strat.zarr", tmp_path)
    assert action == ce.EXPORT
    assert set(export) == set(CHANNELS)  # all, not just the missing ones


def test_plan_subset_date_partial_suffix_exports_all(mem_fs, tmp_path):
    """sfc exported, 25m missing, FULL store -> EXPORT all (suffix scenario).

    The case raised in review: a subset fully exported at one depth suffix but
    not another.  Suffixes are individual channels, so the missing 25m files
    flag the date, and a full store re-exports every channel.
    """
    write_netcdfs(tmp_path, ["Theta_sfc", "N2_sfc"])  # both sfc, no 25m
    seed_store(mem_fs, "memory://b/strat.zarr", channels=CHANNELS, written=True)
    action, export = _plan(mem_fs, "memory://b/strat.zarr", tmp_path)
    assert action == ce.EXPORT
    assert set(export) == set(CHANNELS)


def test_plan_subset_date_generate_when_store_incomplete(mem_fs, tmp_path):
    """Some .nc missing + INCOMPLETE store -> GENERATE all channels.

    (generate_global then raises the 'incomplete; delete & rerun' error.)
    """
    write_netcdfs(tmp_path, ["Theta_sfc"])
    seed_store(mem_fs, "memory://b/strat.zarr",
               channels=["Theta_sfc"], written=True)  # missing 25m + N2
    action, export = _plan(mem_fs, "memory://b/strat.zarr", tmp_path)
    assert action == ce.GENERATE
    assert set(export) == set(CHANNELS)


def test_plan_subset_date_generate_when_store_missing(mem_fs, tmp_path):
    """Some .nc missing + MISSING store -> GENERATE all channels."""
    write_netcdfs(tmp_path, ["Theta_sfc"])
    # No store seeded.
    action, export = _plan(mem_fs, "memory://b/strat.zarr", tmp_path)
    assert action == ce.GENERATE
    assert set(export) == set(CHANNELS)


# ===========================================================================
# End-to-end: walk ONE subset MISSING -> INCOMPLETE -> FULL on REAL S3.
# ===========================================================================
#
# A single subset is generated for real and judged against two depth-suffix
# sets so each state arises naturally (no fabricated metadata).  Skipped
# unless DBOF_E2E=1.

# --- configuration (override via environment) ------------------------------
E2E_ENABLED = bool(os.environ.get("DBOF_E2E"))
E2E_CONFIG = os.environ.get("DBOF_E2E_CONFIG", "configs/global/run/run.yaml")
E2E_RUN_ID = os.environ.get("DBOF_E2E_RUN_ID", "vtest_e2e_check")
# The single subset walked through all three states (must be valid for the
# config's pipeline AND expand with depth suffixes; default is a DEPTH subset).
E2E_SUBSET = os.environ.get("DBOF_E2E_SUBSET", "stratification")


def _resolve_base(config_path):
    """Resolve pipeline / output / date_prefix / full suffixes from the YAML.

    Mirrors how generate_global + run_all_subsets resolve these so seeded
    paths and expected channels match the production code exactly.
    """
    import yaml
    from dbof.global_dataset_creation.config import default_output_folder
    from dbof.global_dataset_creation.iterations import date_to_run_id
    from dbof.io.filesystems import create_s3_filesystems

    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    pipeline = raw["pipeline"].upper()
    out = raw.get("output") or {}
    s3_endpoint = out.get("s3_endpoint", "https://s3-west.nrp-nautilus.io")
    bucket = out.get("bucket", "dbof/")
    folder = out.get("folder") or default_output_folder(pipeline)
    date_str = raw["data"]["date_iterations"][0]

    _, fs_sync = create_s3_filesystems(s3_endpoint)
    return {
        "raw": raw,
        "pipeline": pipeline,
        "full_suffixes": raw.get("depth_suffixes"),
        "bucket": bucket,
        "folder": folder,
        "date_prefix": date_to_run_id(date_str),
        "fs_sync": fs_sync,
    }


def _subset_info(env, subset, suffixes):
    """Return ``(dataset_name, channels)`` for *subset* at *suffixes*.

    Reuses run_all_subsets._build_work_list so the channel set (incl. suffix
    expansion) is identical to what generate_global will compute.
    """
    from dbof.cli.run_all_subsets import _build_work_list
    (_, dataset_name, channels), = _build_work_list(
        env["pipeline"], [subset], list(suffixes))
    return dataset_name, channels


def _store_path(env, dataset_name):
    from dbof.global_dataset_creation.zarr_dataset_global import make_run_prefix
    return make_run_prefix(
        env["bucket"], env["folder"], E2E_RUN_ID, dataset_name,
        date_prefix=env["date_prefix"])


def _write_temp_config(base_raw, depth_suffixes, dest_path):
    """Write a copy of *base_raw* with ``depth_suffixes`` overridden."""
    import copy
    import yaml
    raw = copy.deepcopy(base_raw)
    raw["depth_suffixes"] = list(depth_suffixes)
    with open(dest_path, "w") as fh:
        yaml.safe_dump(raw, fh)
    return str(dest_path)


def _remove_store(fs_sync, store_path):
    """Best-effort recursive delete of one store (scoped to the test run_id)."""
    key = ce._store_key(store_path)
    try:
        if fs_sync.exists(key):
            fs_sync.rm(key, recursive=True)
    except Exception:
        pass


def _store_max_mtime(fs_sync, store_path):
    """Newest LastModified across all objects in the store, or None if absent.

    Used to prove a SKIP left the store untouched: a regeneration would PUT
    new objects and advance this; a skip writes nothing so it stays equal.
    """
    key = ce._store_key(store_path)
    fs_sync.invalidate_cache(key)
    info = fs_sync.find(key, detail=True)
    times = []
    for v in info.values():
        t = v.get("LastModified") or v.get("last_modified") or v.get("mtime")
        if t is not None:
            times.append(t)
    return max(times) if times else None


@pytest.fixture
def e2e_progression(tmp_path):
    """Resolve the subset's channel sets + paths and write two temp configs.

    Produces a partial-suffix config (one suffix) and a full-suffix config
    (all suffixes from the base YAML), pre-cleans the store, and tears it down
    afterwards unless ``DBOF_E2E_KEEP=1``.  All writes/deletes are scoped to
    ``E2E_RUN_ID``.
    """
    env = _resolve_base(E2E_CONFIG)

    full_suffixes = env["full_suffixes"]
    if not full_suffixes or len(full_suffixes) < 2:
        pytest.skip("base config needs depth_suffixes with >=2 entries "
                    "for the MISSING->INCOMPLETE->FULL progression")
    partial_suffixes = full_suffixes[:1]

    dataset_name, partial_channels = _subset_info(env, E2E_SUBSET, partial_suffixes)
    _, full_channels = _subset_info(env, E2E_SUBSET, full_suffixes)
    if set(partial_channels) >= set(full_channels):
        pytest.skip(f"subset '{E2E_SUBSET}' channels do not expand between "
                    f"{partial_suffixes} and {full_suffixes}; set DBOF_E2E_SUBSET")

    store_path = _store_path(env, dataset_name)
    fs_sync = env["fs_sync"]
    _remove_store(fs_sync, store_path)  # clean slate

    ctx = {
        "pipeline": env["pipeline"],
        "fs_sync": fs_sync,
        "store_path": store_path,
        "partial_channels": partial_channels,
        "full_channels": full_channels,
        "cfg_partial": _write_temp_config(env["raw"], partial_suffixes,
                                          tmp_path / "run_partial.yaml"),
        "cfg_full": _write_temp_config(env["raw"], full_suffixes,
                                       tmp_path / "run_full.yaml"),
        "partial_suffixes": partial_suffixes,
        "full_suffixes": full_suffixes,
    }
    yield ctx

    if not os.environ.get("DBOF_E2E_KEEP"):
        _remove_store(fs_sync, store_path)


@pytest.mark.skipif(
    not E2E_ENABLED,
    reason=("real-S3 end-to-end generate_global test -- set DBOF_E2E=1 to run "
            "(needs S3 credentials + LLC4320 data; generates one subset, slow)"),
)
class TestGenerateGlobalEndToEnd:
    """Walk one subset through MISSING -> INCOMPLETE -> FULL on real S3.

    Run it (on a host with S3 + data access, e.g. the server)::

        conda activate fronts        # or any env with dbof installed
        pip install pytest           # the 'test' extra is not installed by default

        DBOF_E2E=1 \\
        pytest tests/test_check_existence.py::TestGenerateGlobalEndToEnd -v -s

    Useful overrides (all optional)::

        DBOF_E2E_CONFIG=configs/global/run/run.yaml   # which YAML to use
        DBOF_E2E_RUN_ID=vtest_e2e_check               # throwaway run_id to seed
        DBOF_E2E_SUBSET=stratification                # subset to walk (must
                                                      #   expand with suffixes)
        DBOF_E2E_KEEP=1                               # keep generated stores

    No .yaml editing is required: the partial- and full-suffix configs are
    derived from DBOF_E2E_CONFIG into a temp dir automatically.  The base
    config's ``depth_suffixes`` (>=2 entries) defines the full set; its first
    entry is the partial set.
    """

    def test_state_progression_missing_incomplete_full(self, e2e_progression):
        from dbof.cli.generate_global import main as generate_main

        c = e2e_progression
        fs_sync, store, pipeline = c["fs_sync"], c["store_path"], c["pipeline"]

        # -- Phase 1: MISSING -> GENERATE (one depth suffix) ----------------
        assert ce.plan_zarr(fs_sync, store, c["partial_channels"]) == ce.ZARR_MISSING
        generate_main(config_file=c["cfg_partial"], run_id=E2E_RUN_ID,
                      subset=E2E_SUBSET, pipeline=pipeline)
        assert ce.plan_zarr(fs_sync, store, c["partial_channels"]) == ce.ZARR_FULL, (
            "one-suffix generation did not produce a store complete for that "
            "suffix set")

        # -- Phase 2: same store, judged vs BOTH suffixes -> INCOMPLETE -----
        # generate_global must refuse to touch it and tell the user to delete.
        assert ce.plan_zarr(fs_sync, store, c["full_channels"]) == ce.ZARR_INCOMPLETE
        with pytest.raises(ValueError, match="incomplete"):
            generate_main(config_file=c["cfg_full"], run_id=E2E_RUN_ID,
                          subset=E2E_SUBSET, pipeline=pipeline)

        # -- Phase 3: delete (the documented remediation) + regenerate full -
        _remove_store(fs_sync, store)
        assert ce.plan_zarr(fs_sync, store, c["full_channels"]) == ce.ZARR_MISSING
        generate_main(config_file=c["cfg_full"], run_id=E2E_RUN_ID,
                      subset=E2E_SUBSET, pipeline=pipeline)
        assert ce.plan_zarr(fs_sync, store, c["full_channels"]) == ce.ZARR_FULL

        # -- Phase 4: FULL -> SKIP (re-run leaves the store untouched) ------
        before = _store_max_mtime(fs_sync, store)
        assert before is not None
        generate_main(config_file=c["cfg_full"], run_id=E2E_RUN_ID,
                      subset=E2E_SUBSET, pipeline=pipeline)
        after = _store_max_mtime(fs_sync, store)
        assert after == before, (
            "FULL store was modified on re-run -- generate_global did not SKIP")
