"""
Integration test: run the SURF global pipeline off
``configs/global/run/cutout_test_surface_data.yaml`` and assert the produced
log is clean — specifically that the dask ``Detected different run_spec``
warning does NOT recur.

Background
----------
`compute_surface_wind` emits six channels (oceTAUX/oceTAUY + wind_stress_curl,
ekman_pumping, u_ekman, v_ekman) that share one lazy oceTAUX/oceTAUY stitch.
When ``stitch_and_mask`` materialised channels one-at-a-time (``.values`` per
channel), the shared keys were re-optimized differently on each ``update_graph``
call and the scheduler logged ``Detected different run_spec`` — which "can
cause failures and deadlocks".  The fix materialises all channels in a single
``dask.compute`` (see ``dbof.utils.faces_to_latlon.stitch_and_mask``).  This
test regenerates the two test dates × five surface subsets and fails if that
warning — or any ERROR-level line — reappears in the run log.

Running
-------
Heavy: builds a dask cluster and writes global zarr stores to the dbof S3
bucket.  Skipped unless ``--run-integration`` is passed, and needs OSN network
access + NRP S3 credentials (markers: integration, s3_dbof, osn):

    pytest --run-integration -m "s3_dbof and osn" \
        tests/test_generate_global_surface_integration.py

A FRESH run_id is generated each run so the pipeline's pre-flight existence
check never skips (a matching store would otherwise short-circuit generation).
Each run therefore writes a new S3 store; it is left in place (no deletes in
code) and the delete command is printed at the end.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from dbof.cli import generate_global

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "global" / "run" / "cutout_test_surface_data.yaml"


def _latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    assert logs, f"no log file was written under {log_dir}"
    return logs[-1]


@pytest.mark.integration
@pytest.mark.s3_dbof
@pytest.mark.osn
def test_surface_pipeline_log_has_no_run_spec_warning_or_errors():
    # Fresh run_id each run -> pre-flight sees ZARR_MISSING -> actually
    # regenerates (so the stitch/materialise path under test runs).
    run_id = f"run_spec_regression_{uuid.uuid4().hex[:8]}"
    log_dir = REPO / "logs" / run_id

    generate_global.main(config_file=str(CONFIG), run_id=run_id)

    log_text = _latest_log(log_dir).read_text()

    # 1. The regression this test exists for: the run_spec collision must be gone.
    assert "Detected different run_spec" not in log_text, (
        "dask 'Detected different run_spec' warning reappeared — the shared "
        "oceTAUX/oceTAUY stitch is being materialised per-channel again "
        "(see dbof.utils.faces_to_latlon.stitch_and_mask)."
    )

    # 2. No ERROR-level records anywhere in the run.
    error_lines = [ln for ln in log_text.splitlines() if "| ERROR |" in ln]
    assert not error_lines, "pipeline logged ERROR lines:\n" + "\n".join(error_lines)

    # 3. Output sanity: the run actually generated snapshots (did not no-op).
    assert "Surface snapshot assembly complete" in log_text, (
        "no snapshots were assembled — the run produced no output"
    )

    print(
        "\nLeft S3 test store in place. To remove it manually:\n"
        "  aws --endpoint-url https://s3-west.nrp-nautilus.io s3 rm --recursive "
        f"s3://dbof/test_data_for_cutouts/{run_id}/"
    )
