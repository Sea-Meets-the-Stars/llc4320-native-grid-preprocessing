"""Verify a raw-store copy: compare sample chunks between an old and new
S3 location for given dates.

Usage (2026 LLC4320_v1 -> LLC4320_RAW/DEPTH migration):
    python -m dbof.transfer.verify_copy \
        --endpoint https://s3-west.nrp-nautilus.io --bucket dbof/ \
        --old-folder LLC4320_v1 --new-folder LLC4320_RAW/DEPTH \
        "2012-01-15 12:00:00" ["2012-03-01 00:00:00" ...]
"""

import argparse

import numpy as np

import dbof.llc4320_ingestion.get_raw_data as get_raw_data

VARS = ["Theta", "Salt", "Eta", "U", "V", "W", "oceTAUX", "oceTAUY", "oceQnet"]


def _sample(ds, var, face=1, tile=720):
    """One tile of *face* at the shallowest level (handles staggered dims)."""
    da = ds[var].isel(face=face)
    sel = {d: 0 for d in da.dims if d.startswith("k")}
    sel |= {d: slice(0, tile) for d in da.dims if d[0] in "ij"}
    return da.isel(sel).values


def main(endpoint, bucket, old_folder, new_folder, dates):
    failures = 0
    for date in dates:
        ds_old = get_raw_data.get_llc_timestep_data(
            endpoint, bucket, old_folder, date, vars_requested=VARS)
        ds_new = get_raw_data.get_llc_timestep_data(
            endpoint, bucket, new_folder, date, vars_requested=VARS)
        for var in VARS:
            if var not in ds_old or var not in ds_new:
                print(f"{date}  {var:8s}  SKIP (missing: "
                      f"old={var in ds_old}, new={var in ds_new})")
                continue
            old, new = _sample(ds_old, var), _sample(ds_new, var)
            identical = np.array_equal(old, new, equal_nan=True)
            valid = np.unique(new).size > 2
            status = "OK" if identical and valid else "FAIL"
            failures += status == "FAIL"
            print(f"{date}  {var:8s}  {status}"
                  f"{'' if identical else '  (old != new)'}"
                  f"{'' if valid else '  (degenerate: all zeros/NaN)'}")
    print(f"\n{'ALL GOOD' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--old-folder", required=True)
    p.add_argument("--new-folder", required=True)
    p.add_argument("dates", nargs="+")
    a = p.parse_args()
    raise SystemExit(main(a.endpoint, a.bucket, a.old_folder, a.new_folder, a.dates))
