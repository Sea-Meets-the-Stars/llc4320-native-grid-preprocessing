"""Verify the LLC4320 raw-store copy: compare sample chunks between the old
and new S3 locations for given dates.

Usage:
    python dev/verify_raw_copy.py "2012-01-15 12:00:00" ["2012-03-01 00:00:00" ...]
"""

import sys

import numpy as np

import dbof.llc4320_ingestion.get_raw_data as get_raw_data

ENDPOINT = "https://s3-west.nrp-nautilus.io"
BUCKET = "dbof/"
OLD_FOLDER = "LLC4320_v1"
NEW_FOLDER = "LLC4320_RAW/DEPTH"
VARS = ["Theta", "Salt", "Eta", "U", "V", "W", "oceTAUX", "oceTAUY", "oceQnet"]


def _sample(ds, var, face=1, tile=720):
    """One tile of *face* at the shallowest level (handles staggered dims)."""
    da = ds[var].isel(face=face)
    sel = {d: 0 for d in da.dims if d.startswith("k")}
    sel |= {d: slice(0, tile) for d in da.dims if d[0] in "ij"}
    return da.isel(sel).values


def main(dates):
    failures = 0
    for date in dates:
        ds_old = get_raw_data.get_llc_timestep_data(
            ENDPOINT, BUCKET, OLD_FOLDER, date, vars_requested=VARS)
        ds_new = get_raw_data.get_llc_timestep_data(
            ENDPOINT, BUCKET, NEW_FOLDER, date, vars_requested=VARS)
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
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
