#!/usr/bin/env python3
"""
Verify timestep alignment between the OSN kerchunk store and the
MIT-transferred S3 timestep stores.

This script answers one question: when generate_global.py converts a
human-readable date to an OSN iteration number (via _date_to_iteration +
FIRST_WIND_RECORD_OFFSET), does the resulting kerchunk file actually
contain data for that physical time?

Strategy
--------
1. Open the OSN kerchunk store for a known iteration and inspect its
   time coordinate / metadata.
2. Open the MIT-transferred S3 store for the same date and inspect its
   time coordinate / metadata.
3. Cross-check that both refer to the same physical instant.

Usage
-----
    # Check alignment for a single date (default: '2012-11-09 12:00:00')
    python verify_timestep_alignment.py

    # Check a specific date
    python verify_timestep_alignment.py --date '2011-12-09 12:00:00'

    # Check all dates in the transfer config
    python verify_timestep_alignment.py --all-dates

    # Also probe OSN for the iteration WITHOUT the offset (diagnostic)
    python verify_timestep_alignment.py --probe-no-offset
"""

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import s3fs
import ujson
import xarray as xr

# ---------------------------------------------------------------------------
# Constants (must match generate_global.py and transfer_llc4320.py)
# ---------------------------------------------------------------------------
TS_PER_HOUR = 144
LLC4320_START_DATE = datetime(2011, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
LLC4320_TIMESTEP_SECS = 25
FIRST_WIND_RECORD_OFFSET = 10_368
DATE_FMT = "%Y-%m-%d %H:%M:%S"

OSN_ENDPOINT = "https://mghp.osn.xsede.org"
S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
S3_BUCKET = "dbof/"
S3_FOLDER = "LLC4320"


# ---------------------------------------------------------------------------
# Helpers (copied from the pipeline for standalone use)
# ---------------------------------------------------------------------------

def date_to_iteration(date_str: str) -> int:
    """Date → iteration count since LLC4320_START_DATE (no offset)."""
    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)
    delta = dt - LLC4320_START_DATE
    return round(delta.total_seconds() / LLC4320_TIMESTEP_SECS)


def iteration_to_date(iteration: int) -> datetime:
    """Absolute model iteration → datetime, assuming iter 0 = START_DATE - offset."""
    total_secs = iteration * LLC4320_TIMESTEP_SECS
    # Model iteration 0 is FIRST_WIND_RECORD_OFFSET steps before LLC4320_START_DATE
    model_epoch = LLC4320_START_DATE  # This is what _date_to_iteration uses as origin
    # But the OSN iteration includes the offset, so subtract it to get back to our epoch
    secs_since_start = (iteration - FIRST_WIND_RECORD_OFFSET) * LLC4320_TIMESTEP_SECS
    return LLC4320_START_DATE + __import__("datetime").timedelta(seconds=secs_since_start)


def dataset_name_from_date(date_str: str) -> str:
    """'2012-11-09 12:00:00' → '20121109T12.zarr'"""
    dt = datetime.strptime(date_str, DATE_FMT)
    return dt.strftime("%Y%m%dT%H") + ".zarr"


# ---------------------------------------------------------------------------
# OSN kerchunk access
# ---------------------------------------------------------------------------

def open_osn_kerchunk(iteration: int, face: int = 0):
    """
    Open a single kerchunk reference file from OSN for inspection.
    Returns the xarray Dataset (lazily loaded).
    """
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": OSN_ENDPOINT})

    fname = (
        f"cnh-bucket-1/llc_surf/kerchunk_files/"
        f"llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{iteration}.json"
    )

    print(f"  OSN file: {fname}")
    try:
        with fs.open(fname, mode="rb") as f:
            ref = ujson.load(f)
    except FileNotFoundError:
        print(f"  *** FILE NOT FOUND: {fname}")
        return None

    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "storage_options": {
                "fo": ref,
                "asynchronous": False,
                "remote_protocol": "s3",
                "remote_options": {
                    "client_kwargs": {"endpoint_url": OSN_ENDPOINT},
                    "anon": True,
                },
            },
            "consolidated": False,
        },
    )
    return ds


def inspect_osn_time(ds, label=""):
    """Print time-related coordinates and attributes from an OSN dataset."""
    prefix = f"  [{label}] " if label else "  "

    if "time" in ds.dims or "time" in ds.coords:
        time_vals = ds["time"].values
        print(f"{prefix}time coordinate: {time_vals}")
        if hasattr(ds["time"], "attrs"):
            print(f"{prefix}time attrs: {dict(ds['time'].attrs)}")
    else:
        print(f"{prefix}No 'time' dimension/coordinate found.")
        print(f"{prefix}Available dims: {list(ds.dims)}")
        print(f"{prefix}Available coords: {list(ds.coords)}")

    # Check for iter coordinate
    if "iter" in ds.coords or "iter" in ds.dims:
        print(f"{prefix}iter coordinate: {ds['iter'].values}")

    # Print global attributes that might contain time info
    time_attrs = {k: v for k, v in ds.attrs.items()
                  if any(t in k.lower() for t in ["time", "iter", "date", "start"])}
    if time_attrs:
        print(f"{prefix}Time-related global attrs: {time_attrs}")


# ---------------------------------------------------------------------------
# MIT S3 timestep store access
# ---------------------------------------------------------------------------

def open_mit_s3_store(date_str: str):
    """Open the MIT-transferred S3 timestep store for a given date."""
    ds_name = dataset_name_from_date(date_str)
    bucket = S3_BUCKET.strip("/")
    folder = S3_FOLDER.strip("/")
    s3_url = f"s3://{bucket}/{folder}/{ds_name}"

    print(f"  S3 store: {s3_url}")
    try:
        ds = xr.open_zarr(
            s3_url,
            storage_options={
                "client_kwargs": {"endpoint_url": S3_ENDPOINT},
            },
            consolidated=False,
        )
        return ds
    except Exception as e:
        print(f"  *** Could not open S3 store: {e}")
        return None


def inspect_mit_time(ds, date_str, label=""):
    """Print time-related info from a MIT-transferred store."""
    prefix = f"  [{label}] " if label else "  "

    # Check stored attributes
    time_attrs = {k: v for k, v in ds.attrs.items()
                  if any(t in k.lower() for t in ["time", "iter", "date", "start"])}
    if time_attrs:
        print(f"{prefix}Store attrs: {time_attrs}")

    # The transfer script stores these attributes:
    if "selected_iteration" in ds.attrs:
        print(f"{prefix}selected_iteration (stored by transfer): {ds.attrs['selected_iteration']}")
    if "selected_date_utc" in ds.attrs:
        print(f"{prefix}selected_date_utc (stored by transfer): {ds.attrs['selected_date_utc']}")

    # Check if time coordinate exists
    if "time" in ds.coords:
        print(f"{prefix}time coord values: {ds['time'].values}")
        print(f"{prefix}time attrs: {dict(ds['time'].attrs)}")


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def check_alignment(date_str: str, probe_no_offset: bool = False):
    """
    Full alignment check for one date.

    Computes the iteration numbers both ways and opens both stores to
    cross-check their time metadata.
    """
    print(f"\n{'='*70}")
    print(f"CHECKING DATE: {date_str}")
    print(f"{'='*70}")

    # --- Arithmetic ---
    iter_no_offset = date_to_iteration(date_str)
    iter_with_offset = iter_no_offset + FIRST_WIND_RECORD_OFFSET
    time_idx_mit = iter_no_offset // TS_PER_HOUR

    dt = datetime.strptime(date_str, DATE_FMT).replace(tzinfo=timezone.utc)

    print(f"\n  Input date:              {date_str}")
    print(f"  LLC4320_START_DATE:      {LLC4320_START_DATE}")
    print(f"  Delta from start:        {dt - LLC4320_START_DATE}")
    print(f"  Iteration (no offset):   {iter_no_offset}")
    print(f"  FIRST_WIND_RECORD_OFFSET:{FIRST_WIND_RECORD_OFFSET}")
    print(f"  Iteration (with offset): {iter_with_offset}  ← used for OSN kerchunk")
    print(f"  MIT time_idx:            {time_idx_mit}  (= {iter_no_offset} / {TS_PER_HOUR})")
    print(f"  MIT store name:          {dataset_name_from_date(date_str)}")

    # --- Reverse check: does the offset iteration map back to the same date? ---
    reverse_dt = iteration_to_date(iter_with_offset)
    print(f"\n  Reverse check: iteration {iter_with_offset} → {reverse_dt.strftime(DATE_FMT)}")
    if reverse_dt == dt:
        print(f"  ✓ Round-trip date matches!")
    else:
        print(f"  ✗ MISMATCH: expected {date_str}, got {reverse_dt.strftime(DATE_FMT)}")

    # --- OSN kerchunk inspection (WITH offset — what the pipeline uses) ---
    print(f"\n--- OSN kerchunk (iteration={iter_with_offset}, WITH offset) ---")
    ds_osn = open_osn_kerchunk(iter_with_offset, face=0)
    if ds_osn is not None:
        inspect_osn_time(ds_osn, label="with_offset")
        ds_osn.close()
    else:
        print("  Could not load OSN data for this iteration.")

    # --- Optionally probe WITHOUT offset (diagnostic) ---
    if probe_no_offset:
        print(f"\n--- OSN kerchunk (iteration={iter_no_offset}, NO offset) ---")
        ds_osn_raw = open_osn_kerchunk(iter_no_offset, face=0)
        if ds_osn_raw is not None:
            inspect_osn_time(ds_osn_raw, label="no_offset")
            ds_osn_raw.close()
        else:
            print("  File not found (expected if offset is correct).")

    # --- MIT S3 store inspection ---
    print(f"\n--- MIT S3 timestep store ({dataset_name_from_date(date_str)}) ---")
    ds_mit = open_mit_s3_store(date_str)
    if ds_mit is not None:
        inspect_mit_time(ds_mit, date_str, label="mit")
        ds_mit.close()
    else:
        print("  Could not load MIT S3 store for this date.")

    # --- Cross-check ---
    print(f"\n--- CROSS-CHECK ---")
    if ds_osn is not None and ds_mit is not None:
        # If both have time coordinates, compare them
        osn_has_time = "time" in (ds_osn.dims if ds_osn is not None else {})
        mit_has_time = "time" in (ds_mit.dims if ds_mit is not None else {})

        # The most robust check: compare actual field values at a single point
        print("  To do a field-value comparison, pick a point and compare")
        print("  Theta values from OSN vs MIT at the same (face, j, i).")
        print("  Example:")
        print(f"    OSN: ds_osn.Theta.isel(time=0, k=0, face=0, j=2160, i=2160).values")
        print(f"    MIT: ds_mit.Theta.isel(face=0, j=2160, i=2160).values")
        print("  If these match, the timesteps are aligned.")
    else:
        print("  Cannot cross-check (one or both stores failed to open).")

    print()


# ---------------------------------------------------------------------------
# Bonus: list available OSN iterations in a range
# ---------------------------------------------------------------------------

def probe_osn_iteration_range(center_iter: int, radius: int = 5, face: int = 0):
    """
    Check which iteration files exist around a given center iteration.
    Useful for understanding the OSN iteration numbering.
    """
    print(f"\n--- Probing OSN iterations around {center_iter} (±{radius}) ---")
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": OSN_ENDPOINT})

    for it in range(center_iter - radius, center_iter + radius + 1):
        fname = (
            f"cnh-bucket-1/llc_surf/kerchunk_files/"
            f"llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{it}.json"
        )
        exists = fs.exists(fname)
        marker = "✓" if exists else "✗"
        # Convert to date assuming offset
        secs = (it - FIRST_WIND_RECORD_OFFSET) * LLC4320_TIMESTEP_SECS
        approx_date = LLC4320_START_DATE + __import__("datetime").timedelta(seconds=secs)
        print(f"  {marker} iter={it}  →  ~{approx_date.strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------------------------------------------------------------------
# Bonus: check the very first iterations to pin down the epoch
# ---------------------------------------------------------------------------

def probe_osn_epoch(face: int = 0):
    """
    Try to find the smallest iteration that exists on OSN.
    This helps confirm what model iteration 0 / FIRST_WIND_RECORD_OFFSET means.
    """
    print("\n--- Probing OSN for earliest available iterations ---")
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": OSN_ENDPOINT})

    # Check iterations around 0 and around FIRST_WIND_RECORD_OFFSET
    test_iters = list(range(0, 300, 144)) + list(range(
        FIRST_WIND_RECORD_OFFSET - 2 * TS_PER_HOUR,
        FIRST_WIND_RECORD_OFFSET + 3 * TS_PER_HOUR,
        TS_PER_HOUR,
    ))
    test_iters = sorted(set(test_iters))

    for it in test_iters:
        fname = (
            f"cnh-bucket-1/llc_surf/kerchunk_files/"
            f"llc4320_Eta-U-V-W-Theta-Salt_f{face}_k0_iter_{it}.json"
        )
        exists = fs.exists(fname)
        if exists:
            marker = "✓ EXISTS"
        else:
            marker = "✗"

        # What date would this be with our offset interpretation?
        secs = (it - FIRST_WIND_RECORD_OFFSET) * LLC4320_TIMESTEP_SECS
        approx_date = LLC4320_START_DATE + __import__("datetime").timedelta(seconds=secs)
        print(f"  {marker}  iter={it:>8d}  →  {approx_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # Also try to open iteration 10368 and read its time metadata
    print(f"\n  Opening iter={FIRST_WIND_RECORD_OFFSET} to inspect time metadata...")
    ds = open_osn_kerchunk(FIRST_WIND_RECORD_OFFSET, face=face)
    if ds is not None:
        inspect_osn_time(ds, label="epoch_check")
        ds.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify timestep alignment between OSN and MIT data sources."
    )
    parser.add_argument(
        "--date",
        default="2012-11-09 12:00:00",
        help="Date to check (ISO format, default: '2012-11-09 12:00:00').",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Check all dates from the transfer config.",
    )
    parser.add_argument(
        "--probe-no-offset",
        action="store_true",
        help="Also probe OSN for the iteration WITHOUT the offset (diagnostic).",
    )
    parser.add_argument(
        "--probe-epoch",
        action="store_true",
        help="Probe OSN for the earliest available iterations to verify the epoch.",
    )
    parser.add_argument(
        "--probe-range",
        type=int,
        default=None,
        metavar="ITER",
        help="Probe OSN for iterations around ITER (±5).",
    )
    args = parser.parse_args()

    # Header
    print("=" * 70)
    print("LLC4320 TIMESTEP ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"LLC4320_START_DATE:        {LLC4320_START_DATE}")
    print(f"LLC4320_TIMESTEP_SECS:     {LLC4320_TIMESTEP_SECS}")
    print(f"TS_PER_HOUR:               {TS_PER_HOUR}")
    print(f"FIRST_WIND_RECORD_OFFSET:  {FIRST_WIND_RECORD_OFFSET}")
    print(f"  = {FIRST_WIND_RECORD_OFFSET * LLC4320_TIMESTEP_SECS / 3600:.1f} hours")
    print(f"  = {FIRST_WIND_RECORD_OFFSET * LLC4320_TIMESTEP_SECS / 86400:.1f} days")

    if args.probe_epoch:
        probe_osn_epoch()
        return

    if args.probe_range is not None:
        probe_osn_iteration_range(args.probe_range)
        return

    if args.all_dates:
        dates = [
            "2011-12-09 12:00:00",
            "2012-01-09 12:00:00",
            "2012-02-09 12:00:00",
            "2012-03-09 12:00:00",
            "2012-04-09 12:00:00",
            "2012-05-09 12:00:00",
            "2012-06-09 12:00:00",
            "2012-07-09 12:00:00",
            "2012-08-09 12:00:00",
            "2012-09-09 12:00:00",
            "2012-10-09 12:00:00",
            "2012-11-09 12:00:00",
        ]
    else:
        dates = [args.date]

    for date_str in dates:
        check_alignment(date_str, probe_no_offset=args.probe_no_offset)

    print("\nDone.")


if __name__ == "__main__":
    main()
