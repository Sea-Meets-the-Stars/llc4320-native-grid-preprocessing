"""
Fixed data-source definitions for each pipeline variant.

These are infrastructure constants -- bucket paths, endpoints, and grid
locations for the three LLC4320 data stores.  Users select a pipeline
by name in the YAML config; the source is determined automatically.

Pipelines
---------
SURF
    Core ocean variables from OSN kerchunk; forcing variables (oceTAUX,
    oceTAUY, SIarea) from S3 timestep stores written by
    ``transfer_llc4320.py`` into the ``LLC4320`` folder.

OSN
    All variables from OSN kerchunk endpoints (surface + wind).  No S3
    timestep stores are used.

DEPTH
    All variables from S3 timestep stores in ``LLC4320_v1`` (full depth).
    Grid is read from the ``LLC4320`` folder (original, non-corrupt
    transfer location).
"""

# ---------------------------------------------------------------------------
# OSN kerchunk
# ---------------------------------------------------------------------------

#: Public OSN endpoint hosting kerchunk JSON references for LLC4320.
OSN_ENDPOINT = "https://mghp.osn.xsede.org"

#: Variables available from the ``llc_surf`` kerchunk store (surface ocean).
OSN_SURFACE_VARS = {"Theta", "Salt", "Eta", "U", "V", "W"}

#: Variables available from the ``llc_wind`` kerchunk store (surface forcing).
OSN_WIND_VARS = {"KPPhbl", "PhiBot", "oceTAUX", "oceTAUY", "SIarea"}

# ---------------------------------------------------------------------------
# LLC timestep stores (on S3)
# ---------------------------------------------------------------------------

#: LLC_SURF source for the surface-only pipeline (timestep stores).
LLC_SURF_SOURCE = {
    "s3_endpoint": "https://s3-west.nrp-nautilus.io",
    "bucket":      "dbof/",
    "folder":      "LLC4320",
}

#: LLC_DEPTH source for the depth-resolved pipeline (full-depth timestep stores).
#: ``grid_folder`` points to the original transfer location where
#: ``grid.zarr`` lives (the non-corrupt copy).
LLC_DEPTH_SOURCE = {
    "s3_endpoint":  "https://s3-west.nrp-nautilus.io",
    "bucket":       "dbof/",
    "folder":       "LLC4320_v1",
    "grid_folder":  "LLC4320",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_data_source(pipeline: str) -> dict | None:
    """
    Return the S3 data-source dict for *pipeline*, or ``None`` for OSN.

    Parameters
    ----------
    pipeline : str
        One of ``"SURF"``, ``"OSN"``, ``"DEPTH"``.

    Returns
    -------
    dict or None
        S3 source dict with keys ``s3_endpoint``, ``bucket``, ``folder``
        (and optionally ``grid_folder``).  ``None`` for OSN (pure kerchunk).
    """
    if pipeline == "SURF":
        return dict(LLC_SURF_SOURCE)
    elif pipeline == "DEPTH":
        return dict(LLC_DEPTH_SOURCE)
    elif pipeline == "OSN":
        return None
    raise ValueError(
        f"Unknown pipeline '{pipeline}'.  Expected one of: SURF, OSN, DEPTH."
    )
