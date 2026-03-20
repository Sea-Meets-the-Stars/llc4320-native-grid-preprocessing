"""
zarr_grid_global.py
-------------------
Zarr writer/reader for the LLC4320 static grid variables in rectangular
lat/lon format (12960 × 17280).

The grid is static (identical across all timesteps), so it is stored once in
its own Zarr group alongside — but separate from — the snapshot data written
by GlobalZarrDataset.

Store layout
------------
  <dataset_name>/       ← Zarr root group (e.g. "llc4320_grid.zarr")
    XC    : float32, (H, W)   — longitude of T-cell centre (degrees east)
    YC    : float32, (H, W)   — latitude  of T-cell centre (degrees north)
    lat   : float32, (H, W)   — alias for YC
    lon   : float32, (H, W)   — alias for XC
    Depth : float32, (H, W)   — ocean depth (m, positive downward)
    hFacC : float32, (H, W)   — fractional open thickness, surface level
                                 (0 = land, >0 = ocean)
    rA    : float32, (H, W)   — T-cell area (m²)
    SN    : float32, (H, W)   — sine of grid-rotation angle
    CS    : float32, (H, W)   — cosine of grid-rotation angle
    dxC   : float32, (H, W)   — grid spacing in x at U-face (m)
    dyG   : float32, (H, W)   — grid spacing in y at U-face (m)
    dyC   : float32, (H, W)   — grid spacing in y at V-face (m)
    dxG   : float32, (H, W)   — grid spacing in x at V-face (m)
    rAz   : float32, (H, W)   — vorticity-cell area (m²)

Root group attributes
---------------------
  grid_shape  : [H, W]       — spatial dimensions (typically [12960, 17280])
  variables   : list[str]    — names of all stored arrays
  description : str

Chunk strategy
--------------
  Each variable uses a single chunk covering the full (H, W) image.
  This is efficient for reading one full global field at a time.
"""

import numpy as np
import zarr
import xarray as xr
from pathlib import PurePosixPath


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_grid_store_path(bucket: str, folder: str, dataset_name: str) -> str:
    bucket = bucket.strip().strip("/")
    folder = folder.strip().strip("/")
    return f"s3://{str(PurePosixPath(bucket, folder, dataset_name))}"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class GlobalGridZarrWriter:
    """
    Write the LLC4320 static grid variables to an S3 Zarr store.

    Parameters
    ----------
    bucket : str
    folder : str
    dataset_name : str
        S3 path components. The full store path is
        ``s3://{bucket}/{folder}/{dataset_name}``.
        Default dataset_name: ``'llc4320_grid.zarr'``.
    fs : fsspec AbstractFileSystem
        Async S3 filesystem returned by ``create_s3_filesystems()``.
    """

    def __init__(
        self,
        bucket: str,
        folder: str,
        dataset_name: str,
        fs,
    ):
        path = _make_grid_store_path(bucket, folder, dataset_name)
        self.store = zarr.storage.FsspecStore(path=path, fs=fs)
        self.root = zarr.open_group(store=self.store, mode="w", use_consolidated=False)

    # ------------------------------------------------------------------

    def write(self, ds_rect: xr.Dataset) -> None:
        """
        Write all variables from a rectangular xarray Dataset to the store.

        Parameters
        ----------
        ds_rect : xr.Dataset
            Rectangular LLC4320 grid dataset as returned by
            ``_faces_dataset_to_latlon()``.  Expected variables include
            XC, YC, Depth, hFacC, rA, SN, CS, dxC, dyG, dyC, dxG, rAz.
            ``lat`` and ``lon`` aliases are added automatically if YC/XC
            are present.
        """
        # Add convenience aliases if not already present.
        if 'YC' in ds_rect and 'lat' not in ds_rect:
            ds_rect = ds_rect.assign(lat=ds_rect['YC'])
        if 'XC' in ds_rect and 'lon' not in ds_rect:
            ds_rect = ds_rect.assign(lon=ds_rect['XC'])

        # reset_coords() promotes coordinate variables (e.g. XC, YC, which
        # _faces_dataset_to_latlon sets as coords rather than data_vars) back
        # to data variables so they are captured by the data_vars loop below.
        # Side-effect: 1-D index coords (face, i, j, i_g, j_g) are also
        # promoted; we skip those by only writing exactly 2-D (H, W) arrays.
        ds_rect = ds_rect.reset_coords()

        stored = []
        H, W = None, None

        for vname in ds_rect.data_vars:
            arr = ds_rect[vname].values.astype(np.float32)

            # Skip 1-D and higher-than-2-D coordinate artefacts
            if arr.ndim != 2:
                continue

            if H is None:
                H, W = arr.shape
            elif arr.shape != (H, W):
                # Different stagger size — shouldn't happen for LLC4320 but skip
                continue

            self.root.create_array(
                vname,
                data=arr,
                chunks=(H, W),    # one chunk per variable — full global field
                overwrite=True,
                # dtype is inferred from arr (already cast to float32 above)
            )
            # Store per-variable attrs
            self.root[vname].attrs.update(dict(ds_rect[vname].attrs))
            stored.append(vname)

        # Root-level metadata
        self.root.attrs["grid_shape"]  = [int(H), int(W)]
        self.root.attrs["variables"]   = stored
        self.root.attrs["description"] = (
            "LLC4320 static grid variables converted from native face format "
            "to rectangular lat/lon via xmitgcm faces_dataset_to_latlon. "
            f"Shape: {H} × {W} = 3×4320 × 4×4320. No geographic interpolation."
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class GlobalGridZarrReader:
    """
    Read-only accessor for an LLC4320 grid Zarr store written by
    ``GlobalGridZarrWriter``.

    Usage
    -----
    ::

        reader = GlobalGridZarrReader(bucket, folder, dataset_name, fs)
        XC  = reader['XC']    # np.ndarray (H, W)
        YC  = reader.YC       # convenience property
        lat = reader.lat      # same as reader.YC
        lon = reader.lon      # same as reader.XC

    Parameters
    ----------
    bucket, folder, dataset_name : str
        S3 path components matching those used when writing.
    fs : fsspec AbstractFileSystem
        Async S3 filesystem (from ``create_s3_filesystems()``).
    """

    def __init__(
        self,
        bucket: str,
        folder: str,
        dataset_name: str,
        fs,
    ):
        path = _make_grid_store_path(bucket, folder, dataset_name)
        store = zarr.storage.FsspecStore(path=path, fs=fs)
        # use_consolidated=False: zarr v3 defaults to looking for consolidated
        # metadata and raises GroupNotFoundError when absent.
        self.root = zarr.open_group(store=store, mode="r", use_consolidated=False)

        self.grid_shape = tuple(self.root.attrs["grid_shape"])
        self.variables  = list(self.root.attrs["variables"])

    # ------------------------------------------------------------------
    # Core accessor

    def __getitem__(self, name: str) -> np.ndarray:
        """Return grid variable *name* as a numpy array (H, W)."""
        if name not in self.root:
            raise KeyError(
                f"Variable '{name}' not found in grid store. "
                f"Available: {self.variables}"
            )
        return self.root[name][:]

    def to_dataset(self) -> xr.Dataset:
        """
        Load all grid variables into a single xarray Dataset.

        Useful for downstream processing or quick inspection.
        Returns an in-memory xr.Dataset with dims (j, i).
        """
        data_vars = {}
        for vname in self.variables:
            arr = self.root[vname][:]
            attrs = dict(self.root[vname].attrs)
            data_vars[vname] = xr.Variable(('j', 'i'), arr, attrs)
        return xr.Dataset(data_vars)

    # ------------------------------------------------------------------
    # Convenience properties for the most-used fields

    def _get_lon(self) -> np.ndarray:
        """Return longitude array — checks XC then lon."""
        if 'XC' in self.root:
            return self['XC']
        if 'lon' in self.root:
            return self['lon']
        raise KeyError(
            "No longitude variable found in grid store. "
            f"Available: {self.variables}"
        )

    def _get_lat(self) -> np.ndarray:
        """Return latitude array — checks YC then lat."""
        if 'YC' in self.root:
            return self['YC']
        if 'lat' in self.root:
            return self['lat']
        raise KeyError(
            "No latitude variable found in grid store. "
            f"Available: {self.variables}"
        )

    @property
    def XC(self) -> np.ndarray:
        """Longitude of T-cell centre, shape (H, W), degrees east."""
        return self._get_lon()

    @property
    def YC(self) -> np.ndarray:
        """Latitude of T-cell centre, shape (H, W), degrees north."""
        return self._get_lat()

    @property
    def lat(self) -> np.ndarray:
        """Latitude (degrees north), shape (H, W)."""
        return self._get_lat()

    @property
    def lon(self) -> np.ndarray:
        """Longitude (degrees east), shape (H, W)."""
        return self._get_lon()

    @property
    def land_mask(self) -> np.ndarray:
        """
        Boolean mask: True where hFacC == 0 (land), shape (H, W).

        Returns None if hFacC was not stored (e.g. store written before the
        reset_coords() fix was applied — re-run generate-global-grid-zarr to
        populate the full variable set).
        """
        if 'hFacC' not in self.root:
            return None
        return self['hFacC'] == 0

    def __repr__(self) -> str:
        return (
            f"GlobalGridZarrReader(shape={self.grid_shape}, "
            f"variables={self.variables})"
        )
