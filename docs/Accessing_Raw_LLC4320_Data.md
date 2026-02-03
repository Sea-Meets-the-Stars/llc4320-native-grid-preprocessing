# Remote LLC4320 Access (Kerchunk + Dask) for https://mghp.osn.xsede.org

This module provides lazy, remote access to LLC4320 model output and grid files using kerchunk references, xarray, and Dask. Data are accessed directly from an S3-compatible endpoint without downloading raw binaries.

## Raw Data Source 
There are multiple sources for the llc4320 data. This codebase currently only supports the following source : https://mghp.osn.xsede.org. 
This data was kindly archived by Spencer Jones.

The majority of our code to download data from this source is taken from this repo: https://github.com/cspencerjones/OSN_LLC4320/blob/main/Open_llc4320_surface_velocities.ipynb 

This dataset is a subset of the total output data from LLC4320. 
It contains only surface depth and features : Theta, U, V, W, Salt, Eta. 
The data source also stores a single grid file. This file describes the nature of the llc4320 grid and only needs to 
be loaded once for all data. 
The features are stored at float 32 at hourly time snapshot intervals. The expert on this is Spencer Jones. 

## Design Overview
Kerchunk JSON files map LLC4320 binaries to Zarr-style references.

xarray.open_dataset(..., engine="zarr") opens these references lazily.

Dask defers all I/O and enables parallel reads across faces.

Individual LLC faces are merged using xr.combine_by_coords.

Custom close handlers ensure all underlying file references are released.

## Usage
When accessing the raw data, we only load a single time snapshot (or iteration) in at a time. The amount of iterations 
to load, is decided in the config file by the user. This simply depends on what distribution of data the user wants 
processed. 
We will always load all 13 faces of all available data variables for a given iteration.

Important values :
- First valid wind/forcing record: ~1180
- Last iteration: 1495008
- Timesteps per hour: 144 (because the llc model has a cadence of 25 seconds, 
but we only have snapshots from every hour)

Load in grid file
```
co = get_raw_data.get_remote_gridfile(endpoint_url)
```
Loads and combines grid variables (XC, YC, metrics, CS/SN, etc.) for all 13 faces.

Load in an iteration of data
```
it = iter_range[i]
ds = get_raw_data.get_remote_llc_data(endpoint_url, it, face_range)
```
Loads all faces for one iteration. This should be within a loop.

Returns surface-level fields only (time=0, k=0, k_l=0)

Data remain lazy until .compute() is called

Example:
```
theta_mean = ds["Theta"].mean().compute()
```

Notes

Anonymous, read-only S3 access

One kerchunk JSON per face per iteration

Horizontal chunking applied (i, j)
