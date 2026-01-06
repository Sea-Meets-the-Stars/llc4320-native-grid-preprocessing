# Remote LLC4320 Access (Kerchunk + Dask) for https://mghp.osn.xsede.org

This module provides lazy, remote access to LLC4320 model output and grid files using kerchunk references, xarray, and Dask. Data are accessed directly from an S3-compatible endpoint without downloading raw binaries.

## Raw Data Source 
There are multiple sources for the llc4320 data. This codebase currently only supports the following source : https://mghp.osn.xsede.org. 
This data was kindly archived by Spencer Jones.

The majority of our code to download data from this source is taken from this repo: https://github.com/cspencerjones/OSN_LLC4320/blob/main/Open_llc4320_surface_velocities.ipynb 

This dataset is a subset of the total output data from LLC4320. It contains only surface depth and features : Theta, U, V, W, Salt, Eta. The features are stored at float 32 at hourly intervals. The expert on this is Spencer Jones.

## Design Overview

Kerchunk JSON files map LLC4320 binaries to Zarr-style references.

xarray.open_dataset(..., engine="zarr") opens these references lazily.

Dask defers all I/O and enables parallel reads across faces.

Individual LLC faces are merged using xr.combine_by_coords.

Custom close handlers ensure all underlying file references are released.

## Import 
```
import data_ingestion.get_raw_data as get_raw_data
```

## Dask Setup and Usage

First, set up a dask client for effecient loading. 
```
from dask.distributed import Client
client = Client()  # uses all local cores by default
```

Set up a range of iteration you want to sample. 

LLC4320 output is indexed by iteration number.

Important values :
- First valid wind/forcing record: ~1180
- Last iteration: 1495008
- Timesteps per hour: 144 (because the llc model has a cadence of 25 seconds but we only have snapshots from every hour)


Example iteration range starting with the first valid wind record and looking at the next 12 hours
```
timestep_hours = 12                     # how many hours to load
sampling_step = 1                       # stride in timesteps
ts_per_hour = 144                       # model cadence this is constant

iter_step = sampling_step * ts_per_hour 

start_record = 1180

start_iter = 10368 + start_record * ts_per_hour
end_iter = start_iter + timestep_hours * ts_per_hour

iter_range = np.arange(start_iter, end_iter, iter_step)
```

Load in grid file
```
co = get_raw_data.get_remote_gridfile(endpoint_url)
```

Loads and combines grid variables (XC, YC, metrics, CS/SN, etc.) for all 13 faces.

Loads all requested faces for one iteration. This should be within a loop. 

```
it = iter_range[0]
ds = get_raw_data.get_remote_llc_data(endpoint_url, it, face_range)
```

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
