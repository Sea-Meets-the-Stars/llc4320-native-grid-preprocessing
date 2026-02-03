# List of major tasks
- We should support writing and reading to other distributed storage other than s3 
  - What storage should we support?
    - local file system
    - gcloud
- Dataset reader and metadata should be able to handle empty zarr indices
- Ordering of features needs to be set in stone somewhere
  - Maybe dont even allow user to pass in features, maybe just the calculated ones. 
- Weighted sampling should act on a computed array instead of a dask array
- extract_patch_extents_and_metadata_in_series should be parallizable now
# List of little fixes
- Meta has a redundant id column