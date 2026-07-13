# Testing


The package installs non-editable, so **reinstall after editing `src/`** (`pip install .`)
`pytest` is in the `test` extra: `pip install ".[test]"`.

## Layout
- **Unit tests** — offline, no network (config, sampling, cutout geometry, masks, dask
  pipeline mapping, metadata, access linking).
- **Integration tests** — hit real NRP S3; skip cleanly without credentials:
  `tests/test_global_input_integration.py`

## Credentials (integration only)
Export in the shell you run from (see [S3 / Nautilus](nautilus/s3_DBOF.md)):
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

## Commands
```bash
# refresh install after editing src/
pip install .

# run the suite (integration tests skip without S3 creds) Expect somewhere around 20 mins including intergrations 
pytest tests/ --ignore=tests/test_zarr_to_netcdf.py 
# TODO update the above with the test_zarr test is fixed

# end-to-end cutout pipeline: generate -> read -> render (heavy; needs creds)
pytest tests/test_global_input_integration.py::test_cutout_pipeline_end_to_end -s
```

## Notes
- "skipped" ≠ passed — integration tests skip without creds; run them with creds to verify.
- `tests/test_zarr_to_netcdf.py` has 2 known failures under zarr v3. TODO fix this.
- `-s` prints artifact paths (PNGs under `tests/output/`) and the S3 cleanup command.

## Related
- [Generating Cutout Data](generating_cutout_data_v1_2.md)
