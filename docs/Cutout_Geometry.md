# Cutout Geometry: Fixed km, Fixed Pixels

Every cutout covers the same **physical size** (`target_km_res` km) but a variable number of
native cells — grid spacing changes with latitude — and is then downsampled to the same
**pixel size** (`down_sample_res` × `down_sample_res`).

## Fixed physical extent
From a center `(j, i)`, walk outward accumulating cell spacing (`dxC`/`dyC`) until half of
`target_km_res` is reached in each direction:
- `spatial_cutouts.extent_in_i` / `extent_in_j` — cumulative distance + `searchsorted`.
- `get_lat_lon_extents_of_cutout(index, dxC, dyC, grid_shape, km_size)` — returns the
  `i_start/i_end/j_start/j_end` bounds and the realized `real_km_w`/`real_km_h`, or `None`
  if the cutout runs off the grid edge (those centers are dropped).

Because spacing varies, the native cutout size (in cells) differs by location and direction;
this pre-downsample shape is recorded as `pre_interp_res`.

## Fixed pixel size (downsample)
`spatial_cutouts.downsample_image(img, channels, target_dim=down_sample_res)` resamples each
variable-size cutout to `(C, down_sample_res, down_sample_res)`:
- feature channels — **area** averaging.
- coordinate channels (`XC`/`YC`) — **nearest** (never averaged across the dateline / grid seams).

Upsampling is rejected: `down_sample_res` must be ≤ the native cutout size, so pick it small
enough that `target_km_res` always resolves to at least that many native cells.

## Recorded per cutout
`target_km_res`, `real_km_w` / `real_km_h` (actual km spanned), `pre_interp_res` (native size
before downsample), plus center `(j, i)` and lat/lon. See [Reading Cutout Data](Reading_Cutout_Data.md).

## Related
- [Generating Cutout Data](generating_cutout_data_v1_2.md)
- [Halo Masking](Halo_Masking.md)
