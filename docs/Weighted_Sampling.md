# Weighted Sampling by a Data Value

Draw grid points **without replacement**, with probability proportional to a field of
interest — used to bias cutout centers toward high-value regions (e.g. fronts). The field is
a 2-D stitched `(j, i)` array.

## Import
```
import dbof.preprocessing.weighted_coordinate_sampling as weighted_coordinate_sampling
```

## Design overview
Over the valid (finite, unmasked) cells, each cell's weight and probability are

    w = exp(bias * (value - min))
    p = w / w.sum()

Points are then drawn from `p` without replacement. Since the weight is exponential in the
raw value, feed a **log-scaled** field so `bias` behaves as a power-law knob — the cutout
pipeline passes `log10(gradb2)`. See [Sampling with ΔB²](Sampling_With_GradB2.md) for the
bias visualizations.

## Parameters
- `points_to_sample` (int) — number of points to draw.
- `bias` (float) — higher → more concentrated on high-value cells; `0` → uniform over valid cells.
- `field` (np.ndarray) — the `(j, i)` field; array-likes (xarray) are coerced via `np.asarray`.
- `mask` (bool array or None) — same shape; `False` cells are excluded.

## Returns
List of positional index tuples in field order, i.e. `(j, i)`.

## Usage
```python
idx = weighted_coordinate_sampling.weighted_sample_on_grid(
    points_to_sample=150, bias=1.3, field=log_gradb_np, mask=ocean_mask,
)
# idx -> [(j0, i0), (j1, i1), ...]
```

## Notes
- Non-finite (`NaN`) and masked cells are dropped before normalizing.
- `exp(bias * value)` overflows for large raw values — pass a log-scaled field.

## Related
- [Sampling with ΔB²](Sampling_With_GradB2.md)
- [Generating Cutout Data](generating_cutout_data_v1_2.md)
