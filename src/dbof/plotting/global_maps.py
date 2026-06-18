"""
global_maps.py
--------------
Reusable plotting helpers for global LLC4320 fields in rectangular lat/lon
format (as produced by the generate-global pipeline and read back via
GlobalZarrDatasetReader).

These complement llc_plotting.py, which targets the 13 raw LLC *faces*.  The
helpers here take a single 2D rectangular field plus its (XC, YC) coordinates
and render it onto a Matplotlib axis — optionally a cartopy projection axis.

The colour-normalisation logic (log / diverging / linear) was previously
duplicated across notebook cells (single-channel map, all-channels grid,
depth-variant compare, regional plots).  ``make_field_norm`` is now the single
source of that behaviour.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean.cm as cmo
import cartopy.feature as cfeature

# Diverging colormaps that should be centred at zero.  Callers normally pass the
# set loaded from field_cmaps.yaml; this is a sensible fallback.
DEFAULT_DIVERGING_CMAPS = frozenset({"balance", "curl"})


def make_field_norm(arr, cmap_name, *, log=False,
                    diverging_cmaps=DEFAULT_DIVERGING_CMAPS,
                    percentiles=(1, 99)):
    """
    Build a Matplotlib norm for a 2D field, matching the convention used across
    the global-assessment notebooks.

    - ``log=True``           -> LogNorm over positive values.
    - diverging colormap     -> TwoSlopeNorm centred at zero.
    - otherwise              -> linear Normalize.

    Limits come from the given percentiles of the finite (and, for log, the
    positive) values.

    Parameters
    ----------
    arr : np.ndarray
        2D field, may contain NaNs.
    cmap_name : str
        cmocean colormap name (used to decide whether to centre at zero).
    log : bool
        Use a logarithmic scale.
    diverging_cmaps : set[str]
        Colormap names that should be centred at zero.
    percentiles : tuple[float, float]
        (low, high) percentiles for the colour limits.

    Returns
    -------
    matplotlib.colors.Normalize
    """
    finite = arr[np.isfinite(arr)]
    lo, hi = percentiles
    if finite.size == 0:
        return mcolors.Normalize()

    vmin, vmax = np.nanpercentile(finite, [lo, hi])
    if log:
        pos = finite[finite > 0]
        vmin, vmax = np.nanpercentile(pos, [lo, hi]) if pos.size > 0 else (1e-8, 1e-3)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if cmap_name in diverging_cmaps:
        vlim = max(abs(vmin), abs(vmax))
        return mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_global_field(ax, x, y, arr, field, cmap_cfg, *,
                      log_scale_channels=frozenset(),
                      diverging_cmaps=DEFAULT_DIVERGING_CMAPS,
                      percentiles=(1, 99),
                      transform=None,
                      add_coastline=True,
                      coastline_kw=None):
    """
    Draw one global field onto ``ax`` with the project's standard colour
    handling, and return the mappable plus its physical label.

    Parameters
    ----------
    ax : matplotlib axis
        Plain axis, or a cartopy GeoAxes when ``transform`` is given.
    x, y : np.ndarray
        Coordinate arrays (e.g. downsampled XC_g/YC_g, or a regional slice).
    arr : np.ndarray
        2D field to plot, aligned with ``x``/``y``.
    field : str
        Channel name, used to look up (cmap, label) in ``cmap_cfg``.
    cmap_cfg : dict[str, tuple[str, str]]
        Registry from dbof.plotting.field_cmaps.load_field_cmaps().
    log_scale_channels : set[str]
        Channels that should use a logarithmic colour scale.
    diverging_cmaps : set[str]
        Colormap names centred at zero.
    percentiles : tuple[float, float]
        Colour-limit percentiles passed to make_field_norm.
    transform : cartopy CRS or None
        Data CRS for cartopy axes (e.g. ccrs.PlateCarree()).  Omit for plain axes.
    add_coastline : bool
        Add a coastline feature (cartopy axes only).
    coastline_kw : dict or None
        Keyword args for the coastline feature; defaults to a thin black line.

    Returns
    -------
    (mappable, label) : (matplotlib.cm.ScalarMappable or None, str)
        ``mappable`` is None when the field is entirely NaN (nothing drawn), so
        callers can hide the axis.
    """
    cmap_name, label = cmap_cfg.get(field, ("viridis", field))

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, label

    cmap = getattr(cmo, cmap_name, plt.cm.viridis)
    norm = make_field_norm(arr, cmap_name,
                           log=field in log_scale_channels,
                           diverging_cmaps=diverging_cmaps,
                           percentiles=percentiles)

    kwargs = dict(cmap=cmap, norm=norm, shading="nearest")
    if transform is not None:
        kwargs["transform"] = transform
    im = ax.pcolormesh(x, y, arr, **kwargs)

    if add_coastline:
        ax.add_feature(cfeature.COASTLINE, **(coastline_kw or {"linewidth": 0.6, "color": "k"}))

    return im, label
