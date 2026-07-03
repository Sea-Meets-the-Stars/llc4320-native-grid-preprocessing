"""
Claude generated

field_cmaps.py
--------------
Loader for the per-channel colormap + physical-label registry used when
plotting LLC4320 global fields (SURF / OSN / DEPTH pipelines).

The registry itself lives in the sibling data file ``field_cmaps.yaml`` so it
can be shared as a single source of truth across notebooks and downstream
plotting code.
"""

import yaml
from pathlib import Path

# Registry data file shipped alongside this module.
CONFIG_PATH = Path(__file__).with_name("field_cmaps.yaml")


def load_field_cmaps(path=None):
    """
    Load the field colormap registry.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a YAML registry.  Defaults to the bundled
        ``field_cmaps.yaml``.

    Returns
    -------
    cmaps : dict[str, tuple[str, str]]
        Maps channel name -> (cmocean_colormap_name, physical_label).
    diverging_cmaps : set[str]
        Colormap names that should be centred at zero when plotting.
    """
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cmaps = {name: tuple(spec) for name, spec in cfg["cmaps"].items()}
    diverging_cmaps = set(cfg["diverging_cmaps"])
    return cmaps, diverging_cmaps
