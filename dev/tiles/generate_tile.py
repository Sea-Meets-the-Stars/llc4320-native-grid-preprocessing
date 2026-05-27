"""
CLI entry point for generating a single LLC4320 tile of a chosen property.

This is a thin wrapper around ``tile_utils.run``: it parses CLI args (one
required new flag, ``--property``, that selects which property to extract)
and delegates the entire pipeline to :func:`tile_utils.run`.

CLI usage
---------

    python dev/tiles/generate_tile.py \\
        --i 8123 --j 6044 \\
        --timestamp '2012-11-09 12:00:00' \\
        --property density \\
        [--output ./density_tile123_20121109T12.nc] \\
        [--s3-config configs/global_depth.yaml]

``--property`` accepts any key registered in
:data:`tile_utils.TILE_PROPERTIES` (currently ``density``, ``temperature``,
``salinity``).  If ``--output`` is omitted, the NetCDF is written to
``./{prefix}_tile{tile_idx:03d}_{YYYYMMDDTHH}.nc`` where ``{prefix}`` is the
property's ``filename_prefix`` (``density``, ``theta``, ``salt``, ...).

Adding a new property does **not** require any change in this file -- just
add an entry to ``TILE_PROPERTIES`` in ``tile_utils.py``.
"""

# stdlib
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Make ``import tile_utils`` work whether this script is run directly or
# imported as a package module.  Same pattern used by the other dev/ scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tile_utils  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    """Build the CLI argument parser and parse ``argv``.

    Parameters
    ----------
    argv : list of str or None
        Argument list (default: ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes ``i``, ``j``, ``timestamp``,
        ``property``, ``output``, ``s3_config``.
    """
    p = argparse.ArgumentParser(
        description=(
            "Generate a 3D tile of one property (density / temperature / "
            "salinity / ...) for one LLC4320 snapshot."
        ),
    )
    p.add_argument(
        "--i", type=int, required=True,
        help=(
            "rect-grid i coord (0..17279); any pixel inside the desired tile"
        ),
    )
    p.add_argument(
        "--j", type=int, required=True,
        help=(
            "rect-grid j coord (0..12959); any pixel inside the desired tile"
        ),
    )
    p.add_argument(
        "--timestamp", type=str, required=True,
        help="timestamp 'YYYY-MM-DD HH:MM:SS'",
    )
    p.add_argument(
        "--property", dest="property", type=str, default="density",
        choices=sorted(tile_utils.TILE_PROPERTIES),
        help=(
            "Which property to extract.  Add new properties by registering "
            "them in tile_utils.TILE_PROPERTIES."
        ),
    )
    p.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output path; if omitted, writes "
            "./{prefix}_tile{tile_idx:03d}_{YYYYMMDDTHH}.nc"
        ),
    )
    p.add_argument(
        "--s3-config", type=Path, default=tile_utils.DEFAULT_CONFIG,
        help=(
            "YAML with an s3_source block "
            "(defaults to configs/global_depth.yaml)"
        ),
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point: configure logging, parse args, dispatch to ``tile_utils.run``.

    Parameters
    ----------
    argv : list of str or None
        Argument list (default: ``sys.argv[1:]``).

    Returns
    -------
    None
        Side effect: writes a NetCDF (and matching PNG) to disk.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = _parse_args(argv)
    tile_utils.run(
        i_rect=args.i,
        j_rect=args.j,
        timestamp=args.timestamp,
        property=args.property,
        output=args.output,
        config_path=args.s3_config,
    )


if __name__ == "__main__":
    main()

# conda run -n ocean14 python -m pytest tests/test_generate_tile.py --deselect tests/test_generate_tile.py::test_rect_ij_to_tile_against_grid_zarr -v 2>&1 | tail -45
# python dev/pot_density/generate_tile.py --i 9800 --j 9000 --timestamp '2012-11-09 12:00:00'