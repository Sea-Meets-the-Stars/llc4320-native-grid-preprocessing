"""
Regression tests for depth-suffix channel expansion.

Guards the invariant that surface-only bases (Eta and its derived
diagnostics) are never expanded across the full depth-suffix set -- doing
so requests channels the depth compute functions never produce, which
breaks both generation and the .nc existence check.
"""

from dbof.global_dataset_creation.subset_definitions import (
    DEPTH_SUBSETS,
    SURFACE_ONLY_BASES,
    expand_channels_with_suffixes,
)


SUFFIXES = ["sfc", "z25m", "mld", "mld_mean"]


def test_surface_only_base_gets_only_sfc():
    """A surface-only base expands to exactly one ``_sfc`` channel."""
    for base in SURFACE_ONLY_BASES:
        out = expand_channels_with_suffixes([base], depth_suffixes=SUFFIXES)
        assert out == [f"{base}_sfc"], out


def test_depth_base_gets_all_suffixes():
    """A normal base still expands across every depth suffix."""
    out = expand_channels_with_suffixes(["N2"], depth_suffixes=SUFFIXES)
    assert out == ["N2_sfc", "N2_z25m", "N2_mld", "N2_mld_mean"]


def test_mixed_list_preserves_order_and_surface_rule():
    out = expand_channels_with_suffixes(
        ["Theta", "Eta", "U"], depth_suffixes=SUFFIXES)
    assert out == [
        "Theta_sfc", "Theta_z25m", "Theta_mld", "Theta_mld_mean",
        "Eta_sfc",
        "U_sfc", "U_z25m", "U_mld", "U_mld_mean",
    ]


def test_no_depth_suffixes_returns_bases_unchanged():
    """Surface (SURF/OSN) pipeline path is unaffected by the surface rule."""
    out = expand_channels_with_suffixes(
        ["gradeta2", "gradb2"], depth_suffixes=None)
    assert out == ["gradeta2", "gradb2"]


def test_no_eta_channel_requested_at_depth_in_any_depth_subset():
    """No DEPTH subset expands an Eta-derived base beyond the surface."""
    for name, defn in DEPTH_SUBSETS.items():
        suffixes = defn.get("depth_suffixes")
        if not suffixes:
            continue
        expanded = expand_channels_with_suffixes(
            defn["compute_features_channels"],
            depth_suffixes=suffixes,
            extra_channels=defn.get("extra_channels"),
        )
        for base in SURFACE_ONLY_BASES:
            offenders = [
                ch for ch in expanded
                if ch.startswith(base + "_") and ch != f"{base}_sfc"
            ]
            assert not offenders, f"{name}: {offenders}"
