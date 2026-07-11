"""required_model_variables: every defined channel must map to its raw
variables, for both unsuffixed (surface-pipeline) and depth-suffixed names."""

import pytest

from dbof.global_dataset_creation.subset_definitions import (
    expand_channels_with_suffixes,
    get_subset_definition,
    valid_subsets,
)
from dbof.global_dataset_creation.variable_selection import (
    required_model_variables,
)

#: Channels legitimately derived from the grid alone (no raw model variable).
GRID_ONLY = {"coriolis_f"}

_ALL_SUBSETS = [("SURF", s) for s in valid_subsets("SURF")] + \
               [("DEPTH", s) for s in valid_subsets("DEPTH")]


def _expanded_channels(pipeline, subset):
    d = get_subset_definition(pipeline, subset)
    return d, expand_channels_with_suffixes(
        d["compute_features_channels"],
        depth_suffixes=d.get("depth_suffixes"),
        extra_channels=d.get("extra_channels"))


@pytest.mark.parametrize("pipeline,subset", _ALL_SUBSETS)
def test_every_computed_channel_maps_to_raw_vars(pipeline, subset):
    """Each computed channel, on its own, must pull at least one raw
    variable (except grid-only channels)."""
    _, channels = _expanded_channels(pipeline, subset)
    for ch in channels:
        if ch in GRID_ONLY:
            continue
        assert required_model_variables([], [ch]), \
            f"{pipeline}/{subset}: channel '{ch}' maps to no raw variables"


def test_depth_suffixes_match_like_bare_names():
    """A stem's suffixed forms must pull at least the bare name's variables
    (the _mld/_mld_mean suffixes may add Theta/Salt for the MLD)."""
    for base in ("N2", "gradb2", "relative_vorticity", "turner_angle",
                 "U", "V", "W", "Theta", "Salt", "ertel_pv_tilt"):
        bare = set(required_model_variables([], [base]))
        for suffix in ("sfc", "z25m", "mld", "mld_mean"):
            suffixed = set(required_model_variables([], [f"{base}_{suffix}"]))
            assert bare <= suffixed, \
                f"{base}_{suffix} lost variables relative to bare '{base}'"


def test_exact_mappings():
    """Spot-check exact variable sets (also guards against false-positive
    stem matches, e.g. 'Bu' must not match 'buoyancy')."""
    cases = {
        "density":           {"Theta", "Salt"},
        "buoyancy":          {"Theta", "Salt"},
        "N2_sfc":            {"Theta", "Salt"},
        "wind_stress_curl":  {"oceTAUX", "oceTAUY"},
        "u_ekman":           {"oceTAUX", "oceTAUY"},
        "oceTAUX":           {"oceTAUX", "oceTAUY"},
        "U":                 {"U", "V"},
        "U_sfc":             {"U", "V"},
        "V_mld":             {"U", "V", "Theta", "Salt"},   # _mld adds tracers
        "W_sfc":             {"W"},
        "Theta_sfc":         {"Theta"},
        "Salt_sfc":          {"Salt"},
        "Eta_sfc":           {"Eta"},
        "gradeta2":          {"Eta"},
        "gradrho2_sfc":      {"Theta", "Salt"},
        "rossby_number":     {"U", "V"},
        "vertical_shear_sfc": {"U", "V"},
        "ertel_pv_tilt_sfc": {"Theta", "Salt", "U", "V", "W"},
        "uB_sfc":            {"Theta", "Salt", "U", "V"},
        "frontogenesis_geo": {"Theta", "Salt", "U", "V", "Eta"},
        # Wstar needs tracers (for the buoyancy gradient) + velocities (for
        # the Jacobian) but NOT vertical velocity W — guards against the
        # bare-'W' stem greedily matching 'Wstar'.
        "Wstar_sfc":         {"Theta", "Salt", "U", "V"},
    }
    for ch, expected in cases.items():
        got = set(required_model_variables([], [ch]))
        assert got == expected, f"'{ch}': expected {expected}, got {got}"


def test_model_channels_pass_through():
    assert required_model_variables(["oceQnet", "SIarea"], []) == \
        ["oceQnet", "SIarea"]
