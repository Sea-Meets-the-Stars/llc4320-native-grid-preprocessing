"""
Determine which raw LLC4320 variables must be loaded from storage based
on the requested output channels.

The depth pipeline reads from S3 zarr stores and only fetches the variables
it actually needs.  This module maps computed-channel names to the raw model
variables required to produce them, avoiding unnecessary I/O.
"""


# ---------------------------------------------------------------------------
# Keyword → required-variable mappings
# ---------------------------------------------------------------------------
# Each entry is (tuple_of_channel_stems, list_of_raw_variables).
# A channel matches if it starts with any stem in the tuple.
# Covered by tests/test_variable_selection.py.

_CHANNEL_VARIABLE_RULES = [
    # Tracers: any buoyancy/density/gradient-based diagnostic
    (
        ('N2', 'mixed_layer', 'ml_heat', 'Ri', 'R_ib', 'Fr', 'Bu',
         'ertel_pv', 'uB', 'vB', 'wB', 'Ro', 'KE',
         'gradb2', 'gradtheta2', 'gradsalt2', 'gradrho2',
         'turner_angle', 'frontogenesis', 'density', 'buoyancy',
         'Wstar'),
        ['Theta', 'Salt'],
    ),
    # Velocity: shear, dimensionless numbers, fluxes, kinematics.  The raw
    # pair is always coupled: geographic rotation mixes U and V.
    (
        ('vertical_shear', 'Ri', 'Fr', 'Ro', 'Bu', 'rossby_number',
         'ertel_pv', 'uB', 'vB',
         'relative_vorticity', 'strain', 'divergence',
         'okubo_weiss', 'frontogenesis', 'ug', 'vg',
         'U', 'V', 'Wstar'),
        ['U', 'V'],
    ),
    # Vertical velocity: PV, wB, and the native W channels.  The native
    # stem is 'W_' (not bare 'W') so it matches W_sfc/W_z25m/... but does
    # NOT greedily swallow other capital-W channels such as 'Wstar_*'
    # (which needs no vertical velocity).
    (
        ('ertel_pv', 'wB', 'W_'),
        ['W'],
    ),
    # Wind stress: both raw components always needed (rotation couples them)
    (
        ('wind_stress_curl', 'ekman_pumping', 'u_ekman', 'v_ekman',
         'oceTAUX', 'oceTAUY'),
        ['oceTAUX', 'oceTAUY'],
    ),
    # Sea-surface height
    (
        ('gradeta2', 'frontogenesis', 'ug', 'vg', 'Eta'),
        ['Eta'],
    ),
    # Native tracer channels
    (('Theta',), ['Theta']),
    (('Salt',),  ['Salt']),
]

# MLD / mld_mean depth strategies always need Theta + Salt for density.
# Checked via endswith (not startswith) so it catches ANY channel regardless
# of base name (e.g. relative_vorticity_mld, Theta_mld_mean, etc.).
_MLD_SUFFIXES = ('_mld', '_mld_mean')


def required_model_variables(
    model_feature_channels: list[str],
    computed_feature_channels: list[str],
) -> list[str]:
    """
    Return the list of raw model variables needed from storage.

    Parameters
    ----------
    model_feature_channels : list[str]
        Raw model fields to include directly in the output (e.g. ``['Theta']``).
    computed_feature_channels : list[str]
        Names of derived fields to compute (e.g. ``['N2_sfc', 'Ri_z25m']``).

    Returns
    -------
    list[str]
        De-duplicated list of raw variable names.
    """
    needed = list(dict.fromkeys(model_feature_channels))   # preserve order, dedupe

    for prefixes, variables in _CHANNEL_VARIABLE_RULES:
        if any(ch.startswith(prefixes) for ch in computed_feature_channels):
            for v in variables:
                if v not in needed:
                    needed.append(v)

    # Any _mld or _mld_mean channel needs Theta + Salt for MLD (density).
    if any(ch.endswith(_MLD_SUFFIXES) for ch in computed_feature_channels):
        for v in ('Theta', 'Salt'):
            if v not in needed:
                needed.append(v)

    return needed
