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
# Each entry is (tuple_of_channel_prefixes, list_of_raw_variables).
# A channel matches if it starts with any prefix in the tuple.

_CHANNEL_VARIABLE_RULES = [
    # Tracers: any buoyancy/density/gradient-based diagnostic
    (
        ('N2_', 'mixed_layer', 'ml_heat', 'Ri_', 'Fr_', 'Bu_',
         'ertel_pv', 'uB', 'vB', 'wB', 'Ro_', 'KE_',
         'gradb2_', 'gradtheta2_', 'gradsalt2_', 'gradrho2_',
         'turner_angle_', 'frontogenesis_', 'Wstar_'),
        ['Theta', 'Salt'],
    ),
    # Velocity: shear, dimensionless numbers, fluxes, kinematics
    (
        ('vertical_shear', 'Ri_', 'Fr_', 'Ro_', 'Bu_',
         'ertel_pv', 'uB', 'vB',
         'relative_vorticity_', 'strain_', 'divergence_',
         'okubo_weiss_', 'frontogenesis_', 'ug_', 'vg_', 'Wstar_'),
        ['U', 'V'],
    ),
    # Vertical velocity: PV and wB
    (
        ('ertel_pv', 'wB'),
        ['W'],
    ),
    # Wind stress
    (
        ('wind_stress_curl', 'ekman_pumping', 'u_ekman', 'v_ekman'),
        ['oceTAUX', 'oceTAUY'],
    ),
    # Sea-surface height
    (
        ('gradeta2_', 'frontogenesis_', 'ug_', 'vg_', 'Eta_'),
        ['Eta'],
    ),
    # Native fields at depth (raw model variables through depth strategies)
    (('Theta_',), ['Theta']),
    (('Salt_',),  ['Salt']),
    (('U_',),     ['U']),
    (('V_',),     ['V']),
    (('W_',),     ['W']),
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
