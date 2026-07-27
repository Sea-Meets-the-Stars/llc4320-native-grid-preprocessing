"""
Physical constants used by the LLC4320 preprocessing pipelines.

Centralised here so that every module uses the same values and any
update propagates automatically.
"""

# ---------------------------------------------------------------------------
# Gravitational acceleration
# ---------------------------------------------------------------------------

G = 9.81
"""float: Gravitational acceleration [m s⁻²]."""

G_KM = 0.0098
"""float: Gravitational acceleration [km s⁻²].

Used in buoyancy calculations where the length scale is kilometres
(e.g. ``buoyancy_of_field``, ``buoyancy_field_3d``).
"""

# ---------------------------------------------------------------------------
# Reference densities
# ---------------------------------------------------------------------------

RHO0_BOUSSINESQ = 1000.0
"""float: Boussinesq reference density [kg m⁻³].

Used in the Brunt–Väisälä frequency  N² = (g / ρ₀) dρ/dz  and in
heat-content integrals  Q = ρ₀ Cₚ ∫ θ dz.
"""

RHO0_SEAWATER = 1025.0
"""float: Realistic seawater reference density [kg m⁻³].

Used in buoyancy  b = −g ρ / ρ_ref,  Turner angle, and related
diagnostics where the absolute magnitude matters more than the
Boussinesq simplification.
"""

SIGMA0_REFERENCE_DENSITY = 1000.0
"""float: Reference density subtracted to form the σ₀ anomaly [kg m⁻³].

    σ₀ = ρ(S, Θ, p=0) − SIGMA0_REFERENCE_DENSITY

Used by :func:`calculated_fields_at_depth.potential_density_anomaly_3d`
(and the mixed-layer-depth criterion) so the ``− 1000`` offset is named in
exactly one place.
"""

# ---------------------------------------------------------------------------
# Seawater thermodynamic properties
# ---------------------------------------------------------------------------

CP = 3994.0
"""float: Specific heat capacity of seawater [J kg⁻¹ °C⁻¹]."""

ALPHA = 2.0e-4
"""float: Thermal expansion coefficient [°C⁻¹].

Linear EOS approximation used in the Turner angle calculation.
"""

BETA = 7.4e-4
"""float: Haline contraction coefficient [PSU⁻¹].

Linear EOS approximation used in the Turner angle calculation.
"""

# ---------------------------------------------------------------------------
# Earth rotation
# ---------------------------------------------------------------------------

OMEGA_EARTH = 7.292115e-5
"""float: Earth's angular velocity [rad s⁻¹]."""

# ---------------------------------------------------------------------------
# Mixed-layer depth
# ---------------------------------------------------------------------------

MLD_REFERENCE_DEPTH_M = 10.0
"""float: Reference depth for MLD threshold criterion [m].

Following Bodner et al., the nearest model level is ~9.66 m.
"""

MLD_INTEGRATION_DEPTH_M = 300.0
"""float: Upper-ocean integration depth for the MLD Depth Integration
Method [m].

The Depth Integration estimator (``mixed_layer_depth_DI``) weights the
buoyancy frequency squared N²(z) over the upper ``MLD_INTEGRATION_DEPTH_M``
metres to define the mixed-layer depth.
"""
