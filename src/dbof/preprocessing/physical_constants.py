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

# ---------------------------------------------------------------------------
# Reference density
# ---------------------------------------------------------------------------

RHO0_REFERENCE = 1000.0
"""float: Reference density ρ₀ [kg m⁻³].

Used consistently everywhere a reference density appears:

- σ₀ anomaly:            σ₀ = ρ(S, Θ, p=0) − ρ₀
- buoyancy:              b = g σ₀ / ρ₀
- Brunt–Väisälä:         N² = (g / ρ₀) dρ/dz
- heat-content integral: Q = ρ₀ Cₚ ∫ θ dz
- Turner-angle linear-EOS identity, Ekman transport/pumping

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
