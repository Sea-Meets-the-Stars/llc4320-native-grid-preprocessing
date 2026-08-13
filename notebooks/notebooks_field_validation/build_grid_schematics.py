"""Vertical-grid schematic for the LLC4320 test notebooks.

Draws where each vertical quantity lives on an MITgcm/ECCO column and names
every coordinate and spacing variable at the depth it belongs to.

Companion to ``cgrid_schematic.py``, which does the same for the horizontal.

Convention
----------
The figure is drawn in **dbof's positive-downward depth** (0 at the surface,
increasing with depth), matching the rest of the vertical-helpers notebook and
everything downstream of ``_get_depth_coord``.  MITgcm/ECCO themselves use a
**positive-upward** ``Z`` -- the axis points up, so ``Z`` is *negative* below
the surface -- and each label carries that native value in parentheses.  See the MITgcm vertical-grid documentation:
https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html

Note on the ``Zl`` / ``Zu`` naming: the ``l`` and ``u`` refer to the *index*
position, not to depth.  ``Zl[k]`` is the interface on the low-index
(SHALLOWER) side of cell ``k``; ``Zu[k]`` is the interface on the high-index
(DEEPER) side.  Flipping the sign convention does not change that -- it is a
statement about array indexing, not about geometry.  This is the single most
common source of off-by-one errors when reading MITgcm vertical output.

Depths here are round illustrative numbers, not the real LLC4320 levels -- the
point is the topology.  Pass ``interfaces=`` to draw other levels.
"""

import matplotlib.pyplot as plt

C_CENTRE = "tab:blue"     # tracer cell centres: Z, k
C_IFACE = "tab:red"       # cell interfaces: Zl / Zu / Zp1
C_SPACE = "0.25"          # spacing annotations: drF, drC

MITGCM_DOC = ("https://mitgcm.readthedocs.io/en/latest/algorithm/"
              "vert-grid.html")


def draw_vertical_schematic(axes=None, interfaces=(0.0, 50.0, 120.0)):
    """Draw an annotated MITgcm vertical-grid schematic, positive-downward.

    Parameters
    ----------
    axes : pair of matplotlib Axes, or None
        (column axes, naming-table axes).  None creates a new figure.
    interfaces : sequence of float
        Interface depths in metres, **positive downward**, shallowest first.
        Defaults to two illustrative cells.

    Returns
    -------
    (ax_col, ax_key) : the two axes drawn into.
    """
    if axes is None:
        fig, axes = plt.subplots(
            1, 2, figsize=(12.5, 7.0),
            gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax, ax_key = axes

    ifc = list(interfaces)
    nk = len(ifc) - 1
    ctr = [0.5 * (ifc[k] + ifc[k + 1]) for k in range(nk)]

    x0, x1 = 0.0, 1.0          # horizontal extent of the interface lines
    xcol = 0.12                # the column itself

    # ---- interfaces -------------------------------------------------------
    for kp1, z in enumerate(ifc):
        ax.plot([x0, x1], [z, z], ls="--", color=C_IFACE, lw=1.2, zorder=1)
        ax.plot(xcol, z, "o", color=C_IFACE, ms=11, zorder=3)

        names = [f"Zp1[k_p1={kp1}]"]
        if kp1 < nk:
            names.append(f"Zl[k_l={kp1}]")
        if kp1 > 0:
            names.append(f"Zu[k_u={kp1 - 1}]")
        ax.text(x1 + 0.04, z,
                f"depth {z:g} m  =  " + " = ".join(names) +
                f"      (native Z = {-z if z else 0:g})",
                va="center", fontsize=9.5, color=C_IFACE)

    # ---- cell centres -----------------------------------------------------
    for k, z in enumerate(ctr):
        ax.plot([x0, x1], [z, z], ls=":", color=C_CENTRE, lw=1.0, zorder=1)
        ax.plot(xcol, z, "X", color=C_CENTRE, ms=13, zorder=3)
        ax.text(x1 + 0.04, z,
                f"depth {z:g} m  =  Z[k={k}]     "
                f"(Theta, Salt; U and V at the same k)",
                va="center", fontsize=9.5, color=C_CENTRE)

    # ---- W lives on the interfaces ---------------------------------------
    for z in ifc[:-1]:
        ax.plot(xcol + 0.30, z, "v", color=C_IFACE, ms=9, zorder=3,
                mfc="none")
    span = ifc[-1] - ifc[0]
    ax.annotate("W  (k_l, j, i)", xy=(xcol + 0.30, ifc[0]),
                xytext=(xcol + 0.40, ifc[0] + 0.10 * span),
                fontsize=9.5, color=C_IFACE,
                arrowprops=dict(arrowstyle="->", color=C_IFACE, lw=1.0))

    # ---- drF: thickness of cell 0 (interface to interface) ----------------
    xd = xcol - 0.075
    ax.annotate("", xy=(xd, ifc[1]), xytext=(xd, ifc[0]),
                arrowprops=dict(arrowstyle="<->", color=C_SPACE, lw=1.8))
    ax.text(xd - 0.03, 0.5 * (ifc[0] + ifc[1]), "drF[k=0]",
            rotation=90, va="center", ha="right", fontsize=10,
            color=C_SPACE)

    # ---- drC: centre-to-centre spacing ------------------------------------
    if nk >= 2:
        xc = xcol + 0.16
        ax.annotate("", xy=(xc, ctr[1]), xytext=(xc, ctr[0]),
                    arrowprops=dict(arrowstyle="<->", color=C_SPACE, lw=1.8))
        ax.text(xc + 0.03, 0.5 * (ctr[0] + ctr[1]),
                "drC\n(centre to centre)",
                va="center", ha="left", fontsize=10, color=C_SPACE)

    pad = 0.06 * span
    ax.set_ylim(ifc[0] - pad, ifc[-1] + pad)
    ax.invert_yaxis()                      # depth increases downward
    ax.set_xlim(-0.16, 2.35)
    ax.set_xticks([])
    ax.set_ylabel("depth (m, POSITIVE downward)  --  dbof convention")
    ax.set_title("LLC4320 vertical grid", fontsize=11)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    # ---- naming key -------------------------------------------------------
    ax_key.axis("off")
    ax_key.set_title("What each name means", fontsize=11, loc="left")
    rows = [
        ("Z, k", C_CENTRE, "tracer-cell CENTRE depth. Theta and Salt live\n"
                           "here, and so does every dbof depth reduction."),
        ("Zl, k_l", C_IFACE, "interface on the LOW-INDEX side of cell k --\n"
                             "i.e. the SHALLOWER one. W lives here."),
        ("Zu, k_u", C_IFACE, "interface on the HIGH-INDEX side of cell k --\n"
                             "i.e. the DEEPER one."),
        ("Zp1, k_p1", C_IFACE, "all nk+1 interfaces in one array\n"
                               "(Zl and Zu are overlapping subsets of it)."),
        ("drF", C_SPACE, "THICKNESS of cell k = |Zu[k] - Zl[k]|.\n"
                         "The weight in thickness-weighted vertical means."),
        ("drC", C_SPACE, "distance between adjacent cell CENTRES.\n"
                         "Used for k_l-level spacing, not for drF weights."),
    ]
    y = 0.94
    for name, colour, desc in rows:
        ax_key.text(0.0, y, name, fontsize=11, fontweight="bold",
                    color=colour, va="top")
        ax_key.text(0.30, y, desc, fontsize=9.5, va="top", color="0.15")
        y -= 0.145

    ax_key.text(0.0, y + 0.03,
                "l and u refer to INDEX POSITION, not depth.\n"
                "Zl is the shallower interface; Zu is the deeper one --\n"
                "this is true in either sign convention.",
                fontsize=9.5, va="top", color="tab:red", style="italic")
    ax_key.text(0.0, y - 0.085,
                "3D at cell centres (k):  Theta, Salt   |   3D on the C-grid\n"
                "faces at the same k:  U, V   |   3D on interfaces (k_l):  W\n"
                "2D surface fields:  Eta, oceQnet, oceTAUX, oceTAUY",
                fontsize=9, va="top", color="0.15")
    ax_key.text(0.0, y - 0.185,
                "SIGN CONVENTION: drawn as dbof depth, axis POSITIVE DOWNWARD,\n"
                "so values are positive below the surface. MITgcm/ECCO use a\n"
                "POSITIVE-UPWARD Z, so their values are NEGATIVE below the\n"
                "surface (given in parentheses on the left); depth = -Z.\n"
                "See Section 2 for the conversion and the UserWarning it emits.\n\n"
                "MITgcm vertical grid documentation:\n" + MITGCM_DOC,
                fontsize=8, va="top", color="0.15",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd",
                          ec="#c9a227"))
    return ax, ax_key


if __name__ == "__main__":
    draw_vertical_schematic()
    plt.tight_layout()
    plt.savefig("vertical_schematic.png", dpi=130)



"""Arakawa C-grid schematic for the test notebooks.

Draws where each quantity lives on an LLC (MITgcm/ECCO) grid cell, using
the ECCO/xmitgcm dimension names, and annotates the operation path used
by ``calculate_native_gradient_tracer`` / ``calculate_jacobian``:
difference (lands on a cell face), divide by the spacing metric (defined
on that same face), interpolate back to the tracer centre, then rotate
with ``CS``/``SN`` (defined at the centre).

This addresses the PR #26 review request for "some form of representing
where on a gridcell the operations are actually going".

Generated by JXP and Claude
"""

import matplotlib.pyplot as plt

# Colours shared by both notebooks so the point types read consistently.
C_TRACER = "tab:blue"
C_UPOINT = "tab:red"
C_VPOINT = "tab:green"
C_CORNER = "0.45"


def draw_cgrid_schematic(ax=None, ncells=2):
    """Draw an annotated Arakawa C-grid schematic with ECCO naming.

    Shows an ``ncells x ncells`` block of tracer cells with every point
    type marked and labelled with its ECCO/xmitgcm dimensions:

    * tracer/centre points ``(j, i)`` -- where ``Theta``, ``Salt``,
      ``CS``, ``SN`` live and where all gradient outputs land;
    * U points ``(j, i_g)`` on the *west* cell faces -- where ``U`` and
      ``dxC`` live (``i_g`` is the X axis staggered by half a cell);
    * V points ``(j_g, i)`` on the *south* cell faces -- where ``V`` and
      ``dyC`` live;
    * corner (vorticity) points ``(j_g, i_g)`` -- not used by the
      gradient routines, shown for completeness.

    An arrow chain on the lower-left cell annotates the tracer-gradient
    pipeline: ``grid.diff`` lands on the U point, the result is divided
    by ``dxC`` (defined there), ``grid.interp`` brings it back to the
    centre, and the rotation by ``CS``/``SN`` happens at the centre.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw into.  None (default) creates a new figure.
    ncells : int
        Number of tracer cells per side of the schematic.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the schematic.

    Generated by JXP and Claude
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 7.0))

    n = ncells
    # Cell edges at integers; tracer centres at half-integers.
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color="0.75", lw=1, zorder=1)
        ax.plot([k, k], [0, n], color="0.75", lw=1, zorder=1)

    for jj in range(n):
        for ii in range(n):
            xc, yc = ii + 0.5, jj + 0.5
            # Tracer centre (j, i).
            ax.plot(xc, yc, "o", color=C_TRACER, ms=12, zorder=3)
            # U point on the west face (j, i_g).
            ax.plot(ii, yc, ">", color=C_UPOINT, ms=12, zorder=3)
            # V point on the south face (j_g, i).
            ax.plot(xc, jj, "^", color=C_VPOINT, ms=12, zorder=3)
            # Corner / vorticity point (j_g, i_g).
            ax.plot(ii, jj, "x", color=C_CORNER, ms=9, mew=2, zorder=3)
    # Close the east/north staggered points of the outer ring.
    for jj in range(n):
        ax.plot(n, jj + 0.5, ">", color=C_UPOINT, ms=12, zorder=3)
    for ii in range(n):
        ax.plot(ii + 0.5, n, "^", color=C_VPOINT, ms=12, zorder=3)
    for k in range(n + 1):
        ax.plot(n, k, "x", color=C_CORNER, ms=9, mew=2, zorder=3)
        ax.plot(k, n, "x", color=C_CORNER, ms=9, mew=2, zorder=3)

    # Point-type labels on the lower-left cell.
    ax.annotate("tracer point (j, i)\nTheta, Salt, CS, SN\n"
                "ALL gradient outputs",
                xy=(0.5, 0.5), xytext=(0.62, 0.62),
                textcoords=ax.transData, fontsize=9, color=C_TRACER,
                arrowprops=dict(arrowstyle="->", color=C_TRACER))
    ax.annotate("U point (j, i_g)\nU, dxC\n(west cell face)",
                xy=(0.0, 0.5), xytext=(-0.68, 0.95),
                fontsize=9, color=C_UPOINT, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_UPOINT))
    ax.annotate("V point (j_g, i)\nV, dyC\n(south cell face)",
                xy=(0.5, 0.0), xytext=(0.75, -0.42),
                fontsize=9, color=C_VPOINT, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_VPOINT))
    ax.annotate("corner (j_g, i_g)\n(vorticity point;\nunused here)",
                xy=(2.0, 1.0), xytext=(2.12, 0.62),
                fontsize=9, color=C_CORNER, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_CORNER))

    # Operation-path annotation for the tracer gradient (X direction):
    # centre -> U point (diff / dxC), U point -> centre (interp, rotate).
    ax.annotate("", xy=(1.0, 1.5), xytext=(0.5, 1.5),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_UPOINT))
    ax.annotate("", xy=(1.5, 1.5), xytext=(1.0, 1.5),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_TRACER))
    ax.text(-0.65, 1.62,
            "1. grid.diff(s, 'X') / dxC\n    lands on U point (i_g)",
            fontsize=9, color=C_UPOINT)
    ax.text(1.62, 1.42,
            "2. grid.interp back\n    to tracer point (i)\n"
            "3. rotate with CS/SN\n    at the tracer point",
            fontsize=9, color=C_TRACER)

    ax.set_xlim(-0.75, n + 1.45)
    ax.set_ylim(-0.6, n + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("Arakawa C grid, ECCO naming: where every operation "
                 "lives", fontsize=11)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_TRACER, ms=10,
                   label="tracer / centre (j, i)"),
        plt.Line2D([], [], marker=">", ls="", color=C_UPOINT, ms=10,
                   label="U point (j, i_g)"),
        plt.Line2D([], [], marker="^", ls="", color=C_VPOINT, ms=10,
                   label="V point (j_g, i)"),
        plt.Line2D([], [], marker="x", ls="", color=C_CORNER, ms=9,
                   mew=2, label="corner (j_g, i_g)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              framealpha=0.95)
    return ax
