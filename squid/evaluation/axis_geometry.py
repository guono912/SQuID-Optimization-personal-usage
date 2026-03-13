"""
Magnetic axis Frenet-Serret geometry — curvature κ and torsion τ.

Ported from Goodman et al. PRX Energy 3, 023010 (2024),
``plots/plot_axis_params/PltAxisProperties.py``.

These axis-level quantities govern the near-axis expansion of |B|.
For QI stellarators, curvature should be minimal at the straight
sections and torsion should change sign appropriately within each
half-period.
"""

import numpy as np


def axis_curvature_torsion(raxis_cc, zaxis_cs, nfp, num_points=2000):
    """
    Compute curvature κ(φ) and torsion τ(φ) of the magnetic axis.

    Parameters
    ----------
    raxis_cc : 1-d array
        Cosine Fourier coefficients of R(φ) for the axis.
    zaxis_cs : 1-d array
        Sine   Fourier coefficients of Z(φ) for the axis.
    nfp : int
        Number of field periods.
    num_points : int
        Grid resolution on [0, 2π/nfp).

    Returns
    -------
    dict with keys:
        ``phi``, ``R0``, ``Z0``, ``curvature``, ``torsion``,
        ``axis_length`` (total axis length in metres).
    """
    phi = np.linspace(0, 2 * np.pi / nfp, num_points, endpoint=False)

    raxis_cc = np.asarray(raxis_cc, dtype=float)
    zaxis_cs = np.asarray(zaxis_cs, dtype=float)
    n_max = max(len(raxis_cc), len(zaxis_cs))
    rcc = np.zeros(n_max)
    zcs = np.zeros(n_max)
    rcc[:len(raxis_cc)] = raxis_cc
    zcs[:len(zaxis_cs)] = zaxis_cs

    R0    = np.zeros_like(phi)
    Z0    = np.zeros_like(phi)
    R0p   = np.zeros_like(phi)
    Z0p   = np.zeros_like(phi)
    R0pp  = np.zeros_like(phi)
    Z0pp  = np.zeros_like(phi)
    R0ppp = np.zeros_like(phi)
    Z0ppp = np.zeros_like(phi)

    for imn in range(n_max):
        n = nfp * imn
        angle = n * phi
        ca, sa = np.cos(angle), np.sin(angle)
        R0    += rcc[imn] * ca
        Z0    += zcs[imn] * sa
        R0p   += -n * rcc[imn] * sa
        Z0p   +=  n * zcs[imn] * ca
        R0pp  += -n**2 * rcc[imn] * ca
        Z0pp  += -n**2 * zcs[imn] * ca
        R0ppp +=  n**3 * rcc[imn] * sa
        Z0ppp +=  n**3 * zcs[imn] * sa

    dl = np.sqrt(R0**2 + R0p**2 + Z0p**2)

    dr  = np.array([R0p,      R0,        Z0p])
    d2r = np.array([R0pp - R0, 2 * R0p,   Z0pp])
    d3r = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp])

    cross_dr_d2r = np.array([
        dr[1] * d2r[2] - dr[2] * d2r[1],
        dr[2] * d2r[0] - dr[0] * d2r[2],
        dr[0] * d2r[1] - dr[1] * d2r[0],
    ])
    curvature = np.sqrt(np.sum(cross_dr_d2r**2, axis=0)) / dl**3

    torsion_num = np.sum(dr * np.array([
        d2r[1] * d3r[2] - d2r[2] * d3r[1],
        d2r[2] * d3r[0] - d2r[0] * d3r[2],
        d2r[0] * d3r[1] - d2r[1] * d3r[0],
    ]), axis=0)
    torsion_den = np.sum(cross_dr_d2r**2, axis=0)
    torsion = np.where(torsion_den > 1e-30,
                       torsion_num / torsion_den, 0.0)

    dphi = phi[1] - phi[0]
    axis_length = float(np.sum(dl) * dphi * nfp)

    return dict(
        phi=phi,
        R0=R0,
        Z0=Z0,
        curvature=curvature,
        torsion=torsion,
        axis_length=axis_length,
    )


def axis_geometry_from_vmec(vmec, num_points=2000):
    """
    Extract axis curvature and torsion from a VMEC equilibrium.

    Parameters
    ----------
    vmec : simsopt.mhd.Vmec
        VMEC equilibrium (must have been run).
    num_points : int
        Grid resolution per half-period.

    Returns
    -------
    Same as :func:`axis_curvature_torsion`.
    """
    raxis_cc = np.array(vmec.wout.raxis_cc)
    zaxis_cs = np.array(vmec.wout.zaxis_cs)
    nfp = int(vmec.wout.nfp)
    return axis_curvature_torsion(raxis_cc, zaxis_cs, nfp, num_points)


def plot_axis_geometry(vmec, num_points=2000):
    """Plot curvature and torsion of the magnetic axis."""
    import matplotlib.pyplot as plt

    info = axis_geometry_from_vmec(vmec, num_points)
    phi = info["phi"]
    nfp = int(vmec.wout.nfp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(phi, info["curvature"], "k-", lw=1.5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\kappa$ (curvature)")
    ax.set_yscale("log")
    ax.set_xlim(0, 2 * np.pi / nfp)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(phi, info["torsion"], "k-", lw=1.5)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\tau$ (torsion)")
    ax.set_xlim(0, 2 * np.pi / nfp)
    ax.set_ylim(-15, 15)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Magnetic axis geometry (nfp={nfp}, "
                 f"L={info['axis_length']:.3f} m)")
    plt.tight_layout()
    return fig
