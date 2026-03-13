"""
Generate initial VMEC boundary conditions from physical target parameters.

Based on the analytic stellarator boundary parametrisation from
Goodman et al., PRX Energy 3, 023010 (2024), supplementary code
``generate_initial_condition.py``.

Four physical knobs fully determine a low-mode QI-flavoured boundary:
  aspect (A), elongation (E), mirror ratio (Δ), axis excursion (Z₂s).
"""

import numpy as np


def generate_boundary(nfp=2, aspect=8, elongation=2, mirror=0.20, Z2s=0.2,
                      R0=1.0):
    """
    Compute RBC/ZBS Fourier coefficients for an analytic QI boundary.

    Parameters
    ----------
    nfp : int
        Number of field periods.
    aspect : float
        Target aspect ratio A = R / a.
    elongation : float
        Cross-section elongation E >= 1.
    mirror : float
        Target mirror ratio Δ = (B_max - B_min) / (B_max + B_min).
    Z2s : float
        Vertical excursion of the magnetic axis (controls axis torsion).
    R0 : float
        Major radius scale [m].  All RBC/ZBS are multiplied by R0.

    Returns
    -------
    coeffs : dict
        ``{(n, m): (rbc, zbs)}`` Fourier coefficients in VMEC convention.
    """
    b = 1 / np.sqrt(elongation * aspect**2)
    a = elongation * b
    cplus = (elongation + 1) * b / 2
    cminus = (elongation - 1) * b / 2
    delta2 = mirror * 2
    xi = (2 - np.sqrt(4 - delta2**2)) / delta2

    coeffs = {
        (0, 0): (R0, 0.0),
        (2, 0): (-0.2 * R0, Z2s * R0),
        (2, 1): (cminus * R0, -cminus * R0),
        (0, 1): (cplus * R0, cplus * R0),
        (1, 1): (-xi * a / 2 * R0, -xi * b / 2 * R0),
        (-1, 1): (-xi * a / 2 * R0, -xi * b / 2 * R0),
    }
    return coeffs


def write_vmec_input(path, nfp=2, aspect=8, elongation=2, mirror=0.20,
                     Z2s=0.2, R0=1.0, iota_axis=0.80, iota_edge=0.60,
                     phiedge=None, ns=51):
    """
    Write a complete VMEC input file from analytic physical parameters.

    Parameters
    ----------
    path : str
        Output file path (e.g. ``input.qi_seed``).
    nfp, aspect, elongation, mirror, Z2s, R0
        Boundary parameters (see :func:`generate_boundary`).
    iota_axis, iota_edge : float
        Rotational transform at axis and edge.  Written as a linear
        ``power_series`` AI profile.
    phiedge : float or None
        Boundary toroidal flux [T·m²].  If *None*, estimated from
        ``R0``, ``aspect``, and a reference field of 1 T.
    ns : int
        Number of radial surfaces for VMEC.
    """
    coeffs = generate_boundary(nfp, aspect, elongation, mirror, Z2s, R0)

    if phiedge is None:
        a_minor = R0 / aspect
        phiedge = np.pi * a_minor**2 * 1.0

    with open(path, "w") as f:
        f.write("&INDATA\n")
        f.write(f"! Analytic QI seed: nfp={nfp}, A={aspect}, "
                f"E={elongation}, Δ={mirror}, R0={R0}\n")
        f.write("  DELT = 0.9\n  TCON0 = 1.0\n  NSTEP = 200\n")
        f.write(f"  NFP = {nfp}\n  MPOL = 5\n  NTOR = 10\n")
        f.write(f"  NS_ARRAY = {ns}\n  NITER_ARRAY = 10000\n")
        f.write("  FTOL_ARRAY = 1.0E-12\n")
        f.write(f"  PHIEDGE = {phiedge:.15e}\n")
        f.write("  GAMMA = 0.0\n  LFREEB = F\n")
        f.write("  NCURR = 0\n")
        f.write("  PIOTA_TYPE = 'power_series'\n")
        f.write(f"  AI(0) = {iota_axis:.15e}\n")
        f.write(f"  AI(1) = {iota_edge - iota_axis:.15e}\n")
        f.write("  PMASS_TYPE = 'power_series'\n")
        f.write("  PRES_SCALE = 0.0\n  AM(0) = 0.0\n")
        for (n, m) in sorted(coeffs.keys()):
            rbc, zbs = coeffs[(n, m)]
            f.write(f"  RBC({n:d},{m:d}) = {rbc:.15e}\n")
            if abs(zbs) > 1e-30:
                f.write(f"  ZBS({n:d},{m:d}) = {zbs:.15e}\n")
        f.write("/\n")
