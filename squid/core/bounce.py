"""
Bounce-point finding utilities.

Merged from:
  - Helpers.py  (GetBranches)
  - max_J_evaluation.py  (_find_bounce_points)
"""

import numpy as np


def find_bounce_points(phi, B, B_star, B_max=None, B_min=None):
    """
    Find the two toroidal angles where B(phi) = B_star (bounce points).

    This is a thin wrapper that dispatches to the robust algorithm
    ported from Helpers.py.

    Parameters
    ----------
    phi : 1-D array   toroidal-angle grid
    B   : 1-D array   |B| evaluated on *phi*
    B_star : float     mirror-point field level
    B_max, B_min : float  (optional, computed if not given)

    Returns
    -------
    phi1, phi2 : float   left and right bounce angles
    m1, m2     : float   slopes at crossings (for diagnostics)
    """
    if B_max is None:
        B_max = np.max(B)
    if B_min is None:
        B_min = np.min(B)
    return _get_branches(phi, B, B_star, B_max, B_min)


def _get_branches(phiBs, Ba, Bj, Bmax, Bmin):
    """
    Core bounce-point finder ported from Helpers.py:GetBranches.

    Handles edge cases (B near Bmin/Bmax, single crossing, etc.)
    via linear interpolation at sign changes.
    """
    diffs = Ba - Bj
    diffsgn = diffs[:-1] * diffs[1:]
    inds = np.where(diffsgn < 0)[0]
    inds = np.sort(inds)

    if Bj - Bmin < 1e-15 or Bj < Bmin:
        imin = np.argmin(Ba)
        phimin = phiBs[imin]
        return phimin, phimin, imin, imin
    elif Bmax - Bj < 1e-15 or Bj > Bmax:
        return 0, phiBs[-1], 0, len(Ba) - 1

    if len(inds) != 2:
        inds = np.where(diffsgn <= 0)[0]
        for iind in range(1, len(inds)):
            if inds[iind] != inds[iind - 1] + 1:
                inds = [inds[iind - 1], inds[-1]]
                break

    if len(inds) == 1:
        if Bj > Ba[-1]:
            ind1 = inds[0]
            dy1 = Ba[ind1] - Ba[ind1 + 1]
            dx1 = phiBs[ind1] - phiBs[ind1 + 1]
            m1 = dy1 / dx1
            b1 = Ba[ind1] - m1 * phiBs[ind1]
            phiB1 = (Bj - b1) / m1 if m1 != 0 else phiBs[ind1]
            return phiB1, phiBs[-1], m1, 0
        elif Bj > Ba[0]:
            ind2 = inds[0]
            dy2 = Ba[ind2] - Ba[ind2 + 1]
            dx2 = phiBs[ind2] - phiBs[ind2 + 1]
            m2 = dy2 / dx2
            b2 = Ba[ind2] - m2 * phiBs[ind2]
            phiB2 = (Bj - b2) / m2 if m2 != 0 else phiBs[ind2 + 1]
            return 0, phiB2, 0, m2

    ind1 = inds[0]
    ind2 = inds[1]

    dy1 = Ba[ind1] - Ba[ind1 + 1]
    dx1 = phiBs[ind1] - phiBs[ind1 + 1]
    m1 = dy1 / dx1
    b1 = Ba[ind1] - m1 * phiBs[ind1]
    phiB1 = (Bj - b1) / m1 if m1 != 0 else phiBs[ind1]

    dy2 = Ba[ind2] - Ba[ind2 + 1]
    dx2 = phiBs[ind2] - phiBs[ind2 + 1]
    m2 = dy2 / dx2
    b2 = Ba[ind2] - m2 * phiBs[ind2]
    phiB2 = (Bj - b2) / m2 if m2 != 0 else phiBs[ind2 + 1]

    return phiB1, phiB2, m1, m2


def find_bounce_points_boozer(zeta, B_C, B_star):
    """
    Boozer-coordinate bounce-point finder (from max_J_evaluation.py).

    B_C must be monotone-on-each-arm (output of squash_and_stretch).
    Uses numpy.interp on each monotone arm.
    """
    idx_min = int(np.argmin(B_C))
    left_z = zeta[:idx_min + 1]
    left_B = B_C[:idx_min + 1]
    right_z = zeta[idx_min:]
    right_B = B_C[idx_min:]

    if B_star <= np.min(B_C) or B_star >= np.max(B_C):
        return None, None

    z1 = float(np.interp(B_star, left_B[::-1], left_z[::-1]))
    z2 = float(np.interp(B_star, right_B, right_z))
    return z1, z2
