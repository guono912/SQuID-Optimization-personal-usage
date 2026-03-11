"""
Squash-and-Stretch algorithms for constructing the ideal QI field B_C.

Variants (from simplest to most complete):
  - simple:  flatten + linear rescale (max_J_evaluation.py)
  - R0:      flatten + linear rescale + shuffle (Targets.py, nfp=1)
  - R2:      flatten + cosine-smooth rescale + shuffle
             (Targets.py:QuasiIsodynamicResidual2, the published target)

Goodman (2023/2024): B_C satisfies the three QI conditions when all
three steps (squash, stretch, shuffle) are performed correctly.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from ..core.bounce import find_bounce_points


# ------------------------------------------------------------------ #
#  Simple variant (from max_J_evaluation.py)
# ------------------------------------------------------------------ #

def squash_and_stretch_simple(zeta, B_I, B_min_surface, B_max_surface):
    """
    Two-step squash-and-stretch without shuffle.

    1. Squash: force B_C to be monotone on each arm
    2. Stretch: linearly rescale each arm to [B_min_surface, B_max_surface]

    Returns B_C (same shape as B_I).
    """
    B_C = B_I.copy()
    idx_min = int(np.argmin(B_C))

    # squash left arm
    for i in range(idx_min - 1, -1, -1):
        if B_C[i] < B_C[i + 1]:
            B_C[i] = B_C[i + 1]
    # squash right arm
    for i in range(idx_min + 1, len(B_C)):
        if B_C[i] < B_C[i - 1]:
            B_C[i] = B_C[i - 1]

    # stretch left arm
    left = B_C[:idx_min + 1]
    if abs(left[0] - left[-1]) > 1e-12:
        left[:] = B_min_surface + (left - left[-1]) * (B_max_surface - B_min_surface) / (left[0] - left[-1])

    # stretch right arm
    right = B_C[idx_min:]
    if abs(right[-1] - right[0]) > 1e-12:
        right[:] = B_min_surface + (right - right[0]) * (B_max_surface - B_min_surface) / (right[-1] - right[0])

    return B_C


# ------------------------------------------------------------------ #
#  R2 variant (from Targets.py:QuasiIsodynamicResidual2)
# ------------------------------------------------------------------ #

def squash_and_stretch_r2(B_fieldline, Bmin_norm=0.0, Bmax_norm=1.0,
                          pmax=50, pmin=50):
    """
    R2 squash-and-stretch on a *single* pre-normalised field line.

    B_fieldline is assumed normalised to [0, 1] with:
        Bmin_norm = 0, Bmax_norm = 1

    Steps:
      1. Squash: clamp & flatten non-monotone regions
      2. Stretch: cosine-based smooth rescale (power-50 envelope)

    Returns (Bl_stretched, Br_stretched) concatenated as B_C.
    """
    Ba = B_fieldline.copy()
    indmin = np.argmin(Ba)

    # --- left arm squash ---
    Bl = Ba[:indmin + 1].copy()
    indmax_l = np.argmax(Bl)
    Bl[:indmax_l] = Bl[indmax_l]
    for i in range(len(Bl) - 1):
        if Bl[i] <= Bl[i + 1]:
            jf = len(Bl) - 1
            for j in range(i + 1, len(Bl)):
                if Bl[j] < Bl[i]:
                    jf = j
                    break
            Bl[i:jf] = Bl[i]

    # --- right arm squash ---
    Br = Ba[indmin:].copy()
    indmax_r = np.argmax(Br)
    Br[indmax_r:] = Br[indmax_r]
    for j in range(len(Br) - 1, 1, -1):
        if Br[j - 1] >= Br[j]:
            kf = 0
            for k in range(j - 1, 1, -1):
                if Br[k] < Br[j]:
                    kf = k
                    break
            Br[kf + 1:j] = Br[j]

    # --- stretch (cosine-based, power-50) ---
    def _F_left(Bl_arm, pmax_=pmax, pmin_=pmin):
        R1 = 1.0 - Bl_arm[0]
        R2 = -Bl_arm[-1]
        if abs(Bl_arm[-1] - Bl_arm[0]) < 1e-15:
            return np.zeros_like(Bl_arm)
        x = (Bl_arm - Bl_arm[0]) / (Bl_arm[-1] - Bl_arm[0])
        xlp5 = x < 0.5
        cos_term = ((np.cos(2 * np.pi * x) + 1) / 2)
        t1 = xlp5 * R1 * cos_term ** pmax_
        t2 = (~xlp5) * R2 * cos_term ** pmin_
        return t1 + t2

    def _F_right(Br_arm, pmax_=pmax, pmin_=pmin):
        R1 = 1.0 - Br_arm[-1]
        R2 = -Br_arm[0]
        if abs(Br_arm[-1] - Br_arm[0]) < 1e-15:
            return np.zeros_like(Br_arm)
        x = (Br_arm - Br_arm[0]) / (Br_arm[-1] - Br_arm[0])
        xlp5 = x < 0.5
        cos_term = ((np.cos(2 * np.pi * x) + 1) / 2)
        t1 = xlp5 * R2 * cos_term ** pmin_
        t2 = (~xlp5) * R1 * cos_term ** pmax_
        return t1 + t2

    Bl = Bl + _F_left(Bl)
    Br = Br + _F_right(Br)

    B_C = np.concatenate([Bl[:-1], Br])
    return B_C


def shuffle(Bp_arr, B_arr, phis2D, nBj, Bjs, nfp, nalpha, nphi,
            weights=None, arr_out=True):
    """
    Shuffle step: remap bounce distances to weighted-mean across all
    field lines, enforcing the third QI condition (constant bounce
    distance at fixed B*).

    Ported from Targets.py:QuasiIsodynamicResidual2 (lines 694-743).

    Parameters
    ----------
    Bp_arr  : (nalpha, nphi) — squash-stretched B on each field line
    B_arr   : (nphi, nalpha) — actual |B| (normalised to [0,1])
    phis2D  : (nphi, nalpha) — toroidal angle grid
    nBj     : int            — number of B* levels
    Bjs     : 1-D (nBj,)     — B* sampling points in [0, 1]
    nfp     : int
    nalpha, nphi : int
    weights : 1-D (nalpha,) or None
    arr_out : bool — if True, return (nalpha, nphi) residual; else (nalpha,)

    Returns
    -------
    Bp_arr  : (nalpha, nphi) — updated B_C field
    out     : residual vector
    """
    bncs = np.zeros((nalpha, nBj))
    phipp_arr = np.zeros((nalpha, 2 * nBj - 1))
    wts = np.zeros(nalpha)

    phimin = phis2D[0, 0]
    phimax = phis2D[-1, 0]

    for ialpha in range(nalpha):
        Blr = Bp_arr[ialpha, :]
        phisa = phis2D[:, ialpha]

        wtf = UnivariateSpline(phisa, np.abs(B_arr[:, ialpha] - Blr) ** 2, k=1, s=0)
        integral = wtf.integral(phimin, phimax)
        wts[ialpha] = (phimax - phimin) / max(integral, 1e-30)

        for j in range(nBj):
            Bj = Bjs[j]
            phip1, phip2, _, _ = find_bounce_points(phisa, Blr, Bj, 1.0, 0.0)
            bncs[ialpha, j] = phip2 - phip1
            phipp_arr[ialpha, nBj - j - 1] = phip1
            phipp_arr[ialpha, nBj + j - 1] = phip2

    wts = wts / np.sum(wts)
    mbncs = np.sum(bncs * wts[:, None], axis=0)
    mbncf = UnivariateSpline(mbncs, Bjs, k=1, s=0)

    if arr_out:
        out = np.zeros((nalpha, nphi))
    else:
        out = np.zeros(nalpha)

    mean_denom = 0.0
    for ialpha in range(nalpha):
        Bpp_arr_local = mbncf(bncs[ialpha, :])
        Bp_to_Bpp_f = UnivariateSpline(Bjs, Bpp_arr_local, k=1, s=0)
        Ba = B_arr[:, ialpha]
        Bpp = Bp_to_Bpp_f(Bp_arr[ialpha, :])

        denom = 1.0
        pen = (Bpp - Ba) / denom
        mean_denom += np.mean(denom) / nalpha

        if arr_out:
            w = 1.0 if weights is None else weights
            out[ialpha, :] = w * pen / np.sqrt(nphi)
        else:
            w = 1.0 if weights is None else weights
            out[ialpha] = w * np.sqrt(np.mean(pen ** 2))

    out *= mean_denom
    return Bp_arr, out
