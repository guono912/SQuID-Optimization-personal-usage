"""
Quasi-isodynamic (QI) residual targets.

Ported from Targets.py, providing three variants:
  - QuasiIsodynamicResidual0  (VMEC pest, nfp=1, simple squash+stretch+shuffle)
  - QuasiIsodynamicResidual1  (Boozer, nfp=3, poloidal/toroidal/helical contours)
  - QuasiIsodynamicResidual2  (Boozer, multi-surface, cosine stretch+shuffle,
                                published target that produced the SQuID configs)

All wrapped with a thin Optimizable subclass for use with simsopt's
LeastSquaresProblem.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from simsopt._core import Optimizable
from simsopt.mhd import Boozer

from ..core.bounce import find_bounce_points
from ..core.squash_stretch import squash_and_stretch_r2


# ===================================================================
#  QuasiIsodynamicResidual2  (the published target)
# ===================================================================

def compute_qi_residual_r2(vmec, snorms, nphi=601, nalpha=75, nBj=601,
                           mpol=20, ntor=20, nphi_out=2000,
                           arr_out=True):
    """
    Compute the QI deviation on multiple flux surfaces using the R2
    (cosine-smooth stretch + shuffle) algorithm.

    This is the exact algorithm from Targets.py:QuasiIsodynamicResidual2,
    ported verbatim with minimal cleanup.

    Returns
    -------
    Bp_arr : ndarray   — constructed B_C (last surface)
    out    : ndarray   — flattened residual vector
    """
    vmec.run()
    try:
        ns = len(snorms)
    except TypeError:
        snorms = [snorms]
        ns = 1

    weights = np.ones(ns)

    if arr_out:
        out = np.zeros((ns, nalpha, nphi))
    else:
        out = np.zeros((ns, nalpha))

    boozer = Boozer(vmec, mpol, ntor)
    boozer.register(snorms)
    boozer.run()

    nfp = vmec.wout.nfp

    if vmec.wout.bmnc[1, 1] < 0:
        phimin = np.pi / nfp
    else:
        phimin = 0
    phimax = phimin + 2 * np.pi / nfp

    phis2D = np.tile(np.linspace(phimin, phimax, nphi), (nalpha, 1)).T
    Bjs = np.linspace(0, 1, nBj)

    for si in range(ns):
        snorm = snorms[si]
        xm_nyq = boozer.bx.xm_b
        xn_nyq = boozer.bx.xn_b
        bmnc = boozer.bx.bmnc_b[:, si]

        iota = UnivariateSpline(
            vmec.s_half_grid, vmec.wout.iotas[1:], k=1, s=0
        )(snorm)

        B = np.zeros((nphi, nalpha))
        thetamin = -iota * phimin
        thetas2D = np.tile(
            np.linspace(thetamin, thetamin + 2 * np.pi, nalpha), (nphi, 1)
        ) + iota * phis2D

        for jmn in range(len(bmnc)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * thetas2D - n * phis2D
            B += bmnc[jmn] * np.cos(angle)

        Bmin = np.min(B)
        Bmax = np.max(B)
        B = (B - Bmin) / (Bmax - Bmin)

        # ---- SQUASH + STRETCH (R2 variant) ----
        Bp_arr = np.zeros((nalpha, nphi))
        bncs = np.zeros((nalpha, nBj))
        wts = np.zeros(nalpha)

        for ialpha in range(nalpha):
            Ba = B[:, ialpha].copy()
            phisa = phis2D[:, ialpha]

            B_C = squash_and_stretch_r2(Ba)
            Bp_arr[ialpha, :] = B_C

            wtf = UnivariateSpline(
                phisa, np.abs(Ba - B_C) ** 2, k=1, s=0
            )
            integral = wtf.integral(phimin, phimax)
            wts[ialpha] = (phimax - phimin) / max(integral, 1e-30)

            for j in range(nBj):
                Bj = Bjs[j]
                phip1, phip2, _, _ = find_bounce_points(
                    phisa, B_C, Bj, 1.0, 0.0
                )
                bncs[ialpha, j] = phip2 - phip1

        # ---- SHUFFLE ----
        wts = wts / np.sum(wts)
        mbncs = np.sum(bncs * wts[:, None], axis=0)
        mbncf = UnivariateSpline(mbncs, Bjs, k=1, s=0)

        mean_denom = 0.0
        for ialpha in range(nalpha):
            Bpp_vals = mbncf(bncs[ialpha, :])
            Bp_to_Bpp_f = UnivariateSpline(Bjs, Bpp_vals, k=1, s=0)
            Ba = B[:, ialpha]
            Bpp = Bp_to_Bpp_f(Bp_arr[ialpha, :])

            denom = 1.0
            pen = (Bpp - Ba) / denom
            mean_denom += np.mean(denom) / nalpha

            if arr_out:
                out[si, ialpha, :] = weights[si] * pen / np.sqrt(nphi)
            else:
                out[si, ialpha] = weights[si] * np.sqrt(np.mean(pen ** 2))

        if arr_out:
            out[si, :, :] *= mean_denom
        else:
            out[si, :] *= mean_denom

    out = out.flatten()
    out = out / np.sqrt(nalpha)
    return Bp_arr, out


class QIResidual(Optimizable):
    """
    simsopt Optimizable wrapper around compute_qi_residual_r2.

    Usage::

        qi = QIResidual(vmec, snorms=[0.25, 0.5])
        prob = LeastSquaresProblem.from_tuples([
            (qi.residuals, 0.0, weight),
        ])
    """

    def __init__(self, vmec, snorms, nphi=601, nalpha=75, nBj=601,
                 mpol=20, ntor=20, nphi_out=2000, arr_out=True):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.snorms = np.atleast_1d(snorms).tolist()
        self.nphi = nphi
        self.nalpha = nalpha
        self.nBj = nBj
        self.mpol = mpol
        self.ntor = ntor
        self.nphi_out = nphi_out
        self.arr_out = arr_out
        self._cache_x = None
        self._residuals = None

    def _compute(self):
        try:
            cx = tuple(self.vmec.x)
        except Exception:
            cx = None
        if cx is not None and cx == self._cache_x:
            return
        _, self._residuals = compute_qi_residual_r2(
            self.vmec, self.snorms, self.nphi, self.nalpha,
            self.nBj, self.mpol, self.ntor, self.nphi_out,
            self.arr_out,
        )
        self._cache_x = cx

    def residuals(self):
        self._compute()
        return self._residuals

    def total(self):
        self._compute()
        return float(np.sum(self._residuals ** 2))
