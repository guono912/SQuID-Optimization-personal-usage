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
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from simsopt._core import Optimizable
from simsopt.mhd import Boozer

from ..core.boozer_utils import run_boozer, reconstruct_B
from ..core.bounce import find_bounce_points
from ..core.squash_stretch import squash_and_stretch_r2


# ===================================================================
#  QuasiIsodynamicResidual2  (the published target)
# ===================================================================

def compute_qi_residual_r2(vmec, snorms, nphi=601, nalpha=75, nBj=601,
                           mpol=20, ntor=20, arr_out=True):
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
    # arr_out=True already divides each field-line trace by sqrt(nphi), so the
    # remaining resolution normalisation is over (surface, alpha), matching the
    # arr_out=False per-line RMS branch.
    out = out / np.sqrt(max(ns * nalpha, 1))
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
                 mpol=20, ntor=20, arr_out=True):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.snorms = np.atleast_1d(snorms).tolist()
        self.nphi = nphi
        self.nalpha = nalpha
        self.nBj = nBj
        self.mpol = mpol
        self.ntor = ntor
        self.arr_out = arr_out
        self._cache_x = None
        self._residuals = None
        self._warned_failure = False

    def _fallback_residuals(self):
        if self._residuals is not None:
            return np.full_like(self._residuals, 1e3, dtype=float)
        ns = max(len(self.snorms), 1)
        if self.arr_out:
            size = ns * self.nalpha * self.nphi
        else:
            size = ns * self.nalpha
        return np.full(size, 1e3, dtype=float)

    def _compute(self):
        try:
            cx = tuple(self.vmec.x)
        except Exception:
            cx = None
        if cx is not None and cx == self._cache_x:
            return
        try:
            _, self._residuals = compute_qi_residual_r2(
                self.vmec, self.snorms, self.nphi, self.nalpha,
                self.nBj, self.mpol, self.ntor, self.arr_out,
            )
        except Exception as exc:
            if not self._warned_failure:
                print(f"[QIResidual] QI R2 evaluation failed; using large residual penalty: {exc}")
                self._warned_failure = True
            self._residuals = self._fallback_residuals()
        self._cache_x = cx

    def residuals(self):
        self._compute()
        return self._residuals

    def total(self):
        self._compute()
        return float(np.sum(self._residuals ** 2))


# ===================================================================
#  High-field contour topology proxy
# ===================================================================

def _longest_false_run_periodic(active):
    """Return the longest inactive run length in a periodic boolean vector."""
    active = np.asarray(active, dtype=bool)
    n = active.size
    if n == 0 or np.all(active):
        return 0
    if not np.any(active):
        return n
    doubled = np.concatenate([~active, ~active])
    best = cur = 0
    for val in doubled:
        cur = cur + 1 if val else 0
        best = max(best, min(cur, n))
    return best


def compute_highB_contour_residual(
    vmec, snorms, thresholds=None, mpol=20, ntor=20,
    ntheta=96, nphi=96, target_phi_coverage=0.85,
    max_phi_gap=0.20,
):
    """
    Proxy for high-|B| contour topology in Boozer coordinates.

    QI configurations of interest should not leave high-field contours as
    isolated poloidal islands.  This diagnostic thresholds normalised |B| on
    each surface and penalises three failure modes:

      * high-field set does not cover most toroidal columns;
      * high-field coverage is more poloidal than toroidal;
      * high-field set splits into several disconnected islands.

    This is a topology/geometry proxy, not a substitute for the full QI
    residual.  It is intended to focus the optimiser on the high-B region
    that the averaged bounce-integral residual can miss.
    """
    vmec.run()
    snorms = np.atleast_1d(snorms).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.75, 0.98, 6)
    thresholds = np.atleast_1d(thresholds).astype(float)

    _, surface_data = run_boozer(vmec, snorms, mpol=mpol, ntor=ntor)
    residuals = []

    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    for data in surface_data:
        zeta = np.linspace(0.0, 2.0 * np.pi / data["nfp"], int(nphi), endpoint=False)
        TH, ZE = np.meshgrid(theta, zeta, indexing="ij")
        B = reconstruct_B(data["m"], data["n"], data["bmnc"], TH, ZE)
        bmin = float(np.nanmin(B))
        bmax = float(np.nanmax(B))
        if not np.isfinite(bmin + bmax) or bmax <= bmin:
            residuals.extend([1.0e3] * (4 * len(thresholds)))
            continue
        Bn = (B - bmin) / (bmax - bmin)

        for thr in thresholds:
            mask = Bn >= thr
            if not np.any(mask):
                residuals.extend([1.0, 1.0, 1.0, 1.0])
                continue

            phi_active = np.any(mask, axis=0)
            theta_active = np.any(mask, axis=1)
            phi_cov = float(np.mean(phi_active))
            theta_cov = float(np.mean(theta_active))
            phi_gap = _longest_false_run_periodic(phi_active) / max(phi_active.size, 1)

            labels, ncomp = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
            if ncomp > 0:
                sizes = np.bincount(labels.ravel())[1:]
                island_fraction = 1.0 - float(np.max(sizes)) / max(float(np.sum(sizes)), 1.0)
            else:
                island_fraction = 1.0

            residuals.append(max(target_phi_coverage - phi_cov, 0.0))
            residuals.append(max(theta_cov - phi_cov, 0.0))
            residuals.append(max(phi_gap - max_phi_gap, 0.0))
            residuals.append(island_fraction)

    residuals = np.asarray(residuals, dtype=float)
    return residuals / np.sqrt(max(residuals.size, 1))


class HighBContourResidual(Optimizable):
    """simsopt wrapper for the high-field Boozer contour topology proxy."""

    def __init__(self, vmec, snorms, thresholds=None, mpol=20, ntor=20,
                 ntheta=96, nphi=96, target_phi_coverage=0.85,
                 max_phi_gap=0.20):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.snorms = np.atleast_1d(snorms).tolist()
        self.thresholds = thresholds
        self.mpol = mpol
        self.ntor = ntor
        self.ntheta = ntheta
        self.nphi = nphi
        self.target_phi_coverage = target_phi_coverage
        self.max_phi_gap = max_phi_gap
        self._cache_x = None
        self._residuals = None
        self._warned_failure = False

    def _fallback_residuals(self):
        if self._residuals is not None:
            return np.full_like(self._residuals, 1e3, dtype=float)
        n_thr = len(np.atleast_1d(self.thresholds if self.thresholds is not None else np.linspace(0.75, 0.98, 6)))
        return np.full(max(len(self.snorms), 1) * n_thr * 4, 1e3, dtype=float)

    def _compute(self):
        try:
            cx = tuple(self.vmec.x)
        except Exception:
            cx = None
        if cx is not None and cx == self._cache_x:
            return
        try:
            self._residuals = compute_highB_contour_residual(
                self.vmec, self.snorms, thresholds=self.thresholds,
                mpol=self.mpol, ntor=self.ntor,
                ntheta=self.ntheta, nphi=self.nphi,
                target_phi_coverage=self.target_phi_coverage,
                max_phi_gap=self.max_phi_gap,
            )
        except Exception as exc:
            if not self._warned_failure:
                print(f"[HighBContourResidual] evaluation failed; using large residual penalty: {exc}")
                self._warned_failure = True
            self._residuals = self._fallback_residuals()
        self._cache_x = cx

    def residuals(self):
        self._compute()
        return self._residuals

    def total(self):
        self._compute()
        return float(np.sum(self._residuals ** 2))
