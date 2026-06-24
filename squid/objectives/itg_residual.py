"""
ITG turbulence target (f_nabla_s).

Dual-method implementation:
  - "drift_curvature": uses simsopt.mhd.vmec_diagnostics.vmec_fieldlines
    for the full
    curvature drift B x kappa . grad alpha  (physically correct)
  - "vacuum_dBds": uses dB/ds < 0 as a vacuum proxy
    (from simsopt_optimize_v3.py:compute_f_grad_s)

BUG FIX (B1): The original ITG_optimization.py had
    np.heaviside(-bad_curvature, 0)
where bad_curvature was a boolean array, resulting in xi = 0 always.
Fixed here: we use bad_curvature.astype(float) directly.
"""

import numpy as np
from simsopt._core import Optimizable
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines


class ITGResidual(Optimizable):
    """
    ITG turbulence residual (f_nabla_s) as a simsopt Optimizable.

    Computes xi = (a_min * Theta(bad_curv) * |nabla s|)^2 on each
    target surface, then integrates xi over the sampled surface.
    """

    def __init__(self, vmec, snorms, method="drift_curvature",
                 nphi=201, nalpha=50):
        """
        Parameters
        ----------
        vmec : simsopt Vmec object
        snorms : array-like of normalised flux surfaces
        method : "drift_curvature" or "vacuum_dBds"
        nphi, nalpha : grid resolution
        """
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.snorms = np.atleast_1d(snorms).astype(float)
        self.method = method
        self.nphi = nphi
        self.nalpha = nalpha
        self._cache_x = None
        self._residuals_1d = None
        self._total = 0.0

    def _compute(self):
        try:
            cx = tuple(self.vmec.x)
        except Exception:
            cx = None
        if cx is not None and cx == self._cache_x:
            return
        self.vmec.run()

        if self.method == "drift_curvature":
            self._compute_drift_curvature()
        else:
            self._compute_vacuum_dBds()
        self._cache_x = cx

    # ---- method 1: full drift curvature (from ITG_optimization.py, bug-fixed) ----

    def _compute_drift_curvature(self):
        vmec = self.vmec
        nfp = vmec.wout.nfp
        a_min = vmec.wout.Aminor_p

        alpha = np.linspace(0, 2 * np.pi, self.nalpha)
        theta = np.linspace(0, 2 * np.pi, self.nphi)

        residuals = []
        for s in self.snorms:
            try:
                data = vmec_fieldlines(vmec, s, alpha, theta1d=theta)
            except Exception:
                residuals.append(np.nan)
                continue

            edge_toroidal_flux = vmec.wout.phipf[-1]
            kappa_tilde = data.B_cross_kappa_dot_grad_alpha * np.sign(edge_toroidal_flux)
            bad_curvature = (kappa_tilde < 0).astype(float)  # BUG B1 FIX

            grad_S = np.sqrt(data.grad_s_dot_grad_s)
            xi = (a_min * bad_curvature * grad_S) ** 2

            if not np.any(xi > 0):
                residuals.append(0.0)
                continue

            dtheta = 2 * np.pi / self.nalpha
            dphi = 2 * np.pi / (nfp * self.nphi)
            f_VS = np.sum(xi) * dtheta * dphi
            residuals.append(float(f_VS))

        self._residuals_1d = np.nan_to_num(
            np.array(residuals, dtype=float), nan=1.0e3, posinf=1.0e3,
            neginf=1.0e3)
        self._total = float(np.sum(self._residuals_1d))

    # ---- method 2: vacuum dB/ds proxy (from simsopt_optimize_v3.py) ----

    def _compute_vacuum_dBds(self):
        vmec = self.vmec
        wout = vmec.wout
        ns = int(wout.ns)
        nfp = int(wout.nfp)

        xm = np.ravel(np.array(wout.xm, dtype=int))
        xn = np.ravel(np.array(wout.xn, dtype=int))
        xm_nyq = np.ravel(np.array(wout.xm_nyq, dtype=int))
        xn_nyq = np.ravel(np.array(wout.xn_nyq, dtype=int))

        rmnc = np.array(wout.rmnc)
        zmns = np.array(wout.zmns)
        bmnc = np.array(wout.bmnc)
        gmnc = np.array(wout.gmnc)

        a_min = float(wout.Aminor_p)
        nu, nv = self.nphi, self.nalpha

        theta = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / nfp, nv, endpoint=False)
        th, ze = np.meshgrid(theta, zeta, indexing="ij")

        m_3d_geom = xm[:, None, None]
        n_3d_geom = xn[:, None, None]
        angles_geom = m_3d_geom * th[None, :, :] - n_3d_geom * ze[None, :, :]
        cos_a_geom = np.cos(angles_geom)
        sin_a_geom = np.sin(angles_geom)

        m_3d_nyq = xm_nyq[:, None, None]
        n_3d_nyq = xn_nyq[:, None, None]
        angles_nyq = m_3d_nyq * th[None, :, :] - n_3d_nyq * ze[None, :, :]
        cos_a_nyq = np.cos(angles_nyq)

        s_grid = np.linspace(0, 1, ns)

        def _recon_geom_c(fmnc, k):
            return np.sum(fmnc[:, k][:, None, None] * cos_a_geom, axis=0)

        def _recon_nyq_c(fmnc, k):
            return np.sum(fmnc[:, k][:, None, None] * cos_a_nyq, axis=0)

        f_total = 0.0
        residuals = []
        for s_t in self.snorms:
            js = int(np.argmin(np.abs(s_grid - s_t)))
            js = max(1, min(js, ns - 2))
            ds = s_grid[js + 1] - s_grid[js - 1]

            R = _recon_geom_c(rmnc, js)
            R_u = np.sum(-m_3d_geom * rmnc[:, js][:, None, None] * sin_a_geom, axis=0)
            R_v = np.sum(n_3d_geom * rmnc[:, js][:, None, None] * sin_a_geom, axis=0)
            Z_u = np.sum(m_3d_geom * zmns[:, js][:, None, None] * cos_a_geom, axis=0)
            Z_v = np.sum(-n_3d_geom * zmns[:, js][:, None, None] * cos_a_geom, axis=0)

            sqrtg = _recon_nyq_c(gmnc, js)
            sqrtg = np.where(np.abs(sqrtg) < 1e-30, 1e-30, sqrtg)

            cross_sq = R ** 2 * (R_u ** 2 + Z_u ** 2) + (Z_u * R_v - R_u * Z_v) ** 2
            grad_s = np.sqrt(np.maximum(cross_sq / sqrtg ** 2, 1e-30))

            dBds = (_recon_nyq_c(bmnc, js + 1) - _recon_nyq_c(bmnc, js - 1)) / ds
            bad_mask = np.where(dBds < 0, 1.0, 0.0)

            xi = (a_min * bad_mask * grad_s) ** 2
            if not np.any(xi > 0):
                residuals.append(0.0)
                continue

            dth = 2 * np.pi / nu
            dze = 2 * np.pi / (nfp * nv)
            f_val = float(np.sum(xi)) * dth * dze
            residuals.append(f_val)
            f_total += f_val

        self._residuals_1d = np.array(residuals)
        self._total = f_total

    # ---- simsopt interface ----

    def residuals(self):
        self._compute()
        return self._residuals_1d

    def total(self):
        self._compute()
        return self._total
