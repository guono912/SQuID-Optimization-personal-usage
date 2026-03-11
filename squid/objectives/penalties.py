"""
Penalty functions for stellarator optimisation constraints.

Consolidated from:
  - Targets.py  (MirrorRatioPen, AspectRatioPen, BetaPen, IotaPen,
                  MaxElongationPen)
  - simsopt_optimize_v3.py  (hinge_loss, compute_mirror_penalty,
                              compute_iota_penalty)

All penalties are Optimizable subclasses with residuals()/total()
for use with simsopt's LeastSquaresProblem.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe
from simsopt._core import Optimizable


# ===================================================================
#  Utility
# ===================================================================

def hinge_loss(x, x_target):
    """M_t(x, x_t) = max(0, x - x_t)^2  — one-sided quadratic penalty."""
    return max(0.0, x - x_target) ** 2


# ===================================================================
#  Mirror ratio
# ===================================================================

class MirrorRatioPenalty(Optimizable):
    """
    Penalise when the mirror ratio drops BELOW *target*.

    f_Delta = max(0, target - Delta)^2
    where Delta = (Bmax - Bmin) / (Bmax + Bmin).
    """

    def __init__(self, vmec, target=0.20):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.target = target

    def _delta(self):
        self.vmec.run()
        w = self.vmec.wout
        xm_nyq, xn_nyq = w.xm_nyq, w.xn_nyq
        bmnc = w.bmnc.T
        nfp = w.nfp
        N = 100
        thetas = np.linspace(0, 2 * np.pi, N)
        phis = np.linspace(0, 2 * np.pi / nfp, N)
        phis2D, thetas2D = np.meshgrid(phis, thetas)
        b = np.zeros((N, N))
        for i in range(len(xn_nyq)):
            angle = xm_nyq[i] * thetas2D - xn_nyq[i] * phis2D
            b += bmnc[1, i] * np.cos(angle)
        Bmax, Bmin = np.max(b), np.min(b)
        return (Bmax - Bmin) / (Bmax + Bmin) if (Bmax + Bmin) > 1e-30 else 0.0

    def residuals(self):
        delta = self._delta()
        return np.array([np.sqrt(hinge_loss(self.target - delta, 0.0))])

    def total(self):
        delta = self._delta()
        return hinge_loss(self.target - delta, 0.0)


# ===================================================================
#  Aspect ratio
# ===================================================================

class AspectRatioPenalty(Optimizable):
    """Penalise when aspect ratio exceeds *target*."""

    def __init__(self, vmec, target=10.0):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.target = target

    def residuals(self):
        self.vmec.run()
        asp = self.vmec.wout.aspect
        return np.array([max(0.0, asp - self.target)])

    def total(self):
        return float(self.residuals()[0] ** 2)


# ===================================================================
#  Beta
# ===================================================================

class BetaPenalty(Optimizable):
    """Penalise when plasma beta exceeds *target*."""

    def __init__(self, vmec, target=0.02):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.target = target

    def residuals(self):
        self.vmec.run()
        beta = self.vmec.wout.betatotal
        return np.array([max(0.0, beta - self.target)])

    def total(self):
        return float(self.residuals()[0] ** 2)


# ===================================================================
#  Iota (rotational transform)
# ===================================================================

class IotaPenalty(Optimizable):
    """
    Penalise rotational transform outside [tmin, tmax].

    f_iota = max(0, iota_max - tmax) + max(0, tmin - iota_min)
    """

    def __init__(self, vmec, tmax=0.62, tmin=0.62):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.tmax = tmax
        self.tmin = tmin

    def residuals(self):
        self.vmec.run()
        iotaf = np.abs(self.vmec.wout.iotaf)
        imax, imin = np.max(iotaf), np.min(iotaf)
        out = 0.0
        if imax > self.tmax:
            out += imax - self.tmax
        if imin < self.tmin:
            out += self.tmin - imin
        return np.array([out])

    def total(self):
        return float(self.residuals()[0] ** 2)


class IotaProfilePenalty(Optimizable):
    """
    Soft quadratic penalty on axis and edge iota.

    f_iota = (iota_axis - target_axis)^2 + (iota_edge - target_edge)^2
    """

    def __init__(self, vmec, target_axis=None, target_edge=None):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.target_axis = target_axis
        self.target_edge = target_edge
        self._auto_detected = False

    def _auto_detect(self):
        if self._auto_detected:
            return
        self.vmec.run()
        iotaf = self.vmec.wout.iotaf
        if self.target_axis is None:
            self.target_axis = round(float(iotaf[0]), 3)
        if self.target_edge is None:
            self.target_edge = round(float(iotaf[-1]), 3)
        self._auto_detected = True

    def residuals(self):
        self._auto_detect()
        self.vmec.run()
        iotaf = self.vmec.wout.iotaf
        iota_ax = float(iotaf[0])
        iota_ed = float(iotaf[-1])
        r1 = iota_ax - self.target_axis
        r2 = iota_ed - self.target_edge
        return np.array([r1, r2])

    def total(self):
        r = self.residuals()
        return float(np.sum(r ** 2))


# ===================================================================
#  Maximum elongation
# ===================================================================

class MaxElongationPenalty(Optimizable):
    """
    Penalise maximum cross-section elongation exceeding *target*.

    Uses elliptic-integral fitting (from Targets.py:MaxElongationPen).
    """

    def __init__(self, vmec, target=6.0, ntheta=50, nphi=50):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.target = target
        self.ntheta = ntheta
        self.nphi = nphi

    def _max_elongation(self):
        self.vmec.run()
        w = self.vmec.wout
        nfp = w.nfp
        xm = w.xm
        xn = w.xn
        rmnc = w.rmnc.T
        zmns = w.zmns.T
        lasym = w.lasym
        raxis_cc = w.raxis_cc
        zaxis_cs = w.zaxis_cs

        if lasym:
            raxis_cs = w.raxis_cs
            zaxis_cc = w.zaxis_cc
        else:
            raxis_cs = 0 * raxis_cc
            zaxis_cc = 0 * zaxis_cs

        theta1D = np.linspace(0, 2 * np.pi, num=self.ntheta)
        phi1D = np.linspace(0, 2 * np.pi / nfp, num=self.nphi)

        def FindBoundary(theta, phi):
            R = 0.0
            Z = 0.0
            for i in range(len(xm)):
                angle = xm[i] * theta - xn[i] * phi
                R += rmnc[-1, i] * np.cos(angle)
                Z += zmns[-1, i] * np.sin(angle)
            X = R * np.cos(phi)
            Y = R * np.sin(phi)
            return np.array([X, Y, Z]).flatten()

        # Axis geometry
        Rax = np.zeros(self.nphi)
        Zax = np.zeros(self.nphi)
        Raxp = np.zeros(self.nphi)
        Zaxp = np.zeros(self.nphi)

        for jn in range(len(raxis_cc)):
            n = jn
            cosangle = np.cos(n * nfp * phi1D)
            sinangle = np.sin(n * nfp * phi1D)
            Rax += raxis_cc[jn] * cosangle
            Zax += zaxis_cs[jn] * sinangle
            Raxp += raxis_cc[jn] * (-n * nfp * sinangle)
            Zaxp += zaxis_cs[jn] * (n * nfp * cosangle)
            Rax += raxis_cs[jn] * sinangle
            Zax += zaxis_cc[jn] * cosangle
            Raxp += raxis_cs[jn] * (n * nfp * cosangle)
            Zaxp += zaxis_cc[jn] * (-n * nfp * sinangle)

        Xax = Rax * np.cos(phi1D)
        Yax = Rax * np.sin(phi1D)

        d_l_d_phi = np.sqrt(Rax ** 2 + Raxp ** 2 + Zaxp ** 2)
        d_r_d_phi_cyl = np.array([Raxp, Rax, Zaxp]).T
        tangent_cyl = np.zeros((self.nphi, 3))
        for j in range(3):
            tangent_cyl[:, j] = d_r_d_phi_cyl[:, j] / d_l_d_phi

        tangent_R = tangent_cyl[:, 0]
        tangent_phi = tangent_cyl[:, 1]
        tangent_Z = tangent_cyl[:, 2]
        tangent_X = tangent_R * np.cos(phi1D) - tangent_phi * np.sin(phi1D)
        tangent_Y = tangent_R * np.sin(phi1D) + tangent_phi * np.cos(phi1D)

        Xp = np.zeros(self.ntheta)
        Yp = np.zeros(self.ntheta)
        Zp = np.zeros(self.ntheta)
        elongs = np.zeros(self.nphi)
        a1_prev = 1.0

        for iphi in range(self.nphi):
            phi_val = phi1D[iphi]
            t_ = np.array([tangent_X[iphi], tangent_Y[iphi], tangent_Z[iphi]])
            pax = np.array([Xax[iphi], Yax[iphi], Zax[iphi]])

            for ipt in range(self.ntheta):
                theta = theta1D[ipt]
                fdot = lambda p: np.dot(t_, (FindBoundary(theta, p) - pax))
                phi_x = fsolve(fdot, phi_val, full_output=False)
                sbound = FindBoundary(theta, phi_x)
                sbound -= np.dot(sbound, t_) * t_
                Xp[ipt] = sbound[0]
                Yp[ipt] = sbound[1]
                Zp[ipt] = sbound[2]

            perim = np.sum(np.sqrt(
                (Xp - np.roll(Xp, 1)) ** 2
                + (Yp - np.roll(Yp, 1)) ** 2
                + (Zp - np.roll(Zp, 1)) ** 2
            ))
            A_cross = _find_area(Xp, Yp, Zp)

            perim_resid = lambda a: perim - (4 * a * ellipe(1 - (A_cross / (np.pi * a ** 2)) ** 2))
            a1 = fsolve(perim_resid, a1_prev, full_output=False)
            a1_prev = float(a1)
            a2 = A_cross / (np.pi * a1_prev)
            maj = max(a1_prev, a2)
            mn = min(a1_prev, a2)
            elongs[iphi] = maj / mn if mn > 1e-15 else 1.0

        return float(np.max(elongs))

    def residuals(self):
        e = self._max_elongation()
        return np.array([max(0.0, e - self.target)])

    def total(self):
        return float(self.residuals()[0] ** 2)


# ===================================================================
#  Tikhonov regularisation
# ===================================================================

class TikhonovRegularization(Optimizable):
    """
    f_reg = ||x - x_ref||^2  — L2 distance from initial DOFs.
    """

    def __init__(self, vmec, x_ref=None):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        if x_ref is None:
            self.vmec.run()
            self._x_ref = np.array(vmec.x, dtype=float).copy()
        else:
            self._x_ref = np.array(x_ref, dtype=float).copy()

    def residuals(self):
        x_cur = np.array(self.vmec.x, dtype=float)
        return x_cur - self._x_ref

    def total(self):
        return float(np.sum(self.residuals() ** 2))


# ===================================================================
#  Geometry helpers (ported from Helpers.py)
# ===================================================================

def _find_unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]], [1, b[1], b[2]], [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]], [b[0], 1, b[2]], [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (x / mag, y / mag, z / mag)


def _find_area(X, Y, Z):
    total = np.zeros(3)
    for i in range(len(X)):
        vi1 = np.array([X[i], Y[i], Z[i]])
        vi2 = np.array([X[(i + 1) % len(X)], Y[(i + 1) % len(Y)], Z[(i + 1) % len(Z)]])
        total += np.cross(vi1, vi2)
    pt0 = [X[0], Y[0], Z[0]]
    pt1 = [X[1], Y[1], Z[1]]
    pt2 = [X[2], Y[2], Z[2]]
    return abs(np.dot(total, _find_unit_normal(pt0, pt1, pt2)) / 2)
