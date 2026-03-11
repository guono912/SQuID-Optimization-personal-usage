"""
Maximum-J residual target (f_maxJ).

Unified implementation combining:
  - Optimizable subclass structure from squid_alternative/max_J_optimization.py
  - Correct 1/B_I Jacobian from max_J_evaluation.py
  - Full cross-alpha gradient and full B* range (PRX Energy Eqs. 4, 6, 7)

PRX Energy reference:
  Eq. (4):  J_C = integral sqrt(1 - B_C/B*) * dl/B_I
  Eq. (6):  partial_s J  (finite-difference, cross-alpha)
  Eq. (7):  f_maxJ = sum M_t(<partial_s J>_alpha, T_J)
"""

import numpy as np
from simsopt._core import Optimizable

from ..core.boozer_utils import run_boozer, reconstruct_B
from ..core.squash_stretch import squash_and_stretch_simple


class MaxJResidual(Optimizable):
    """
    Compute the maximum-J penalty f_maxJ as a simsopt Optimizable.

    Uses Boozer coordinates with the correct 1/B_I Jacobian factor
    and full cross-alpha gradient differencing.
    """

    def __init__(self, vmec, s_vals=None, num_alpha=8, num_pitch=50,
                 T_J=-0.06, mboz=8, nboz=8):
        super().__init__(depends_on=[vmec])
        self.vmec = vmec
        self.s_vals = s_vals if s_vals is not None else np.linspace(0.2, 0.8, 5)
        self.num_alpha = num_alpha
        self.num_pitch = num_pitch
        self.T_J = T_J
        self.mboz = mboz
        self.nboz = nboz
        self._cache_x = None
        self._f_maxJ = 0.0
        self._f_QI = 0.0
        self._mirror_ratio = 0.0
        self._iota_axis = 0.0
        self._iota_edge = 0.0
        self._B_min = 0.0
        self._B_max = 0.0

    def _compute(self):
        try:
            cx = tuple(self.vmec.x)
        except Exception:
            cx = None
        if cx is not None and cx == self._cache_x:
            return

        info = _evaluate_squid(
            self.vmec, self.s_vals,
            np.linspace(0, 2 * np.pi, self.num_alpha, endpoint=False),
            self.num_pitch, self.T_J, self.mboz, self.nboz,
        )
        self._f_maxJ = info["f_maxJ"]
        self._f_QI = info["f_QI"]
        self._mirror_ratio = info["mirror_ratio"]
        self._iota_axis = info["iota_axis"]
        self._iota_edge = info["iota_edge"]
        self._B_min = info["B_min"]
        self._B_max = info["B_max"]
        self._cache_x = cx

    def residuals(self):
        """1-D residual for LeastSquaresProblem."""
        self._compute()
        return np.array([np.sqrt(max(self._f_maxJ, 0.0))])

    def total(self):
        self._compute()
        return self._f_maxJ

    @property
    def f_QI(self):
        self._compute()
        return self._f_QI

    @property
    def mirror_ratio(self):
        self._compute()
        return self._mirror_ratio

    @property
    def iota_axis(self):
        self._compute()
        return self._iota_axis

    @property
    def iota_edge(self):
        self._compute()
        return self._iota_edge


# ------------------------------------------------------------------ #
#  Core evaluation (from simsopt_optimize_v3.py:evaluate_squid)
# ------------------------------------------------------------------ #

def _extract_field_line(data, alpha, npts=500):
    """Sample |B| along one field line in Boozer coordinates."""
    nfp = data["nfp"]
    iota = data["iota"]
    zeta = np.linspace(0, 2 * np.pi / nfp, npts)
    theta = iota * zeta + alpha
    B_I = reconstruct_B(data["m"], data["n"], data["bmnc"], theta, zeta)
    return zeta, B_I


def _compute_J_C(zeta, B_I, B_C, B_stars):
    """
    Bounce integral for constructed field (Eq. 4).

    J_C(lambda) = integral sqrt(1 - lambda*B_C) / B_I  dzeta

    Includes the 1/B_I Jacobian factor for the Boozer line element.
    """
    idx_min = int(np.argmin(B_C))
    left_z, left_B = zeta[:idx_min + 1], B_C[:idx_min + 1]
    right_z, right_B = zeta[idx_min:], B_C[idx_min:]

    J_vals = np.zeros(len(B_stars))
    for k, B_star in enumerate(B_stars):
        lam = 1.0 / B_star
        if B_star <= np.min(B_C) or B_star >= np.max(B_C):
            continue
        z1 = float(np.interp(B_star, left_B[::-1], left_z[::-1]))
        z2 = float(np.interp(B_star, right_B, right_z))
        mask = (zeta >= z1) & (zeta <= z2)
        if np.sum(mask) < 2:
            continue
        z_seg = zeta[mask]
        bc_seg = B_C[mask]
        bi_seg = B_I[mask]
        integrand = np.sqrt(np.maximum(1.0 - lam * bc_seg, 0.0)) / np.maximum(bi_seg, 1e-30)
        J_vals[k] = float(np.trapezoid(integrand, z_seg))
    return J_vals


def _compute_J_I(zeta, B_I, B_C, B_stars):
    """
    Bounce integral for actual field (Eq. 3).

    J_I(lambda) = integral sign(1-lambda*B_I) sqrt(|1-lambda*B_I|) / B_I  dzeta
    """
    idx_min = int(np.argmin(B_C))
    left_z, left_B = zeta[:idx_min + 1], B_C[:idx_min + 1]
    right_z, right_B = zeta[idx_min:], B_C[idx_min:]

    J_vals = np.zeros(len(B_stars))
    for k, B_star in enumerate(B_stars):
        lam = 1.0 / B_star
        if B_star <= np.min(B_C) or B_star >= np.max(B_C):
            continue
        z1 = float(np.interp(B_star, left_B[::-1], left_z[::-1]))
        z2 = float(np.interp(B_star, right_B, right_z))
        mask = (zeta >= z1) & (zeta <= z2)
        if np.sum(mask) < 2:
            continue
        z_seg = zeta[mask]
        bi_seg = B_I[mask]
        term = 1.0 - lam * bi_seg
        integrand = np.sign(term) * np.sqrt(np.abs(term)) / np.maximum(bi_seg, 1e-30)
        J_vals[k] = float(np.trapezoid(integrand, z_seg))
    return J_vals


def _evaluate_squid(vmec_ro, s_vals, alphas, num_pitch, T_J, mboz, nboz):
    """
    Compute f_maxJ, f_QI, mirror ratio, and iota profile.

    Uses the correct 1/B_I Jacobian and full cross-alpha gradient
    differencing from max_J_evaluation.py.
    """
    _, surface_data = run_boozer(vmec_ro, s_vals, mpol=mboz, ntor=nboz)

    ns = len(s_vals)
    na = len(alphas)
    np_ = num_pitch

    vmec_iotaf = np.array(vmec_ro.wout.iotaf)
    iota_axis = float(vmec_iotaf[0])
    iota_edge = float(vmec_iotaf[-1])

    all_bs, all_jc, all_ji = [], [], []
    global_Bmin, global_Bmax = 1e30, -1e30

    for k in range(ns):
        data = surface_data[k]
        global_Bmin = min(global_Bmin, data["B_min"])
        global_Bmax = max(global_Bmax, data["B_max"])
        B_stars = np.linspace(data["B_min"], data["B_max"], np_ + 2)[1:-1]
        jc_rows, ji_rows = [], []
        for alpha in alphas:
            zeta, B_I = _extract_field_line(data, alpha)
            B_C = squash_and_stretch_simple(zeta, B_I, data["B_min"], data["B_max"])
            jc_rows.append(_compute_J_C(zeta, B_I, B_C, B_stars))
            ji_rows.append(_compute_J_I(zeta, B_I, B_C, B_stars))
        all_bs.append(B_stars)
        all_jc.append(np.array(jc_rows).T)
        all_ji.append(np.array(ji_rows).T)

    mirror_ratio = (
        (global_Bmax - global_Bmin) / (global_Bmax + global_Bmin)
        if (global_Bmax + global_Bmin) > 1e-30 else 0.0
    )

    b_lo = max(b[0] for b in all_bs)
    b_hi = min(b[-1] for b in all_bs)
    if b_lo >= b_hi:
        return dict(f_maxJ=1e6, f_QI=1e6, mirror_ratio=mirror_ratio,
                    iota_axis=iota_axis, iota_edge=iota_edge,
                    B_min=global_Bmin, B_max=global_Bmax)

    common_B = np.linspace(b_lo, b_hi, np_)
    JC_int, JI_int = [], []
    for k in range(ns):
        jc_k = np.zeros((np_, na))
        ji_k = np.zeros((np_, na))
        for a in range(na):
            jc_k[:, a] = np.interp(common_B, all_bs[k], all_jc[k][:, a])
            ji_k[:, a] = np.interp(common_B, all_bs[k], all_ji[k][:, a])
        JC_int.append(jc_k)
        JI_int.append(ji_k)

    # f_maxJ (Eqs. 6-7)
    f_maxJ = 0.0
    for k in range(ns - 1):
        ds = s_vals[k + 1] - s_vals[k]
        J_lo, J_hi = JC_int[k], JC_int[k + 1]
        denom = float(np.mean(J_hi + J_lo))
        if abs(denom) < 1e-30:
            denom = 1e-30
        diff = J_hi[:, :, None] - J_lo[:, None, :]
        mean_dJ = (diff / (ds * denom)).mean(axis=-1)
        penalty = np.maximum(0.0, mean_dJ - T_J) ** 2
        f_maxJ += float(penalty.sum())

    # f_QI (Eq. 5)
    f_QI = 0.0
    for k in range(ns):
        denom = float(np.mean(JI_int[k] + JC_int[k]))
        if abs(denom) < 1e-30:
            denom = 1e-30
        diff = JI_int[k][:, :, None] - JC_int[k][:, None, :]
        f_QI += float(np.sum((diff / denom) ** 2))

    return dict(
        f_maxJ=f_maxJ, f_QI=f_QI, mirror_ratio=mirror_ratio,
        iota_axis=iota_axis, iota_edge=iota_edge,
        B_min=global_Bmin, B_max=global_Bmax,
    )
