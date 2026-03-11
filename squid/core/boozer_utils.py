"""
Boozer coordinate utilities.

Merged from:
  - max_J_evaluation.py  (get_boozer_data, _reconstruct_B)
  - Targets.py / QuasiIsodynamicResidual2  (Boozer extraction)
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from simsopt.mhd import Boozer


def run_boozer(vmec, surfaces, mpol=20, ntor=20):
    """
    Run the Boozer transform on *vmec* for the requested *surfaces*.

    Returns
    -------
    bx : booz_xform object  (low-level handle)
    surface_data : list[dict]
        One dict per surface with keys:
          m, n, bmnc, nfp, iota, s, B_min, B_max
    """
    vmec.run()
    boozer = Boozer(vmec, mpol=mpol, ntor=ntor)
    for s in surfaces:
        boozer.register(s)
    boozer.run()

    bx = boozer.bx
    nfp = int(bx.nfp)
    bmnc_all = np.array(bx.bmnc_b)
    m_arr = np.ravel(bx.xm_b)
    n_arr = np.ravel(bx.xn_b)

    vmec_iotaf = np.array(vmec.wout.iotaf)
    vmec_s_grid = np.linspace(0, 1, len(vmec_iotaf))

    surface_data = []
    for idx, s_val in enumerate(surfaces):
        bmnc = bmnc_all[:, idx] if bmnc_all.ndim == 2 else np.ravel(bmnc_all)
        iota = float(np.interp(s_val, vmec_s_grid, vmec_iotaf))

        th, ze = np.meshgrid(
            np.linspace(0, 2 * np.pi, 80),
            np.linspace(0, 2 * np.pi / nfp, 80),
            indexing="ij",
        )
        B_2d = reconstruct_B(m_arr, n_arr, bmnc, th, ze)
        surface_data.append(dict(
            m=m_arr, n=n_arr, bmnc=bmnc, nfp=nfp, iota=iota,
            s=s_val,
            B_min=float(np.min(B_2d)),
            B_max=float(np.max(B_2d)),
        ))
    return bx, surface_data


def reconstruct_B(m, n, bmnc, theta, zeta):
    """
    Reconstruct |B| from Boozer Fourier harmonics.

    B(theta, zeta) = sum_i bmnc[i] * cos(m[i]*theta - n[i]*zeta)
    """
    B = np.zeros_like(theta, dtype=float)
    for i in range(len(bmnc)):
        B += bmnc[i] * np.cos(m[i] * theta - n[i] * zeta)
    return B


def get_iota_profile(vmec):
    """Return (s_grid, iotaf) from VMEC's full radial grid."""
    vmec.run()
    iotaf = np.array(vmec.wout.iotaf)
    s_grid = np.linspace(0, 1, len(iotaf))
    return s_grid, iotaf


def get_iota(vmec, s):
    """Interpolated rotational transform at normalised flux *s*."""
    s_grid, iotaf = get_iota_profile(vmec)
    return float(np.interp(s, s_grid, iotaf))


def get_iota_half_grid(vmec, s):
    """Iota via half-grid spline (matches colleague's convention)."""
    vmec.run()
    return float(
        UnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:], k=1, s=0)(s)
    )
