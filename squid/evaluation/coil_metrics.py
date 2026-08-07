"""Shared coil-feasibility geometry helpers for the external coil gates.

Owns the winding-surface construction and curve metrics used by both
``coil_feasibility_gate`` (current-potential scan) and
``coil_contour_metrics`` (true coil contour review). ``current_spectrum_metrics``
is the same helper used by the inner-loop coil proxy and is imported from
``squid/objectives/coil_proxy.py`` rather than duplicated.
"""

import numpy as np

from ..objectives.coil_proxy import current_spectrum_metrics  # noqa: F401  (re-export)


def _float(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else float("nan")


def make_scaled_winding_surface(eq, offset_fraction):
    """Approximate a winding surface by scaling non-R00 Fourier coefficients."""
    plasma = eq.surface
    R_lmn = np.array(plasma.R_lmn, dtype=float).copy()
    Z_lmn = np.array(plasma.Z_lmn, dtype=float).copy()
    modes_R = np.array(plasma.R_basis.modes, dtype=int)
    modes_Z = np.array(plasma.Z_basis.modes, dtype=int)

    scale = 1.0 + float(offset_fraction)
    r00 = np.where((modes_R[:, 1] == 0) & (modes_R[:, 2] == 0))[0]
    for i in range(R_lmn.size):
        if not (r00.size and i == r00[0]):
            R_lmn[i] *= scale
    Z_lmn *= scale

    from desc.geometry import FourierRZToroidalSurface

    return FourierRZToroidalSurface(
        R_lmn=R_lmn,
        Z_lmn=Z_lmn,
        modes_R=modes_R[:, 1:],
        modes_Z=modes_Z[:, 1:],
        NFP=eq.NFP,
        sym=eq.sym,
    )


def make_winding_surface(eq, offset_fraction, method="normal", M=None, N=None):
    """Build a winding surface from the plasma boundary.

    The preferred method is a true constant-normal offset. The scaled method is
    kept as a fallback for debugging because it is cheap but can locally approach
    or intersect the plasma boundary.
    """
    if method == "scaled":
        return make_scaled_winding_surface(eq, offset_fraction)
    if method != "normal":
        raise ValueError(f"unknown winding surface method: {method}")
    aminor = _float(eq.compute("a")["a"])
    offset_m = float(offset_fraction) * aminor
    return eq.surface.constant_offset_surface(
        offset_m,
        M=M or max(eq.surface.M, eq.M),
        N=N or max(eq.surface.N, eq.N),
    )


def curve_metrics_xyz(x):
    """Length / curvature statistics of a closed curve given XYZ points."""
    x = np.asarray(x, dtype=float)
    if np.linalg.norm(x[0] - x[-1]) > 1e-10:
        x = np.vstack([x, x[0]])
    ds_vec = np.diff(x, axis=0)
    ds = np.linalg.norm(ds_vec, axis=1)
    length = float(np.sum(ds))
    # finite-difference curvature on closed curve
    xp = np.roll(x[:-1], -1, axis=0)
    xm = np.roll(x[:-1], 1, axis=0)
    xc = x[:-1]
    d1 = 0.5 * (xp - xm)
    d2 = xp - 2 * xc + xm
    denom = np.linalg.norm(d1, axis=1) ** 3
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(denom, 1e-300)
    return {
        "length": length,
        "curvature_mean": float(np.mean(kappa)),
        "curvature_rms": float(np.sqrt(np.mean(kappa**2))),
        "curvature_max": float(np.max(kappa)),
        "curvature_p95": float(np.percentile(kappa, 95)),
    }
