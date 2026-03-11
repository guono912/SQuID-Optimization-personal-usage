"""
Field-line tracing in Boozer and VMEC-pest coordinates.

Merged from:
  - Helpers.py  (TracedFieldline — pest-coordinate tracer)
  - max_J_evaluation.py  (extract_field_line — Boozer-coordinate tracer)
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from ..core.boozer_utils import reconstruct_B


# ------------------------------------------------------------------ #
#  Boozer-coordinate field-line extraction (from max_J_evaluation.py)
# ------------------------------------------------------------------ #

def extract_field_line_boozer(surface_data, alpha=0.0, npts=500):
    """
    Sample |B| along one field line in Boozer coordinates over one
    field period.

    Parameters
    ----------
    surface_data : dict
        Output of run_boozer() for a single surface.
    alpha : float
        Field-line label.
    npts : int
        Number of sampling points along zeta.

    Returns
    -------
    zeta : 1-D array   Boozer zeta on [0, 2*pi/nfp]
    B_I  : 1-D array   |B| along the field line
    """
    nfp = surface_data["nfp"]
    iota = surface_data["iota"]

    zeta = np.linspace(0, 2 * np.pi / nfp, npts)
    theta = iota * zeta + alpha
    B_I = reconstruct_B(
        surface_data["m"], surface_data["n"], surface_data["bmnc"],
        theta, zeta,
    )
    return zeta, B_I


# ------------------------------------------------------------------ #
#  VMEC-pest coordinate field-line tracer (from Helpers.py)
# ------------------------------------------------------------------ #

def _fzero_residuals(theta_vmec_trys, theta_pest_targets, phis,
                     xm, xn, lmns, lmnc, lasym,
                     wh, ih):
    """Residual function for theta_pest -> theta_vmec conversion."""
    phis1 = np.broadcast_to(
        phis[:, None], (theta_vmec_trys.shape[0], theta_vmec_trys.shape[1])
    ).copy()

    fzero = theta_vmec_trys - theta_pest_targets

    nmodes = len(xm)
    angle = (
        xm[None, None, :] * theta_vmec_trys[:, :, None]
        - xn[None, None, :] * phis1[:, :, None]
    )
    sinangle = np.sin(angle)

    for k in range(2):
        fzero += np.sum(wh[k] * lmns[ih[k], :][None, None, :] * sinangle, axis=2)

    return fzero


def _get_roots(a0, b0, theta_pest_targets, phis, xm, xn, lmns, lmnc,
               lasym, wh, ih):
    """Vectorised Brent root-finding (ported from Helpers.py:get_roots)."""
    a = a0.copy()
    b = b0.copy()
    roots = np.zeros_like(a)
    toteval = a.size
    tol = 1e-10

    fa = _fzero_residuals(a, theta_pest_targets, phis, xm, xn, lmns, lmnc, lasym, wh, ih)
    fb = _fzero_residuals(b, theta_pest_targets, phis, xm, xn, lmns, lmnc, lasym, wh, ih)

    for _ in range(10):
        bad = ((fa > 0) & (fb > 0)) | ((fa < 0) & (fb < 0))
        if not np.any(bad):
            break
        a[bad] -= 0.3
        b[bad] += 0.3
        fa = _fzero_residuals(a, theta_pest_targets, phis, xm, xn, lmns, lmnc, lasym, wh, ih)
        fb = _fzero_residuals(b, theta_pest_targets, phis, xm, xn, lmns, lmnc, lasym, wh, ih)

    c = b.copy()
    fc = fb.copy()
    d = np.zeros_like(b)
    e = np.zeros_like(b)
    eps = np.finfo(float).eps

    for _ in range(100):
        bad_c = ((fb > 0) & (fc > 0)) | ((fb < 0) & (fc < 0))
        c[bad_c] = a[bad_c]
        fc[bad_c] = fa[bad_c]
        d[bad_c] = b[bad_c] - a[bad_c]
        e[bad_c] = d[bad_c]

        swap = np.abs(fc) < np.abs(fb)
        a[swap] = b[swap]
        b[swap] = c[swap]
        c[swap] = a[swap]
        fa[swap] = fb[swap]
        fb[swap] = fc[swap]
        fc[swap] = fa[swap]

        tol1 = 2.0 * eps * np.abs(b) + 0.5 * tol
        Xm = 0.5 * (c - b)
        done = (np.abs(Xm) <= tol1) | (fb == 0.0)
        roots[done] = b[done]
        if np.sum(done) == toteval:
            return roots, True

        use_iq = (np.abs(e) >= tol1) & (np.abs(fa) > np.abs(fb))
        s_val = fb / np.where(np.abs(fa) > 1e-30, fa, 1e-30)

        p = np.zeros_like(a)
        q = np.ones_like(a)

        same = (a == c) & use_iq
        p[same] = 2.0 * Xm[same] * s_val[same]
        q[same] = 1.0 - s_val[same]

        diff = (~same) & use_iq
        q_ac = fa[diff] / np.where(np.abs(fc[diff]) > 1e-30, fc[diff], 1e-30)
        r_ac = fb[diff] / np.where(np.abs(fc[diff]) > 1e-30, fc[diff], 1e-30)
        p[diff] = s_val[diff] * (
            2.0 * Xm[diff] * q_ac * (q_ac - r_ac) - (b[diff] - a[diff]) * (r_ac - 1.0)
        )
        q[diff] = (q_ac - 1.0) * (r_ac - 1.0) * (s_val[diff] - 1.0)

        pos_p = (p > 0) & use_iq
        q[pos_p] = -q[pos_p]
        p[use_iq] = np.abs(p[use_iq])

        use_p = use_iq & (2.0 * p < np.minimum(
            3.0 * Xm * q - np.abs(tol1 * q), np.abs(e * q)
        ))
        e[use_p] = d[use_p]
        d[use_p] = p[use_p] / q[use_p]

        bisect = use_iq & (~use_p)
        d[bisect] = Xm[bisect]
        e[bisect] = d[bisect]

        no_iq = ~use_iq
        d[no_iq] = Xm[no_iq]
        e[no_iq] = d[no_iq]

        a[:] = b
        fa[:] = fb

        big = np.abs(d) > tol1
        b[big] += d[big]
        small = ~big
        b[small] += np.copysign(tol1[small], Xm[small])
        fb = _fzero_residuals(b, theta_pest_targets, phis, xm, xn, lmns, lmnc, lasym, wh, ih)

    return roots, False


def traced_fieldline(vmec, snorm=0.5, nphi=401, nalpha=55, alpha0=0,
                     nfpinc=1, phi_start=0, verbose=False):
    """
    Field-line tracer in VMEC-pest coordinates.

    Ported from Helpers.py:TracedFieldline. Computes iota, I, G, |B|,
    arc-length, and Boozer toroidal angle along field lines.

    Returns a dict with keys:
        iota, I, G, B (nphi, nalpha), ls (nphi, nalpha),
        alphas (nalpha,), phis (nphi,), phiBs (nphi, nalpha)
    """
    vmec.run()
    w = vmec.wout
    pi = np.pi

    xn_nyq = w.xn_nyq
    xm_nyq = w.xm_nyq
    bmnc = w.bmnc.T
    ns = int(w.ns)
    nfp = w.nfp
    xm = w.xm
    xn = w.xn
    lmns = w.lmns.T
    mnmax_nyq = w.mnmax_nyq
    mnmax = w.mnmax
    mpol = w.mpol
    ntor = w.ntor
    rmnc = w.rmnc.T
    gmnc = w.gmnc.T
    zmns = w.zmns.T
    bvco = w.bvco
    buco = w.buco
    bsupvmnc = w.bsupvmnc.T
    lasym = w.lasym

    if lasym == 1:
        lmnc = w.lmnc.T if hasattr(w, 'lmnc') else 0 * lmns
    else:
        lmnc = 0 * lmns

    # ----- radial interpolation weights (full grid) -----
    snorm_full = np.linspace(0, 1, ns)
    snorm_half = (snorm_full[:-1] + snorm_full[1:]) * 0.5

    idx_f = [0, 0]
    wgt_f = [1.0, 0.0]
    if 0 < snorm < 1:
        idx_f[0] = int(np.ceil(snorm * (ns - 1))) - 1
        idx_f[1] = idx_f[0] + 1
        wgt_f[0] = idx_f[1] - snorm * (ns - 1)
    elif snorm >= 1:
        idx_f[0] = ns - 2
        idx_f[1] = ns - 1
        wgt_f[0] = 0.0
    wgt_f[1] = 1.0 - wgt_f[0]

    # ----- radial interpolation weights (half grid) -----
    idx_h = [0, 0]
    wgt_h = [1.0, 0.0]
    if snorm < snorm_half[0]:
        idx_h[0] = 1
        idx_h[1] = 2
        wgt_h[0] = (snorm_half[1] - snorm) / (snorm_half[1] - snorm_half[0])
    elif snorm > snorm_half[-1]:
        idx_h[0] = ns - 2
        idx_h[1] = ns - 1
        wgt_h[0] = (snorm_half[-1] - snorm) / (snorm_half[-1] - snorm_half[-2])
    elif snorm == snorm_half[-1]:
        idx_h[0] = ns - 2
        idx_h[1] = ns - 1
        wgt_h[0] = 0.0
    else:
        idx_h[0] = int(np.ceil(snorm * (ns - 1) + 0.5)) - 1
        if idx_h[0] < 1:
            idx_h[0] = 1
        idx_h[1] = idx_h[0] + 1
        wgt_h[0] = idx_h[0] + 0.5 - snorm * (ns - 1)
    wgt_h[1] = 1.0 - wgt_h[0]

    # ----- derived scalars -----
    iota = float(UnivariateSpline(snorm_half, w.iotas[1:], k=1, s=0)(snorm))
    G = float(UnivariateSpline(snorm_half, bvco[1:], k=1, s=0)(snorm))
    I = float(UnivariateSpline(snorm_half, buco[1:], k=1, s=0)(snorm))

    alphas = np.linspace(alpha0, 2 * pi * (1 - 1 / nalpha), nalpha)

    if nfpinc <= 0:
        nfpinc = nfp

    phis = []
    for iphi in range(nfpinc):
        phii = np.linspace(
            phi_start + iphi * 2 * pi / nfp,
            phi_start + iphi * 2 * pi / nfp + 2 * pi / nfp,
            int(nphi / nfpinc),
        )
        phis = np.append(phis, phii[:-1])
    phis = np.append(phis, 2 * pi * nfpinc / nfp)
    actual_nphi = len(phis)

    # ----- theta_pest -> theta_vmec -----
    theta_pest_targets = alphas[None, :] + iota * phis[:, None]
    theta_vmec_mins = theta_pest_targets - 0.3
    theta_vmec_maxs = theta_pest_targets + 0.3
    theta_vmec, converged = _get_roots(
        theta_vmec_mins, theta_vmec_maxs, theta_pest_targets, phis,
        xm, xn, lmns, lmnc, lasym, wgt_h, idx_h,
    )
    if not converged and verbose:
        print("WARNING: theta_pest -> theta_vmec conversion did not fully converge")

    B = np.zeros((actual_nphi, nalpha))
    B_sup_zeta = np.zeros((actual_nphi, nalpha))
    R = np.zeros((actual_nphi, nalpha))
    Z = np.zeros((actual_nphi, nalpha))

    for imn_nyq in range(mnmax_nyq):
        m_mode = int(xm_nyq[imn_nyq])
        n_mode = int(xn_nyq[imn_nyq] / nfp)
        non_nyq = (abs(m_mode) < mpol and abs(n_mode) <= ntor)
        if non_nyq:
            found_imn = None
            for imn in range(mnmax):
                if xm[imn] == m_mode and xn[imn] == n_mode * nfp:
                    found_imn = imn
                    break

        angle = m_mode * theta_vmec - n_mode * nfp * phis[:, None]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        temp_b = bmnc[idx_h[0], imn_nyq] * wgt_h[0] + bmnc[idx_h[1], imn_nyq] * wgt_h[1]
        B += temp_b * cos_a

        temp_v = bsupvmnc[idx_h[0], imn_nyq] * wgt_h[0] + bsupvmnc[idx_h[1], imn_nyq] * wgt_h[1]
        B_sup_zeta += temp_v * cos_a

        if non_nyq and found_imn is not None:
            temp_r = rmnc[idx_f[0], found_imn] * wgt_f[0] + rmnc[idx_f[1], found_imn] * wgt_f[1]
            R += temp_r * cos_a
            temp_z = zmns[idx_f[0], found_imn] * wgt_f[0] + zmns[idx_f[1], found_imn] * wgt_f[1]
            Z += temp_z * sin_a

    ls = np.zeros((actual_nphi, nalpha))
    phiBs = np.zeros((actual_nphi, nalpha))
    dphi = phis[1] - phis[0]
    for i in range(1, actual_nphi):
        dl = ((B[i, :] + B[i - 1, :]) / 2) * dphi / ((B_sup_zeta[i, :] + B_sup_zeta[i - 1, :]) / 2)
        ls[i, :] = ls[i - 1, :] + dl
        dphiB = (B[i, :] + B[i - 1, :]) / 2 * dl / (iota * I + G)
        phiBs[i, :] = phiBs[i - 1, :] + dphiB

    return dict(
        iota=iota, I=I, G=G, B=B, ls=ls,
        alphas=alphas, phis=phis, phiBs=phiBs,
    )
