"""
Available Energy of trapped particles — turbulence stability diagnostic.

Ported from Goodman et al. PRX Energy 3, 023010 (2024),
``plots/plot_available_energy/AE_routines.py``.

The Available Energy (AE) is the thermodynamic free energy accessible to
trapped-electron-mode (TEM) turbulence.  It integrates over bounce wells,
drift frequencies (ω_ψ, ω_α), and normalised energy z.  Lower AE means
less TEM drive — a key advantage of optimised QI configurations.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy import integrate, special


# ──────────────────────────────────────────────────────────────────
#  Low-level bounce-integral primitives
# ──────────────────────────────────────────────────────────────────

def _zero_cross_idx(y_arr):
    """Indices left of zero-crossings of *y_arr*."""
    mask = y_arr[:-1] * y_arr[1:] < 0
    return np.nonzero(mask)[0], int(np.sum(mask))


def _inner_trapz(h_i, h_j, f_i, f_j, theta_i, theta_j):
    return np.sum(
        (-2 * (np.sqrt(f_j) * (2 * h_i + h_j)
               + np.sqrt(f_i) * (h_i + 2 * h_j))
         * (theta_i - theta_j))
        / (3.0 * (f_i + f_j + 2 * np.sqrt(f_i * f_j)))
    )


def _left_trapz(h_l, h_0, f_0, theta_l, theta_0):
    return 2 * (2 * h_l + h_0) * (theta_0 - theta_l) / (3.0 * np.sqrt(f_0))


def _right_trapz(h_n, h_r, f_n, theta_n, theta_r):
    return 2 * (h_n + 2 * h_r) * (theta_r - theta_n) / (3.0 * np.sqrt(f_n))


# ──────────────────────────────────────────────────────────────────
#  Bounce wells and bounce-averaging
# ──────────────────────────────────────────────────────────────────

def bounce_wells(theta_arr, b_arr, lam_val):
    """
    Identify bounce wells for pitch parameter *lam_val*.

    Returns (bounce_idx, bounce_arr, num_wells).
    """
    zero_arr = 1.0 - lam_val * b_arr
    zero_idx, num_cross = _zero_cross_idx(zero_arr)
    if num_cross % 2 != 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    num_wells = num_cross // 2
    if num_wells == 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0

    if b_arr[zero_idx[0] + 1] - b_arr[zero_idx[0]] > 0:
        zero_idx = np.roll(zero_idx, -1)

    bounce_idx = np.empty((num_wells, 2), dtype=int)
    bounce_arr = np.empty((num_wells, 2))
    for k in range(num_wells):
        li = zero_idx[2 * k]
        ri = zero_idx[2 * k + 1]
        bounce_idx[k] = [li, ri]
        bounce_arr[k, 0] = ((-zero_arr[li + 1] * theta_arr[li]
                              + zero_arr[li] * theta_arr[li + 1])
                             / (zero_arr[li] - zero_arr[li + 1]))
        bounce_arr[k, 1] = ((-zero_arr[ri + 1] * theta_arr[ri]
                              + zero_arr[ri] * theta_arr[ri + 1])
                             / (zero_arr[ri] - zero_arr[ri + 1]))
    return bounce_idx, bounce_arr, num_wells


def bounce_average(theta_arr, h_arr, b_arr, lam):
    """Bounce-average ∫ h(θ)/√(1 − λ B(θ)) dθ over all wells."""
    bounce_idx, bounce_arr, num_wells = bounce_wells(theta_arr, b_arr, lam)
    if num_wells == 0:
        return np.array([])
    f_arr = 1 - lam * b_arr
    result = np.empty(num_wells)

    for k in range(num_wells):
        li = int(bounce_idx[k, 0])
        ri = int(bounce_idx[k, 1])
        if li > ri:
            inner = (_inner_trapz(h_arr[li + 1:-1], h_arr[li + 2:],
                                  f_arr[li + 1:-1], f_arr[li + 2:],
                                  theta_arr[li + 1:-1], theta_arr[li + 2:])
                     + _inner_trapz(h_arr[:ri], h_arr[1:ri + 1],
                                    f_arr[:ri], f_arr[1:ri + 1],
                                    theta_arr[:ri], theta_arr[1:ri + 1]))
        else:
            inner = _inner_trapz(h_arr[li + 1:ri], h_arr[li + 2:ri + 1],
                                 f_arr[li + 1:ri], f_arr[li + 2:ri + 1],
                                 theta_arr[li + 1:ri], theta_arr[li + 2:ri + 1])

        h_l = (h_arr[li] + (bounce_arr[k, 0] - theta_arr[li])
               / (theta_arr[li + 1] - theta_arr[li])
               * (h_arr[li + 1] - h_arr[li]))
        left = _left_trapz(h_l, h_arr[li + 1], f_arr[li + 1],
                            bounce_arr[k, 0], theta_arr[li + 1])

        h_r = (h_arr[ri] + (bounce_arr[k, 1] - theta_arr[ri])
               / (theta_arr[ri + 1] - theta_arr[ri])
               * (h_arr[ri + 1] - h_arr[ri]))
        right = _right_trapz(h_arr[ri], h_r, f_arr[ri],
                              theta_arr[ri], bounce_arr[k, 1])

        result[k] = left + inner + right
    return result


# ──────────────────────────────────────────────────────────────────
#  Drift frequencies
# ──────────────────────────────────────────────────────────────────

def drift_frequencies(q0, L_tot, b_arr, dbdx_arr, dbdy_arr,
                      sqrtg_arr, theta_arr, lam, Delta_x, Delta_y):
    """
    Bounce-averaged drift frequencies and bounce time.

    Returns (w_psi, w_alpha, G) — arrays of length num_wells.
    """
    h0 = q0 * b_arr * sqrtg_arr
    denom = bounce_average(theta_arr, h0, b_arr, lam)
    if len(denom) == 0:
        return np.array([]), np.array([]), np.array([])

    h_alpha = lam * Delta_x * dbdx_arr * q0 * b_arr * sqrtg_arr
    num_alpha = bounce_average(theta_arr, h_alpha, b_arr, lam)

    h_psi = -lam * Delta_y * dbdy_arr * q0 * b_arr * sqrtg_arr
    num_psi = bounce_average(theta_arr, h_psi, b_arr, lam)

    return num_psi / denom, num_alpha / denom, denom / L_tot


# ──────────────────────────────────────────────────────────────────
#  AE integrand and total
# ──────────────────────────────────────────────────────────────────

def _ae_integrand(walpha, wpsi, G, dlnTdx, dlnndx, Delta_x, z):
    wdia = Delta_x * (dlnndx / z + dlnTdx * (1.0 - 1.5 / z))
    return np.sum(
        G * (walpha * (-walpha + wdia) - wpsi**2
             + np.sqrt(walpha**2 + wpsi**2)
             * np.sqrt((walpha - wdia)**2 + wpsi**2))
        * z**2.5 * np.exp(-z)
    )


def _integral_over_z(c0, c1):
    """Analytic energy integral for the omnigenous limit."""
    if c0 >= 0 and c1 <= 0:
        return 2 * c0 - 5 * c1
    if c0 >= 0 and c1 > 0:
        r = np.sqrt(c0 / c1)
        return ((2 * c0 - 5 * c1) * special.erf(r)
                + 2 / (3 * np.sqrt(np.pi)) * (4 * c0 + 15 * c1)
                * r * np.exp(-c0 / c1))
    if c0 < 0 and c1 < 0:
        r = np.sqrt(c0 / c1)
        return ((2 * c0 - 5 * c1) * (1 - special.erf(r))
                - 2 / (3 * np.sqrt(np.pi)) * (4 * c0 + 15 * c1)
                * r * np.exp(-c0 / c1))
    return 0.0


_vint = np.vectorize(_integral_over_z, otypes=[np.float64])


def _filter_lambda(lam_arr, B_arr, delta_lam):
    """Remove λ values near local B maxima (singularities)."""
    peaks = find_peaks(B_arr)[0]
    if len(peaks) == 0:
        return lam_arr
    B_peaks = B_arr[peaks]
    lam_inf = 1.0 / B_peaks
    lam_range = 1.0 / np.min(B_arr) - 1.0 / np.max(B_arr)
    mask = np.ones(len(lam_arr), dtype=bool)
    for li in lam_inf:
        mask &= ~((lam_arr >= li - delta_lam * lam_range)
                   & (lam_arr <= li + delta_lam * lam_range))
    return lam_arr[mask]


def _make_periodic(b, dbdx, dbdy, sqrtg, theta, dtheta):
    return (np.append(b, b[0]),
            np.append(dbdx, dbdx[0]),
            np.append(dbdy, dbdy[0]),
            np.append(sqrtg, sqrtg[0]),
            np.append(theta, theta[-1] + dtheta))


def available_energy(q0, dlnTdx, dlnndx, Delta_x, Delta_y,
                     b_arr, dbdx_arr, dbdy_arr, sqrtg_arr,
                     theta_arr, lam_res=200, delta_sing=0.01,
                     L_tot=1.0, omnigenous=False):
    """
    Total Available Energy integrated over λ and energy z.

    Parameters
    ----------
    q0 : float
        Safety factor (1/iota).
    dlnTdx, dlnndx : float
        Normalised temperature and density gradients.
    Delta_x, Delta_y : float
        Radial and binormal length scales.
    b_arr : array
        |B| along the field line.
    dbdx_arr, dbdy_arr : array
        Gradients of |B| w.r.t. radial and binormal directions.
    sqrtg_arr : array
        Jacobian √g along the field line.
    theta_arr : array
        Field-line-following angle.
    lam_res : int
        Number of λ grid points.
    delta_sing : float
        Singularity padding around local B maxima.
    L_tot : float
        Total field-line length for normalisation.
    omnigenous : bool
        If True, use analytic energy integral (faster, exact for QI).

    Returns
    -------
    ae : float
        Total available energy.
    """
    dtheta = theta_arr[1] - theta_arr[0]
    b, dbx, dby, sg, th = _make_periodic(
        b_arr, dbdx_arr, dbdy_arr, sqrtg_arr, theta_arr, dtheta)

    lam_min = 1.0 / np.max(b)
    lam_max = 1.0 / np.min(b)
    lam_arr = np.linspace(lam_min, lam_max, lam_res + 1, endpoint=False)[1:]
    lam_arr = _filter_lambda(lam_arr, b, delta_sing)

    ae_per_lam = np.zeros(len(lam_arr))
    for i, lam_val in enumerate(lam_arr):
        wp, wa, G = drift_frequencies(
            q0, L_tot, b, dbx, dby, sg, th, lam_val, Delta_x, Delta_y)
        if len(G) == 0:
            continue
        if omnigenous:
            c0 = Delta_x * (dlnndx - 1.5 * dlnTdx) / wa
            c1 = 1.0 - Delta_x * dlnTdx / wa
            ae_per_lam[i] = (0.75 * np.sqrt(np.pi)
                             * np.sum(wa**2 * _vint(c0, c1) * G))
        else:
            ae_per_lam[i] = integrate.quad(
                lambda z: _ae_integrand(wa, wp, G, dlnTdx, dlnndx, Delta_x, z),
                0, np.inf, epsrel=1e-6, epsabs=1e-20, limit=500)[0]

    return float(np.trapz(ae_per_lam, lam_arr))


# ──────────────────────────────────────────────────────────────────
#  VMEC-level wrappers
# ──────────────────────────────────────────────────────────────────

def ae_surface(vmec, s_val, omn=1.0, omt=3.0, n_alpha=4,
               n_turns=3, lam_res=200, gridpoints=512,
               omnigenous=False):
    """
    Available Energy on a single flux surface, averaged over field lines.

    Parameters
    ----------
    vmec : simsopt.mhd.Vmec
        VMEC equilibrium (must have been run).
    s_val : float
        Normalised toroidal flux.
    omn, omt : float
        -d ln n / d s,  -d ln T / d s.
    n_alpha : int
        Number of field-line labels for flux-surface average.
    n_turns : int
        Poloidal turns of the field line to trace.
    lam_res : int
        Lambda resolution.
    gridpoints : int
        Points per field line.
    omnigenous : bool
        If True, set radial drifts to zero (analytic energy integral).

    Returns
    -------
    ae_val : float
        AE / (3/2 n T) per unit volume.
    """
    from simsopt.mhd.vmec_diagnostics import vmec_fieldlines

    nfp = int(vmec.wout.nfp)
    alpha_arr = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)
    B_ref = vmec.wout.volavgB if hasattr(vmec.wout, 'volavgB') else 1.0
    L_ref = 1.0
    phi_edge = float(vmec.wout.phi[-1]) / (2 * np.pi)

    ae_alpha = np.zeros(n_alpha)
    theta1d = np.linspace(-n_turns * np.pi, n_turns * np.pi, gridpoints)

    for ia, alpha_val in enumerate(alpha_arr):
        try:
            fl = vmec_fieldlines(vmec, s_val, alpha_val, theta1d=theta1d)
        except Exception:
            continue

        iota = float(np.array(fl.iota).flatten()[0])
        if abs(iota) < 1e-10:
            continue

        modB   = np.array(fl.modB).flatten()
        theta_p = np.array(fl.theta_pest).flatten()

        grad_B   = np.stack([np.array(fl.grad_B_X).flatten(),
                             np.array(fl.grad_B_Y).flatten(),
                             np.array(fl.grad_B_Z).flatten()], axis=1)
        grad_psi = np.stack([np.array(fl.grad_psi_X).flatten(),
                             np.array(fl.grad_psi_Y).flatten(),
                             np.array(fl.grad_psi_Z).flatten()], axis=1)
        grad_phi = np.stack([np.array(fl.grad_phi_X).flatten(),
                             np.array(fl.grad_phi_Y).flatten(),
                             np.array(fl.grad_phi_Z).flatten()], axis=1)
        grad_alpha = np.stack([np.array(fl.grad_alpha_X).flatten(),
                               np.array(fl.grad_alpha_Y).flatten(),
                               np.array(fl.grad_alpha_Z).flatten()], axis=1)
        B_sup_phi = np.array(fl.B_sup_phi).flatten()

        cross_alpha_phi = np.cross(grad_alpha, grad_phi)
        dBdpsi = (np.einsum('ij,ij->i', cross_alpha_phi, grad_B)
                  / B_sup_phi)

        cross_phi_psi = np.cross(grad_phi, grad_psi)
        dBdalpha = (np.einsum('ij,ij->i', cross_phi_psi, grad_B)
                    / B_sup_phi)

        jac = 1.0 / B_sup_phi

        b_norm = modB / B_ref
        dbdx = dBdpsi * 2 * np.sqrt(s_val) / B_ref * phi_edge
        dbdy = dBdalpha / (np.sqrt(max(s_val, 1e-6)) * B_ref)
        sqrt_g = np.abs(jac / (L_ref**3 * iota / 2) * phi_edge)

        L_tot = float(np.abs(np.trapz(b_norm * sqrt_g / iota, theta_p)))
        if L_tot < 1e-30:
            continue
        rec_B_ave = float(np.abs(np.trapz(sqrt_g / iota, theta_p) / L_tot))

        ae_val = available_energy(
            q0=1.0,
            dlnTdx=-omt,
            dlnndx=-omn,
            Delta_x=1.0,
            Delta_y=1.0,
            b_arr=b_norm,
            dbdx_arr=dbdx,
            dbdy_arr=dbdy,
            sqrtg_arr=sqrt_g,
            theta_arr=theta_p,
            lam_res=lam_res,
            delta_sing=0.0,
            L_tot=L_tot,
            omnigenous=omnigenous,
        )
        if ae_val < 0:
            ae_val = 0.0
        ae_alpha[ia] = ae_val / max(rec_B_ave * 6 * np.sqrt(np.pi), 1e-30)

    return float(np.mean(ae_alpha)) / max(abs(iota), 1e-6)**2


def ae_diagnostics(vmec, s_vals=None, omn=1.0, omt=3.0,
                   n_alpha=4, n_turns=3, lam_res=200,
                   gridpoints=512, verbose=True):
    """
    Evaluate Available Energy on multiple surfaces.

    Returns
    -------
    dict : {s_val: ae_value, ...}, plus 'total' key.
    """
    if s_vals is None:
        s_vals = np.array([0.2, 0.5, 0.8])
    s_vals = np.asarray(s_vals)

    results = {}
    for s in s_vals:
        try:
            ae = ae_surface(vmec, float(s), omn, omt, n_alpha,
                            n_turns, lam_res, gridpoints)
        except Exception as e:
            ae = float('nan')
            if verbose:
                print(f"    AE at s={s:.2f}: FAILED ({e})")
        results[float(s)] = ae

    total = float(np.nanmean(list(results.values())))
    results['total'] = total

    if verbose:
        print(f"\n  Available Energy (TEM turbulence metric):")
        print(f"    omn={omn}, omt={omt}, n_alpha={n_alpha}, "
              f"n_turns={n_turns}")
        for s in s_vals:
            print(f"    s={s:.2f}: AE = {results[float(s)]:.6e}")
        print(f"    Mean AE = {total:.6e}")

    return results
