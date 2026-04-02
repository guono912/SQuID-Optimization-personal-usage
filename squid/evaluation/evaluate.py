"""
Diagnostic evaluation for SQuID equilibria.

Merged from:
  - max_J_evaluation.py  (evaluate_max_J, plot_gradient_diagnostics,
                           plot_J_contours)
  - SQuID_QI.py  (compute_SQuID_QI, plot_B_contours)
"""

import numpy as np
import matplotlib.pyplot as plt

from ..core.boozer_utils import run_boozer, reconstruct_B
from ..core.squash_stretch import squash_and_stretch_simple
from ..objectives.maxj_residual import (
    _extract_field_line, _compute_J_C, _compute_J_I, _evaluate_squid,
)
from ..objectives.itg_residual import ITGResidual


def evaluate_squid(vmec, s_vals=None, num_alpha=8, num_pitch=50,
                   T_J=-0.06, mboz=8, nboz=8, verbose=True,
                   return_details=False):
    """
    Full SQuID diagnostic evaluation.

    Returns a dict with f_maxJ, f_QI, mirror_ratio, iota_axis, iota_edge,
    B_min, B_max.
    """
    if s_vals is None:
        s_vals = np.linspace(0.2, 0.8, 5)
    alphas = np.linspace(0, 2 * np.pi, num_alpha, endpoint=False)

    if return_details:
        info = evaluate_squid_detailed(
            vmec, s_vals=s_vals, num_alpha=num_alpha, num_pitch=num_pitch,
            T_J=T_J, mboz=mboz, nboz=nboz, verbose=False,
        )
    else:
        info = _evaluate_squid(vmec, s_vals, alphas, num_pitch, T_J, mboz, nboz)

    if verbose:
        print(f"\n  SQuID Evaluation Results:")
        print(f"    f_maxJ       = {info['f_maxJ']:.6e}")
        print(f"    f_QI         = {info['f_QI']:.6e}")
        print(f"    mirror ratio = {info['mirror_ratio']:.4f}")
        print(f"    iota_axis    = {info['iota_axis']:.4f}")
        print(f"    iota_edge    = {info['iota_edge']:.4f}")
        print(f"    B_min        = {info['B_min']:.4f}")
        print(f"    B_max        = {info['B_max']:.4f}")

    return info


def evaluate_squid_detailed(vmec, s_vals=None, num_alpha=8, num_pitch=50,
                            T_J=-0.06, mboz=8, nboz=8, verbose=True):
    """
    Evaluate the core SQuID targets and retain local diagnostics.

    The returned dict augments the historical ``evaluate_squid`` outputs with
    per-surface/per-interval summaries that are suitable for a single
    equilibrium diagnostic report:

      - QI surface RMS and 95th percentile
      - QI heatmap over (surface, trapped-depth)
      - max-J pass ratio and violation fraction
      - max-J heatmap over (radial interval, trapped-depth)
    """
    if s_vals is None:
        s_vals = np.linspace(0.2, 0.8, 5)
    s_vals = np.asarray(s_vals, dtype=float)
    alphas = np.linspace(0, 2 * np.pi, num_alpha, endpoint=False)

    _, surface_data = run_boozer(vmec, s_vals, mpol=mboz, ntor=nboz)

    ns = len(s_vals)
    na = len(alphas)
    np_ = num_pitch
    lambda_grid = np.linspace(1.0 / (np_ + 1), np_ / (np_ + 1), np_)

    vmec_iotaf = np.array(vmec.wout.iotaf)
    iota_axis = float(vmec_iotaf[0])
    iota_edge = float(vmec_iotaf[-1])

    all_bs, all_jc, all_ji = [], [], []
    local_jc, local_ji = [], []
    bmins, bmaxs = [], []
    global_Bmin, global_Bmax = 1e30, -1e30

    qi_surface_rms = np.zeros(ns)
    qi_surface_p95 = np.zeros(ns)
    qi_lambda_rms = np.zeros((ns, np_))
    qi_worst_surface_alpha = None

    for k in range(ns):
        data = surface_data[k]
        B_min = float(data["B_min"])
        B_max = float(data["B_max"])
        bmins.append(B_min)
        bmaxs.append(B_max)
        global_Bmin = min(global_Bmin, B_min)
        global_Bmax = max(global_Bmax, B_max)

        B_stars = np.linspace(B_min, B_max, np_ + 2)[1:-1]
        jc_rows, ji_rows = [], []
        for alpha in alphas:
            zeta, B_I = _extract_field_line(data, alpha)
            B_C = squash_and_stretch_simple(zeta, B_I, B_min, B_max)
            jc_rows.append(_compute_J_C(zeta, B_I, B_C, B_stars))
            ji_rows.append(_compute_J_I(zeta, B_I, B_C, B_stars))

        jc_arr = np.array(jc_rows).T
        ji_arr = np.array(ji_rows).T
        all_bs.append(B_stars)
        all_jc.append(jc_arr)
        all_ji.append(ji_arr)
        local_jc.append(jc_arr)
        local_ji.append(ji_arr)

        denom_local = float(np.mean(ji_arr + jc_arr))
        if abs(denom_local) < 1e-30:
            denom_local = 1e-30
        qi_diff_local = (ji_arr[:, :, None] - jc_arr[:, None, :]) / denom_local
        qi_abs_local = np.abs(qi_diff_local)
        qi_surface_rms[k] = float(np.sqrt(np.mean(qi_diff_local ** 2)))
        qi_surface_p95[k] = float(np.percentile(qi_abs_local, 95))
        qi_lambda_rms[k, :] = np.sqrt(np.mean(qi_diff_local ** 2, axis=(1, 2)))

    mirror_ratio = (
        (global_Bmax - global_Bmin) / (global_Bmax + global_Bmin)
        if (global_Bmax + global_Bmin) > 1e-30 else 0.0
    )

    b_lo = max(b[0] for b in all_bs)
    b_hi = min(b[-1] for b in all_bs)
    common_B_valid = b_lo < b_hi
    if not common_B_valid:
        info = dict(
            f_maxJ=1e6,
            f_QI=1e6,
            mirror_ratio=mirror_ratio,
            iota_axis=iota_axis,
            iota_edge=iota_edge,
            B_min=global_Bmin,
            B_max=global_Bmax,
            s_vals=s_vals,
            interval_centers=np.array([]),
            lambda_grid=lambda_grid,
            qi_surface_rms=qi_surface_rms,
            qi_surface_p95=qi_surface_p95,
            qi_lambda_rms=qi_lambda_rms,
            qi_worst_surface_idx=int(np.argmax(qi_surface_rms)),
            maxj_interval_pass_ratio=np.array([]),
            maxj_interval_violation_fraction=np.array([]),
            maxj_interval_worst_lambda=np.array([]),
            maxj_lambda_violation=np.zeros((0, np_)),
            maxj_lambda_exceedance=np.zeros((0, np_)),
            maxj_global_pass_ratio=0.0,
            maxj_global_violation_fraction=1.0,
            common_B_valid=False,
        )
        if verbose:
            print("\n  SQuID Evaluation Results:")
            print(f"    f_maxJ       = {info['f_maxJ']:.6e}")
            print(f"    f_QI         = {info['f_QI']:.6e}")
            print(f"    mirror ratio = {info['mirror_ratio']:.4f}")
            print("    [WARN] No common B* range across the requested surfaces.")
        return info

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

    f_maxJ = 0.0
    maxj_interval_pass_ratio = np.zeros(max(ns - 1, 0))
    maxj_interval_violation_fraction = np.zeros(max(ns - 1, 0))
    maxj_interval_worst_lambda = np.zeros(max(ns - 1, 0))
    maxj_lambda_violation = np.zeros((max(ns - 1, 0), np_))
    maxj_lambda_exceedance = np.zeros((max(ns - 1, 0), np_))

    for k in range(max(ns - 1, 0)):
        ds = s_vals[k + 1] - s_vals[k]
        J_lo, J_hi = JC_int[k], JC_int[k + 1]
        denom = float(np.mean(J_hi + J_lo))
        if abs(denom) < 1e-30:
            denom = 1e-30
        diff = J_hi[:, :, None] - J_lo[:, None, :]
        mean_dJ = (diff / (ds * denom)).mean(axis=-1)
        penalty = np.maximum(0.0, mean_dJ - T_J) ** 2
        f_maxJ += float(penalty.sum())

        pass_mask = mean_dJ < T_J
        exceedance = np.maximum(mean_dJ - T_J, 0.0)
        maxj_interval_pass_ratio[k] = float(np.mean(pass_mask))
        maxj_interval_violation_fraction[k] = 1.0 - maxj_interval_pass_ratio[k]
        maxj_lambda_violation[k, :] = np.mean(~pass_mask, axis=1)
        maxj_lambda_exceedance[k, :] = np.mean(exceedance, axis=1)

        lambda_lo = (common_B - bmins[k]) / max(bmaxs[k] - bmins[k], 1e-30)
        lambda_hi = (common_B - bmins[k + 1]) / max(bmaxs[k + 1] - bmins[k + 1], 1e-30)
        lambda_mid = 0.5 * (lambda_lo + lambda_hi)
        worst_idx = int(np.argmax(maxj_lambda_exceedance[k, :]))
        maxj_interval_worst_lambda[k] = float(lambda_mid[worst_idx])

    f_QI = 0.0
    qi_worst_surface_idx = int(np.argmax(qi_surface_rms))
    for k in range(ns):
        denom = float(np.mean(JI_int[k] + JC_int[k]))
        if abs(denom) < 1e-30:
            denom = 1e-30
        diff = JI_int[k][:, :, None] - JC_int[k][:, None, :]
        f_QI += float(np.sum((diff / denom) ** 2))
        if k == qi_worst_surface_idx:
            qi_worst_surface_alpha = np.sqrt(np.mean((diff / denom) ** 2, axis=2)).T

    interval_centers = 0.5 * (s_vals[:-1] + s_vals[1:])
    if len(interval_centers) > 0:
        worst_interval_idx = int(np.argmax(maxj_interval_violation_fraction))
        worst_lambda = float(maxj_interval_worst_lambda[worst_interval_idx])
    else:
        worst_interval_idx = -1
        worst_lambda = float("nan")

    info = dict(
        f_maxJ=f_maxJ,
        f_QI=f_QI,
        mirror_ratio=mirror_ratio,
        iota_axis=iota_axis,
        iota_edge=iota_edge,
        B_min=global_Bmin,
        B_max=global_Bmax,
        s_vals=s_vals,
        interval_centers=interval_centers,
        lambda_grid=lambda_grid,
        qi_surface_rms=qi_surface_rms,
        qi_surface_p95=qi_surface_p95,
        qi_lambda_rms=qi_lambda_rms,
        qi_worst_surface_idx=qi_worst_surface_idx,
        qi_worst_surface_alpha=qi_worst_surface_alpha,
        maxj_interval_pass_ratio=maxj_interval_pass_ratio,
        maxj_interval_violation_fraction=maxj_interval_violation_fraction,
        maxj_interval_worst_lambda=maxj_interval_worst_lambda,
        maxj_lambda_violation=maxj_lambda_violation,
        maxj_lambda_exceedance=maxj_lambda_exceedance,
        maxj_global_pass_ratio=(
            float(np.mean(maxj_interval_pass_ratio))
            if len(maxj_interval_pass_ratio) > 0 else float("nan")
        ),
        maxj_global_violation_fraction=(
            float(np.mean(maxj_interval_violation_fraction))
            if len(maxj_interval_violation_fraction) > 0 else float("nan")
        ),
        maxj_worst_interval_idx=worst_interval_idx,
        maxj_worst_lambda=worst_lambda,
        common_B_valid=True,
    )

    if verbose:
        print(f"\n  SQuID Evaluation Results:")
        print(f"    f_maxJ       = {info['f_maxJ']:.6e}")
        print(f"    f_QI         = {info['f_QI']:.6e}")
        print(f"    mirror ratio = {info['mirror_ratio']:.4f}")
        print(f"    iota_axis    = {info['iota_axis']:.4f}")
        print(f"    iota_edge    = {info['iota_edge']:.4f}")
        print(f"    B_min        = {info['B_min']:.4f}")
        print(f"    B_max        = {info['B_max']:.4f}")
        print(f"    QI worst s   = {s_vals[qi_worst_surface_idx]:.2f}")
        print(f"    max-J pass   = {info['maxj_global_pass_ratio']:.3f}")

    return info


def evaluate_itg(vmec, snorms=None, method="drift_curvature", verbose=True):
    """
    Evaluate the ITG turbulence target f_nabla_s.

    Parameters
    ----------
    vmec : Vmec object
    snorms : array-like of flux surfaces (default [0.1, 0.3, 0.5])
    method : "drift_curvature" or "vacuum_dBds"
    """
    if snorms is None:
        snorms = np.array([0.1, 0.3, 0.5])

    itg = ITGResidual(vmec, snorms, method=method)
    total = itg.total()
    residuals = itg.residuals()

    if verbose:
        print(f"\n  ITG Evaluation (method={method}):")
        print(f"    Total f_nabla_s = {total:.6e}")
        for s, r in zip(snorms, residuals):
            print(f"    s={s:.2f}: {r:.6e}")

    return dict(total=total, per_surface=dict(zip(snorms, residuals)))


def plot_boozer_surface(vmec, s_val=0.5, mpol=20, ntor=20, ntheta=100, nphi=100):
    """Plot |B| contours on a Boozer surface."""
    _, surface_data = run_boozer(vmec, [s_val], mpol=mpol, ntor=ntor)
    data = surface_data[0]
    nfp = data["nfp"]

    th = np.linspace(0, 2 * np.pi, ntheta)
    ze = np.linspace(0, 2 * np.pi / nfp, nphi)
    TH, ZE = np.meshgrid(th, ze, indexing="ij")
    B_2d = reconstruct_B(data["m"], data["n"], data["bmnc"], TH, ZE)

    fig, ax = plt.subplots(figsize=(10, 6))
    cs = ax.contourf(ZE, TH, B_2d, levels=30, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="|B| (T)")

    iota = data["iota"]
    zeta_line = np.linspace(0, 2 * np.pi / nfp, 200)
    theta_line = iota * zeta_line
    ax.plot(zeta_line, theta_line % (2 * np.pi), "w--", alpha=0.7, label=r"$\alpha=0$ field line")

    ax.set_xlabel(r"Boozer $\zeta$")
    ax.set_ylabel(r"Boozer $\theta$")
    ax.set_title(f"|B| on Boozer surface s={s_val:.2f}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_squash_stretch(vmec, s_val=0.5, alpha=0.0, mpol=20, ntor=20):
    """Plot the squash-and-stretch process on a single field line."""
    _, surface_data = run_boozer(vmec, [s_val], mpol=mpol, ntor=ntor)
    data = surface_data[0]

    zeta, B_I = _extract_field_line(data, alpha)
    B_C = squash_and_stretch_simple(zeta, B_I, data["B_min"], data["B_max"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(zeta, B_I, "b-", lw=2, label=r"$B_I$ (actual)")
    ax.plot(zeta, B_C, "r--", lw=2, label=r"$B_C$ (constructed)")
    ax.axhline(data["B_min"], color="g", ls=":", alpha=0.5, label=f"B_min={data['B_min']:.4f}")
    ax.axhline(data["B_max"], color="m", ls=":", alpha=0.5, label=f"B_max={data['B_max']:.4f}")
    ax.set_xlabel(r"Boozer $\zeta$")
    ax.set_ylabel("|B| (T)")
    ax.set_title(f"Squash & Stretch (s={s_val:.2f}, alpha={alpha:.2f})")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_J_contours(vmec, s_vals=None, alphas=None, lambda_N=0.3,
                    num_pitch=50, mpol=8, ntor=8):
    """
    Polar contours of J̃_C at fixed normalised trapping depth λ_N.

    Replicates PRX Energy (2024) Fig. 9 (Goodman et al.).
    For a max-J configuration, contours should shrink outward.
    """
    import scipy.interpolate as spi

    if s_vals is None:
        s_vals = np.linspace(0.05, 0.95, 12)
    if alphas is None:
        alphas = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    s_vals = np.asarray(s_vals)
    alphas = np.asarray(alphas)

    _, surface_data = run_boozer(vmec, s_vals, mpol=mpol, ntor=ntor)
    num_surfaces = len(s_vals)
    num_alpha = len(alphas)
    J_slice = np.zeros((num_surfaces, num_alpha))

    for k in range(num_surfaces):
        data = surface_data[k]
        B_stars = np.linspace(data["B_min"], data["B_max"], num_pitch + 2)[1:-1]
        B_star_k = data["B_min"] + lambda_N * (data["B_max"] - data["B_min"])
        for a, alpha in enumerate(alphas):
            zeta, B_I = _extract_field_line(data, alpha)
            B_C = squash_and_stretch_simple(
                zeta, B_I, data["B_min"], data["B_max"])
            J_C = _compute_J_C(zeta, B_I, B_C, B_stars)
            J_slice[k, a] = float(np.interp(B_star_k, B_stars, J_C))

    # Axis constraint: at s → 0, J is independent of α
    if s_vals[0] > 0.01:
        s_plot = np.insert(s_vals, 0, 0.0)
        J_axis = np.mean(J_slice[0, :])
        J_plot = np.insert(J_slice, 0, np.full(num_alpha, J_axis), axis=0)
    else:
        s_plot = s_vals.copy()
        J_plot = J_slice.copy()
        J_plot[0, :] = np.mean(J_plot[0, :])

    # Periodic padding in α for smooth spline
    n_pad = 3
    alphas_padded = np.concatenate([
        alphas[-n_pad:] - 2 * np.pi, alphas, alphas[:n_pad] + 2 * np.pi,
    ])
    J_padded = np.column_stack([
        J_plot[:, -n_pad:], J_plot, J_plot[:, :n_pad],
    ])

    spline = spi.RectBivariateSpline(s_plot, alphas_padded, J_padded)
    s_fine = np.linspace(0.0, s_plot[-1], 200)
    alpha_fine = np.linspace(0.0, 2 * np.pi, 360)
    J_fine = spline(s_fine, alpha_fine)

    R, Theta = np.meshgrid(s_fine, alpha_fine, indexing="ij")
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_rmin(0)
    cp = ax.contourf(Theta, R, J_fine, levels=30, cmap="inferno")
    ax.contour(Theta, R, J_fine, levels=15, colors="white",
               linewidths=0.5, alpha=0.6)
    s_ticks = s_vals[s_vals > 0.05]
    ax.set_rticks(s_ticks)
    ax.set_yticklabels([f"{v:.1f}" for v in s_ticks], fontsize=7, color="gray")
    ax.grid(True, alpha=0.25, color="gray", ls="--")
    cbar = fig.colorbar(cp, ax=ax, shrink=0.7, pad=0.1,
                        orientation="horizontal")
    cbar.set_label(r"$\tilde{J}$", fontsize=14)
    depth_pct = int(lambda_N * 100)
    ax.set_title(
        f"Contours of $\\tilde{{J}}$  —  "
        f"{depth_pct}% most deeply trapped\n"
        r"$(B^* \!-\! B_{\min})/(B_{\max} \!-\! B_{\min})"
        f" = {lambda_N}$",
        pad=20, fontsize=13,
    )
    plt.tight_layout()
    return fig


def plot_gradient_diagnostics(vmec, s_vals=None, num_alpha=8,
                              num_pitch=50, T_J=-0.06, mpol=8, ntor=8):
    """
    Plot d_s J vs B* for each radial interval.

    Provides visual confirmation of the max-J property.
    """
    if s_vals is None:
        s_vals = np.linspace(0.2, 0.8, 5)
    alphas = np.linspace(0, 2 * np.pi, num_alpha, endpoint=False)

    _, surface_data = run_boozer(vmec, s_vals, mpol=mpol, ntor=ntor)
    ns = len(s_vals)

    all_bs, all_jc = [], []
    for k in range(ns):
        data = surface_data[k]
        B_stars = np.linspace(data["B_min"], data["B_max"], num_pitch + 2)[1:-1]
        jc_rows = []
        for alpha in alphas:
            zeta, B_I = _extract_field_line(data, alpha)
            B_C = squash_and_stretch_simple(zeta, B_I, data["B_min"], data["B_max"])
            jc_rows.append(_compute_J_C(zeta, B_I, B_C, B_stars))
        all_bs.append(B_stars)
        all_jc.append(np.array(jc_rows).T)

    b_lo = max(b[0] for b in all_bs)
    b_hi = min(b[-1] for b in all_bs)
    if b_lo >= b_hi:
        print("  Cannot plot: no common B* range.")
        return None

    common_B = np.linspace(b_lo, b_hi, num_pitch)
    JC_int = []
    for k in range(ns):
        jc_k = np.zeros((num_pitch, num_alpha))
        for a in range(num_alpha):
            jc_k[:, a] = np.interp(common_B, all_bs[k], all_jc[k][:, a])
        JC_int.append(jc_k)

    n_intervals = ns - 1
    fig, axes = plt.subplots(1, n_intervals, figsize=(5 * n_intervals, 4), squeeze=False)
    for k in range(n_intervals):
        ax = axes[0, k]
        ds = s_vals[k + 1] - s_vals[k]
        J_lo, J_hi = JC_int[k], JC_int[k + 1]
        denom = float(np.mean(J_hi + J_lo))
        if abs(denom) < 1e-30:
            denom = 1e-30
        diff = J_hi[:, :, None] - J_lo[:, None, :]
        mean_dJ = (diff / (ds * denom)).mean(axis=-1)

        for a in range(num_alpha):
            ax.plot(common_B, mean_dJ[:, a], alpha=0.5)

        ax.axhline(T_J, color="r", ls="--", label=f"T_J={T_J}")
        ax.axhline(0, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("B*")
        ax.set_ylabel(r"$\langle \partial_s \tilde{J} \rangle_\alpha$")
        ax.set_title(f"s=[{s_vals[k]:.2f}, {s_vals[k + 1]:.2f}]")
        ax.legend()

    plt.suptitle("Max-J gradient diagnostics")
    plt.tight_layout()
    return fig


def plot_squid_core_diagnostics(info, metadata=None):
    """Compact 2x2 summary figure for the core SQuID metrics."""
    s_vals = np.asarray(info["s_vals"])
    interval_centers = np.asarray(info["interval_centers"])
    lambda_grid = np.asarray(info["lambda_grid"])
    if len(lambda_grid) > 1:
        lam_min, lam_max = lambda_grid[0], lambda_grid[-1]
    else:
        lam_min, lam_max = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(s_vals, info["qi_surface_rms"], marker="o", lw=2, label="QI surface RMS")
    ax.plot(s_vals, info["qi_surface_p95"], marker="s", lw=1.5, ls="--",
            label="QI |error| p95")
    worst_idx = int(info["qi_worst_surface_idx"])
    ax.axvline(s_vals[worst_idx], color="tab:red", ls=":", alpha=0.5)
    ax.set_xlabel("s")
    ax.set_ylabel("Normalised QI error")
    ax.set_title("QI surface summaries")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if len(interval_centers) > 0:
        ax.plot(interval_centers, info["maxj_interval_pass_ratio"], marker="o",
                lw=2, label="Pass ratio")
        ax.plot(interval_centers, info["maxj_interval_violation_fraction"],
                marker="s", lw=1.5, ls="--", label="Violation fraction")
        worst_interval = int(info["maxj_worst_interval_idx"])
        ax.axvline(interval_centers[worst_interval], color="tab:red",
                   ls=":", alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Interval center s")
    ax.set_ylabel("Fraction")
    ax.set_title("max-J interval summaries")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    im_qi = ax.imshow(
        info["qi_lambda_rms"],
        aspect="auto",
        origin="lower",
        extent=[lam_min, lam_max,
                s_vals[0] - 0.5 if len(s_vals) == 1 else s_vals[0],
                s_vals[-1] + 0.5 if len(s_vals) == 1 else s_vals[-1]],
        cmap="magma",
    )
    ax.set_xlabel(r"Normalised trapped depth $\lambda_N$")
    ax.set_ylabel("s")
    ax.set_title("QI RMS over trapped depth")
    fig.colorbar(im_qi, ax=ax, fraction=0.046, pad=0.04, label="RMS")

    ax = axes[1, 1]
    if len(interval_centers) > 0:
        im_mj = ax.imshow(
            info["maxj_lambda_violation"],
            aspect="auto",
            origin="lower",
            extent=[lam_min, lam_max,
                    interval_centers[0] - 0.5 if len(interval_centers) == 1 else interval_centers[0],
                    interval_centers[-1] + 0.5 if len(interval_centers) == 1 else interval_centers[-1]],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(im_mj, ax=ax, fraction=0.046, pad=0.04,
                     label="Violation fraction")
    else:
        im_mj = ax.imshow(np.zeros((1, 1)), aspect="auto", origin="lower",
                          cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(im_mj, ax=ax, fraction=0.046, pad=0.04,
                     label="Violation fraction")
    ax.set_xlabel(r"Representative trapped depth $\lambda_N$")
    ax.set_ylabel("Interval center s")
    ax.set_title("max-J violation map")

    title = "Core SQuID diagnostics"
    if metadata:
        title += (
            f"\nns={metadata.get('num_surfaces', 'n/a')}, "
            f"nalpha={metadata.get('num_alpha', 'n/a')}, "
            f"npitch={metadata.get('num_pitch', 'n/a')}"
        )
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_transport_diagnostics(itg_info=None, ae_info=None, itg_method="vacuum_dBds",
                               metadata=None):
    """Compact summary figure for extended transport diagnostics."""
    has_itg = itg_info is not None and len(itg_info.get("per_surface", {})) > 0
    ae_surface_keys = []
    if ae_info is not None:
        ae_surface_keys = [
            k for k in ae_info.keys()
            if k != "total" and np.isscalar(k)
        ]
    has_ae = len(ae_surface_keys) > 0

    if not has_itg and not has_ae:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    if has_itg:
        s_itg = np.array(sorted(itg_info["per_surface"].keys()), dtype=float)
        y_itg = np.array([itg_info["per_surface"][float(s)] for s in s_itg], dtype=float)
        ax.plot(s_itg, y_itg, marker="o", lw=2, color="tab:blue")
        if np.any(np.isfinite(y_itg)):
            worst_idx = int(np.nanargmax(y_itg))
            ax.axvline(s_itg[worst_idx], color="tab:red", ls=":", alpha=0.5)
        ax.set_title(f"ITG proxy ({itg_method})")
        ax.set_ylabel(r"$f_{\nabla s}$")
    else:
        ax.text(0.5, 0.5, "ITG not computed", ha="center", va="center")
        ax.set_title("ITG proxy")
    ax.set_xlabel("s")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if has_ae:
        s_ae = np.array(sorted(float(k) for k in ae_surface_keys), dtype=float)
        y_ae = np.array([ae_info[float(s)] for s in s_ae], dtype=float)
        ax.plot(s_ae, y_ae, marker="o", lw=2, color="tab:green")
        if np.any(np.isfinite(y_ae)):
            worst_idx = int(np.nanargmax(y_ae))
            ax.axvline(s_ae[worst_idx], color="tab:red", ls=":", alpha=0.5)
        ax.set_title("Available Energy")
        ax.set_ylabel("AE")
    else:
        ax.text(0.5, 0.5, "AE not computed", ha="center", va="center")
        ax.set_title("Available Energy")
    ax.set_xlabel("s")
    ax.grid(True, alpha=0.3)

    title = "Extended transport diagnostics"
    if metadata:
        title += (
            f"\nITG s={metadata.get('itg_surfaces', 'n/a')}, "
            f"AE s={metadata.get('ae_surfaces', 'off')}"
        )
    fig.suptitle(title)
    plt.tight_layout()
    return fig
