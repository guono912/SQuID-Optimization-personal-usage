"""
Diagnostic evaluation for SQuID equilibria.

Merged from:
  - max_J_evaluation.py  (evaluate_max_J, plot_gradient_diagnostics,
                           plot_J_contours)
  - SQuID_QI.py  (compute_SQuID_QI, plot_B_contours)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..core.boozer_utils import run_boozer, reconstruct_B, get_iota
from ..core.squash_stretch import squash_and_stretch_simple
from ..objectives.maxj_residual import (
    _extract_field_line, _compute_J_C, _compute_J_I, _evaluate_squid,
)
from ..objectives.itg_residual import ITGResidual


def evaluate_squid(vmec, s_vals=None, num_alpha=8, num_pitch=50,
                   T_J=-0.06, mboz=8, nboz=8, verbose=True):
    """
    Full SQuID diagnostic evaluation.

    Returns a dict with f_maxJ, f_QI, mirror_ratio, iota_axis, iota_edge,
    B_min, B_max.
    """
    if s_vals is None:
        s_vals = np.linspace(0.2, 0.8, 5)
    alphas = np.linspace(0, 2 * np.pi, num_alpha, endpoint=False)

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
