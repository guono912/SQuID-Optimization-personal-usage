"""SQuID configuration diagnostic report (reusable CLI implementation).

Figures (4 total, paper-quality):
  1. equilibrium_overview.png  — ι(s) + J_tor(s) + well(s) + p(s)   [2×2]
  2. mhd_stability.png         — DMerc(s) + ballooning λ + Newcomb [1×3]
  3. itg_flux_compression.png  — bad-curvature |∇s| proxy           [2×2]
  4. neoclassical_transport.png — ε_eff(s) from DESC + Boozer |B|   [1×2]
  5. j_contours_polar.png      — J|| in Boozer coords               [1×1]

JSON: scalar_summary.json — all metrics (geometry, MHD, transport, SQuID)

No mosaics. No residual bar charts. No table-as-image. No axis artifacts.

The scripts/viz/viz_report.py entry point and the legacy
scripts/viz_report.py wrapper both call into this module.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

from ..diagnostics.mercier_normalization import (
    mercier_profiles,
    mercier_summary,
    vmec_half_grid_profile,
)
from ..diagnostics.iota_rationals import _scan_iota_rationals

C = {
    "iota":   "#1f77b4", "mercier": "#d62728", "well":    "#2ca02c",
    "press":  "#9467bd", "current": "#8c564b", "ripple":  "#17becf",
    "balloon":"#e377c2", "zero":    "#888888", "danger":  "#e41a1c",
    "nfp_danger": "#ff4444",
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _nfp_dangerous_rationals(nfp, lo=0.55, hi=1.05, m_max=20, k_max=4):
    lo, hi = sorted((float(lo), float(hi)))
    rats = []
    for sign in (-1, 1):
        for k in range(1, k_max + 1):
            for m in range(1, m_max + 1):
                signed_k = sign * k
                iota = signed_k * nfp / m
                if lo <= iota <= hi:
                    rats.append((iota, m, signed_k))
    rats.sort(key=lambda x: (x[1], x[0]))
    return rats


def _field_scale_from_flux(phi_edge, minor_radius):
    """ISS04-style positive field scale from signed toroidal flux."""
    minor_radius = float(minor_radius)
    if not np.isfinite(minor_radius) or minor_radius <= 0.0:
        raise ValueError("minor radius must be finite and positive")
    return float(abs(phi_edge) / (np.pi * minor_radius**2))


def _iss04_field_scale(wout):
    """Use the same ISS04 field convention as the canonical diagnostic CLI."""
    try:
        bvco = np.asarray(wout.bvco, dtype=float).reshape(-1)
        indices = np.flatnonzero(np.isfinite(bvco) & (np.abs(bvco) > 1.0e-30))
        rmajor = float(wout.Rmajor_p)
        if not indices.size or not np.isfinite(rmajor) or abs(rmajor) <= 1.0e-30:
            raise ValueError("bvco or major radius unavailable")
        return float(abs(bvco[indices[0]]) / abs(rmajor)), "abs_bvco_first_nonzero_over_Rmajor"
    except (AttributeError, TypeError, ValueError):
        return (
            _field_scale_from_flux(wout.phi[-1], wout.Aminor_p),
            "abs_Phi_edge_over_pi_a2_fallback",
        )


def _axis_field_summary(wout, fallback):
    """Reconstruct |B| along the magnetic axis from VMEC Nyquist modes."""
    try:
        bmnc = np.asarray(wout.bmnc, dtype=float)
        xm = np.asarray(wout.xm_nyq, dtype=float).reshape(-1)
        xn = np.asarray(wout.xn_nyq, dtype=float).reshape(-1)
        if bmnc.ndim == 2 and bmnc.shape[1] == xm.size:
            coeff = bmnc[0]
        elif bmnc.ndim == 2 and bmnc.shape[0] == xm.size:
            coeff = bmnc[:, 0]
        else:
            coeff = bmnc.reshape(-1)
        if coeff.size != xm.size or coeff.size != xn.size:
            raise ValueError("inconsistent VMEC Nyquist arrays")
        zeta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
        field = np.sum(coeff[:, None] * np.cos(-xn[:, None] * zeta), axis=0)
        if not np.all(np.isfinite(field)) or np.min(field) <= 0.0:
            raise ValueError("invalid reconstructed axis field")
        return float(np.mean(field)), float(np.min(field)), float(np.max(field))
    except (AttributeError, IndexError, TypeError, ValueError):
        value = float(abs(getattr(wout, "b0", fallback)))
        return value, value, value


def _compute_newcomb_metric(eq_desc, rho_v, alpha_v=None, nzeta=81):
    """Compute DESC Newcomb ballooning metric on a rho grid.

    Positive values indicate stability; negative values indicate instability.
    The DESC compute function requires a field-line grid in (rho, alpha, zeta)
    with a source grid defined over one field period.
    """
    try:
        from desc.grid import Grid, LinearGrid

        rho_v = np.asarray(rho_v, dtype=float)
        if alpha_v is None:
            alpha_v = np.linspace(0.0, np.pi, 8, endpoint=False)
        alpha_v = np.asarray(alpha_v, dtype=float)
        nfp = int(eq_desc.NFP)
        zeta_v = np.linspace(0.0, 2.0 * np.pi / nfp, int(nzeta),
                             endpoint=False)
        source = LinearGrid(rho=rho_v, theta=alpha_v, zeta=zeta_v,
                            NFP=nfp, sym=False, endpoint=False)
        rr, aa, zz = np.meshgrid(rho_v, alpha_v, zeta_v, indexing="ij")
        nodes = np.column_stack([rr.ravel(), aa.ravel(), zz.ravel()])
        grid = Grid(nodes, coordinates="raz", NFP=nfp, source_grid=source,
                    sort=False, is_meshgrid=True)
        data = eq_desc.compute(["Newcomb ballooning metric"], grid=grid)
        metric = np.asarray(data["Newcomb ballooning metric"], dtype=float).ravel()
        if metric.size != rho_v.size:
            raise ValueError(
                "DESC returned an unexpected Newcomb profile shape: "
                f"expected {rho_v.size}, got {metric.size}"
            )
        return metric, None
    except Exception as exc:
        return np.full(len(rho_v), np.nan), str(exc)


def _itg_flux_compression_data(vmec, s_vals, nalpha=36, ntheta=181):
    """Bad-curvature flux-compression proxy used for ITG visualization.

    xi = (a * H(bad curvature) * |grad s|)^2. Lower radial summaries and
    fewer localized hot spots are preferred. This is a proxy, not GK heat flux.
    """
    from simsopt.mhd.vmec_diagnostics import vmec_fieldlines

    alpha = np.linspace(0.0, 2.0 * np.pi, int(nalpha), endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    a_min = float(vmec.wout.Aminor_p)
    edge_toroidal_flux = float(vmec.wout.phipf[-1])
    sign = np.sign(edge_toroidal_flux) if edge_toroidal_flux != 0 else 1.0

    rows = []
    fields = {}
    for s_val in np.asarray(s_vals, dtype=float):
        try:
            data = vmec_fieldlines(vmec, float(s_val), alpha, theta1d=theta)
            kappa = np.asarray(data.B_cross_kappa_dot_grad_alpha, dtype=float) * sign
            bad = kappa < 0
            grad_s = np.sqrt(np.maximum(
                np.asarray(data.grad_s_dot_grad_s, dtype=float), 0.0))
            xi = (a_min * grad_s) ** 2
            xi_bad = np.where(bad, xi, np.nan)
            vals = xi_bad[np.isfinite(xi_bad)]
            if vals.size:
                p50 = float(np.nanpercentile(vals, 50))
                p95 = float(np.nanpercentile(vals, 95))
                vmax = float(np.nanmax(vals))
                proxy = float(np.nanmean(vals * np.maximum(p95 - vals, 0.0)))
                bad_fraction = float(np.mean(bad))
            else:
                p50 = p95 = vmax = proxy = bad_fraction = 0.0
                xi_bad = np.full_like(xi, np.nan)
            rows.append({
                "s": float(s_val),
                "p50": p50,
                "p95": p95,
                "max": vmax,
                "proxy": proxy,
                "bad_fraction": bad_fraction,
            })
            fields[float(s_val)] = xi_bad
        except Exception as exc:
            rows.append({
                "s": float(s_val),
                "p50": np.nan,
                "p95": np.nan,
                "max": np.nan,
                "proxy": np.nan,
                "bad_fraction": np.nan,
                "error": str(exc),
            })
            fields[float(s_val)] = None
    return {
        "alpha": alpha,
        "theta": theta,
        "summary": rows,
        "fields": fields,
        "definition": "xi=(a*H(bad curvature)*|grad s|)^2",
        "note": "Bad-curvature flux-compression proxy; lower values and fewer hot spots are preferred.",
    }


def _compute_desc_equilibrium(nc_file):
    """Load VMEC into DESC at L=6,M=6,N=6. No force balance solve."""
    from desc.vmec import VMECIO
    return VMECIO.load(nc_file, L=6, M=6, N=6)


def _compute_effective_ripple(eq, s_vals):
    """DESC effective ripple ε_eff via Nemov bounce-integral formula."""
    from desc.grid import LinearGrid
    from desc.objectives import EffectiveRipple
    results = []
    for s in s_vals:
        try:
            rho = np.array([np.sqrt(s)])
            grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid,
                             NFP=int(eq.NFP), sym=False)
            obj = EffectiveRipple(eq, grid=grid, num_transit=10, num_pitch=31)
            obj.build(verbose=0)
            eps = float(obj.compute(eq.params_dict)[0])
            results.append(eps if np.isfinite(eps) else np.nan)
        except Exception:
            results.append(np.nan)
    return np.array(results)


def _compute_desc_toroidal_current(eq, s_vals):
    """Toroidal current profile from DESC — I_tor(s) in Amperes (on flux surfaces)."""
    from desc.grid import LinearGrid
    rho = np.sqrt(np.asarray(s_vals, dtype=float))
    grid = LinearGrid(rho=rho, M=0, N=0, NFP=int(eq.NFP))
    data = eq.compute(["current"], grid=grid)
    cur = np.asarray(data["current"], dtype=float)
    return cur


def _compute_desc_volume_beta(eq):
    """Volume-averaged beta <β>_vol from DESC."""
    data = eq.compute(["<beta>_vol"])
    return float(data["<beta>_vol"])


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Equilibrium Overview
# ═══════════════════════════════════════════════════════════════════

def plot_equilibrium_overview(vmec, eq_desc, s_gate=0.05, output_path=None):
    """2×2: iota + toroidal current + magnetic well + pressure."""
    wout = vmec.wout
    nfp = int(wout.nfp)
    ns = int(wout.ns)
    s_full = np.linspace(0, 1, ns)

    # iota
    s_iota = np.linspace(0, 1, len(wout.iotaf))

    # current from DESC
    s_cur = np.linspace(s_gate, 0.99, 20)
    jtor = _compute_desc_toroidal_current(eq_desc, s_cur)

    # well
    vp = np.array(wout.vp)
    well_prof = (vp[1] - vp) / max(abs(vp[1]), 1e-30)

    # pressure
    presf = np.array(wout.presf)
    s_pres = np.linspace(0, 1, len(presf))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Equilibrium Overview", fontsize=14, fontweight="bold")

    # (a) Iota
    ax = axes[0, 0]
    iotaf = np.array(wout.iotaf)
    ax.plot(s_iota, iotaf, color=C["iota"], lw=2.0)
    lo_i = float(iotaf[s_iota >= s_gate].min()) - 0.03
    hi_i = float(iotaf[s_iota <= 0.95].max()) + 0.03
    for iota_val, m, k in _nfp_dangerous_rationals(nfp, lo_i, hi_i, m_max=16):
        label = f"{Fraction(int(k*nfp), m)} (m={m})" if m <= 11 else None
        if m <= 6:
            ls, lw, a = "--", 1.5, 0.9
        elif m <= 11:
            ls, lw, a = "--", 1.2, 0.7
        else:
            ls, lw, a = ":", 0.6, 0.4
        ax.axhline(iota_val, ls=ls, lw=lw, color=C["nfp_danger"], alpha=a)
        if label and lo_i <= iota_val <= hi_i:
            ax.text(0.95, iota_val, label, ha="right", va="center",
                    fontsize=7, color=C["nfp_danger"],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    for rat in sorted(set(Fraction(p, q) for q in range(1, 7)
                           for p in range(1, int(hi_i*q)+1)
                           if lo_i <= p/q <= hi_i), key=float):
        ax.axhline(float(rat), ls=":", lw=0.3, color="#aaaaaa", alpha=0.2)
    ax.set_xlabel("s"); ax.set_ylabel(r"$\iota$")
    ax.set_xlim(s_gate, 1.0); ax.set_ylim(lo_i, hi_i)
    ax.grid(True, alpha=0.25)
    ax.set_title(r"Rotational Transform $\iota(s)$  —  nfp=4 coupled rationals labeled")

    # (b) Toroidal current (DESC)
    ax = axes[0, 1]
    jtor_ka = jtor / 1e3
    ax.plot(s_cur, jtor_ka, color=C["current"], lw=2.0)
    ax.axhline(0, color=C["zero"], ls="-", lw=0.8)
    ax.fill_between(s_cur, 0, jtor_ka, where=(jtor_ka > 0),
                    color=C["current"], alpha=0.12)
    ax.fill_between(s_cur, 0, jtor_ka, where=(jtor_ka < 0),
                    color="red", alpha=0.2)
    ax.set_xlabel("s"); ax.set_ylabel(r"$I_{\rm tor}$ [kA]")
    ax.set_xlim(s_gate, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Toroidal Current Profile (DESC)")

    # (c) Magnetic well
    ax = axes[1, 0]
    mask = (s_full >= s_gate) & np.isfinite(well_prof)
    ax.plot(s_full[mask], well_prof[mask]*100, color=C["well"], lw=2.0)
    ax.axhline(0, color=C["zero"], ls="-", lw=0.8)
    ax.fill_between(s_full[mask], 0, well_prof[mask]*100,
                    where=well_prof[mask] > 0, color=C["well"], alpha=0.15)
    ax.fill_between(s_full[mask], 0, well_prof[mask]*100,
                    where=well_prof[mask] < 0, color="red", alpha=0.3)
    ax.set_xlabel("s"); ax.set_ylabel("Well Depth [%]")
    ax.set_xlim(s_gate, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Magnetic Well  (edge = {float(well_prof[-1])*100:+.2f}%)")

    # (d) Pressure
    ax = axes[1, 1]
    mask_p = (s_pres >= s_gate) & np.isfinite(presf)
    ax.plot(s_pres[mask_p], presf[mask_p]/1e3, color=C["press"], lw=2.0)
    ax.set_xlabel("s"); ax.set_ylabel("Pressure [kPa]")
    ax.set_xlim(s_gate, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Pressure Profile")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# Figure 2: MHD Stability
# ═══════════════════════════════════════════════════════════════════

def plot_mhd_stability(vmec, eq_desc, nfp, s_gate=0.05, output_path=None,
                       gate_ballooning=None):
    """Mercier DMerc(s), DESC ballooning λ, and Newcomb metric.

    If gate_ballooning is provided (from a DESC force-balanced gate JSON),
    use those ballooning eigenvalues instead of computing on the raw fit.
    """
    from desc.objectives import BallooningStability

    wout = vmec.wout
    mercier = mercier_profiles(wout)
    dmerc_full = mercier["profiles"]["DMerc"]["flux_normalized"]
    s_dm, dmerc = vmec_half_grid_profile(dmerc_full)

    # ballooning: prefer force-balanced gate data when available
    if gate_ballooning is not None:
        rho_v = np.array(gate_ballooning["rho"])
        lam_grid = np.array(gate_ballooning["lam_grid"])
        n_pos = int(gate_ballooning["n_unstable"])
        lam_max = float(gate_ballooning["lam_max"])
        lam_min = float(gate_ballooning["lam_min"])
        lam_env = np.max(lam_grid, axis=1)
        lam_best = np.min(lam_grid, axis=1)
        source_note = " (DESC force-balanced)"
    else:
        rho_v = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95])
        alpha_v = np.linspace(0, np.pi, 8, endpoint=False)
        zeta0 = np.linspace(-0.5*np.pi, 0.5*np.pi, 5)
        shift = 1.0

        lam_grid = np.zeros((len(rho_v), len(alpha_v)))
        for j, a in enumerate(alpha_v):
            obj = BallooningStability(
                eq_desc, rho=rho_v, alpha=np.array([a]),
                nturns=2, nzetaperturn=80, zeta0=zeta0,
                Neigvals=1, lambda0=-shift, w0=0.0, w1=1.0,
            )
            obj.build(use_jit=False, verbose=0)
            val = np.asarray(obj.compute(eq_desc.params_dict), dtype=float)
            lam_grid[:, j] = val.ravel()[:len(rho_v)] - shift

        n_pos = int(np.sum(lam_grid > 0))
        lam_max = float(np.max(lam_grid))
        lam_min = float(np.min(lam_grid))
        lam_env = np.max(lam_grid, axis=1)
        lam_best = np.min(lam_grid, axis=1)
        source_note = " (raw VMEC→DESC fit — no force balance)"

    newcomb, newcomb_error = _compute_newcomb_metric(eq_desc, rho_v)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("MHD Stability", fontsize=14, fontweight="bold")

    # (a) Mercier
    ax = axes[0]
    mask = (s_dm >= s_gate) & np.isfinite(dmerc)
    ax.plot(s_dm[mask], dmerc[mask], color=C["mercier"], lw=2.0)
    ax.axhline(0, color=C["zero"], ls="-", lw=0.8)
    ax.fill_between(s_dm[mask], 0, dmerc[mask],
                    where=dmerc[mask]>0, color=C["mercier"], alpha=0.15)
    ax.fill_between(s_dm[mask], 0, dmerc[mask],
                    where=dmerc[mask]<0, color="red", alpha=0.3)
    yv = dmerc[mask]
    yl = max(abs(yv.min()), abs(yv.max())) * 1.2
    if np.isfinite(yl) and yl > 1e-6:
        ax.set_ylim(-yl, yl)
    ax.set_xlabel("s"); ax.set_ylabel(r"$\Phi_{edge}^2 D_{\rm Merc}$")
    ax.set_xlim(s_gate, 1.0)
    ax.grid(True, alpha=0.25)
    neg_in_gate = int(np.sum(dmerc[mask] < 0))
    ax.set_title(f"Mercier Criterion  (negative points in [{s_gate},1]: {neg_in_gate})")

    # (b) Ballooning envelope: λ band vs ρ (showing full range, not just ReLU)
    ax = axes[1]
    ax.fill_between(rho_v, lam_best, lam_env, color=C["balloon"], alpha=0.12)
    ax.plot(rho_v, lam_env, "o-", color="#d62728", lw=2.0, ms=6,
            label=r"$\max_\alpha\lambda$ (worst)")
    ax.plot(rho_v, lam_best, "s--", color="#2ca02c", lw=1.2, ms=4,
            label=r"$\min_\alpha\lambda$ (best)")
    for i, r in enumerate(rho_v):
        n_a = int(np.sum(lam_grid[i,:] > 0))
        if n_a > 0:
            ax.annotate(f"{n_a}α>0", (r, lam_env[i]),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=6, color="#d62728")
    ax.axhline(0, color=C["zero"], ls="-", lw=0.8)
    ax.set_xlabel(r"$\rho$ (normalized radius)")
    ax.set_ylabel(r"$\lambda$ (ballooning eigenvalue)")
    ax.set_xlim(0.05, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    status = "UNSTABLE" if n_pos > 0 else "STABLE"
    c = "#d62728" if n_pos > 0 else "#2ca02c"
    ax.set_title(f"DESC Ballooning — {status}  "
                 f"({n_pos} unstable, λ∈[{lam_min:.1e},{lam_max:.1e}]){source_note}",
                 color=c, fontsize=10)

    # (c) Newcomb metric: >0 stable, <0 unstable.
    ax = axes[2]
    if np.any(np.isfinite(newcomb)):
        ax.plot(rho_v, newcomb, "o-", color="#4c78a8", lw=2.0, ms=6,
                label="Newcomb metric")
        ax.fill_between(rho_v, 0, newcomb, where=newcomb >= 0,
                        color="#2ca02c", alpha=0.12)
        ax.fill_between(rho_v, 0, newcomb, where=newcomb < 0,
                        color="#d62728", alpha=0.20)
        ax.axhline(0, color=C["zero"], ls="-", lw=0.8)
        ax.set_xlim(0.05, 1.0)
        ax.set_xlabel(r"$\rho$ (normalized radius)")
        ax.set_ylabel("Newcomb ballooning metric")
        ax.grid(True, alpha=0.25)
        n_bad = int(np.sum(newcomb < 0))
        min_metric = float(np.nanmin(newcomb))
        title_color = "#d62728" if n_bad else "#2ca02c"
        ax.set_title(f"Newcomb Proxy — {'UNSTABLE' if n_bad else 'STABLE'} "
                     f"(min={min_metric:.2e})",
                     color=title_color, fontsize=10)
    else:
        ax.text(0.5, 0.5, "Newcomb metric unavailable",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        if newcomb_error:
            ax.text(0.5, 0.38, newcomb_error[:120],
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray", wrap=True)
        ax.set_axis_off()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig, {
        "n_unstable": n_pos,
        "lam_max": lam_max,
        "newcomb_rho": rho_v.tolist(),
        "newcomb_metric": newcomb.tolist(),
        "newcomb_min": float(np.nanmin(newcomb)) if np.any(np.isfinite(newcomb)) else np.nan,
        "newcomb_negative_count": int(np.sum(newcomb < 0)) if np.any(np.isfinite(newcomb)) else -1,
        "newcomb_error": newcomb_error,
    }


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Neoclassical Transport
# ═══════════════════════════════════════════════════════════════════

def plot_neoclassical(vmec, eq_desc, nfp, output_path=None):
    """ε_eff(s) from DESC + Boozer |B| contours."""
    from ..evaluation.evaluate import run_boozer, reconstruct_B

    s_ripple = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    eps_eff = _compute_effective_ripple(eq_desc, s_ripple)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Neoclassical Transport", fontsize=14, fontweight="bold")

    # (a) Effective ripple
    ax = axes[0]
    mask = np.isfinite(eps_eff)
    if np.any(mask):
        ax.semilogy(s_ripple[mask], eps_eff[mask], "o-", color=C["ripple"],
                    lw=2.0, ms=6)
        ax.set_ylabel(r"$\varepsilon_{\rm eff}$ (DESC Nemov)")
    else:
        ax.text(0.5, 0.5, "ε_eff unavailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
    ax.set_xlabel("s")
    ax.set_xlim(0.05, 0.95)
    ax.grid(True, alpha=0.25)
    ax.set_title("Effective Ripple (DESC bounce-integral)")

    # (b) Boozer |B| at θ=π cut — toroidal variation
    ax = axes[1]
    boozer_s = [0.25, 0.50, 0.75]
    safe_s = [max(0.01, min(0.99, s)) for s in boozer_s]
    _, all_surf = run_boozer(vmec, safe_s, mpol=20, ntor=20)

    nphi = 128
    ze = np.linspace(0, 2*np.pi/nfp, nphi)
    colors_b = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (s_val, data) in enumerate(zip(boozer_s, all_surf)):
        # Reconstruct B at θ=π only
        m_arr = np.array(data["m"])
        n_arr = np.array(data["n"])
        bmnc = np.array(data["bmnc"])
        B_cut = np.zeros(nphi)
        for j in range(len(m_arr)):
            B_cut += bmnc[j] * np.cos(m_arr[j]*np.pi - n_arr[j]*ze)
        ax.plot(ze * nfp / (2*np.pi), B_cut, color=colors_b[i], lw=1.8,
                label=f"s={s_val}")

    ax.set_xlabel(r"$\phi$ (normalized field period)")
    ax.set_ylabel(r"$|B|$ [T]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_title("|B| Toroidal Variation at θ=π")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig, eps_eff


# ═══════════════════════════════════════════════════════════════════
# Figure 4: ITG flux-compression proxy
# ═══════════════════════════════════════════════════════════════════

def plot_itg_flux_compression(vmec, output_path=None):
    """Bad-curvature flux-compression proxy for ITG screening."""
    s_vals = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    itg = _itg_flux_compression_data(vmec, s_vals)
    rows = itg["summary"]
    s = np.array([r["s"] for r in rows], dtype=float)
    p50 = np.array([r["p50"] for r in rows], dtype=float)
    p95 = np.array([r["p95"] for r in rows], dtype=float)
    vmax = np.array([r["max"] for r in rows], dtype=float)
    proxy = np.array([r["proxy"] for r in rows], dtype=float)
    bad_fraction = np.array([r["bad_fraction"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("ITG Bad-Curvature Flux-Compression Proxy",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(s, p50, "o-", lw=2, label="p50")
    ax.plot(s, p95, "s-", lw=2, label="p95")
    ax.plot(s, vmax, "^-", lw=1.5, label="max")
    ax.set_xlabel("s")
    ax.set_ylabel(r"$(a|\nabla s|)^2$ in bad curvature")
    ax.set_title("Flux-compression amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(s, proxy, "o-", color="tab:purple", lw=2, label="proxy")
    ax2 = ax.twinx()
    ax2.plot(s, bad_fraction, "s--", color="tab:orange", lw=1.5,
             label="bad-curvature fraction")
    ax.set_xlabel("s")
    ax.set_ylabel("proxy residual")
    ax2.set_ylabel("bad-curvature area fraction")
    ax.set_title("Radial ITG proxy and bad-curvature coverage")
    ax.grid(True, alpha=0.25)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines],
              fontsize=8, loc="best")

    finite_fields = [
        v for v in itg["fields"].values()
        if v is not None and np.any(np.isfinite(v))
    ]
    if finite_fields:
        flat = np.concatenate([np.ravel(v[np.isfinite(v)]) for v in finite_fields])
        vmax_heat = np.nanpercentile(flat, 98) if flat.size else np.nan
    else:
        vmax_heat = np.nan
    for ax, s_heat in zip(axes[1], [0.45, 0.65]):
        field = itg["fields"].get(float(s_heat))
        if field is None or not np.any(np.isfinite(field)):
            ax.text(0.5, 0.5, f"s={s_heat:.2f} unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray")
            ax.set_axis_off()
            continue
        im = ax.imshow(
            np.ma.masked_invalid(field).T,
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi],
            cmap="magma",
            vmin=0.0,
            vmax=vmax_heat if np.isfinite(vmax_heat) and vmax_heat > 0 else None,
            interpolation="nearest",
        )
        ax.set_xlabel(r"field-line label $\alpha$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(f"Bad-curvature hot spots at s={s_heat:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=r"$(a|\nabla s|)^2$")

    fig.text(
        0.5, 0.01,
        "Proxy only: lower values and fewer localized hot spots are preferred; this is not a gyrokinetic heat flux.",
        ha="center", fontsize=9, color="0.35",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    return fig, {
        "surfaces": s.tolist(),
        "p50": p50.tolist(),
        "p95": p95.tolist(),
        "max": vmax.tolist(),
        "proxy": proxy.tolist(),
        "bad_fraction": bad_fraction.tolist(),
        "definition": itg["definition"],
        "note": itg["note"],
    }


# ═══════════════════════════════════════════════════════════════════
# JSON report
# ═══════════════════════════════════════════════════════════════════

def _surface_area_for_report(wout, boundary_geometry):
    """Return the resolved LCFS area, with a circular-torus fallback."""
    resolved = boundary_geometry.get("surface_area_m2_simsopt")
    if resolved is not None and np.isfinite(resolved) and resolved > 0.0:
        return float(resolved), "simsopt_lcfs_quadrature"
    approximate = 4.0 * np.pi**2 * float(wout.Rmajor_p) * float(wout.Aminor_p)
    return float(approximate), "circular_torus_4pi2_Ra_fallback"


def build_scalar_json(nc_file, vmec, eq_desc, info, mercier_facts, well_facts,
                      ballooning_facts, eps_eff, itg_proxy=None, s_gate=0.05):
    """All metrics in one JSON — geometry, MHD, transport, SQuID."""
    wout = vmec.wout
    nfp = int(wout.nfp)
    ns = int(wout.ns)
    s_full = np.linspace(0, 1, ns)

    # ── B-axis (from toroidal flux / cross-section) ──
    B0_est = _field_scale_from_flux(wout.phi[-1], wout.Aminor_p)
    b_axis_mean, b_axis_min, b_axis_max = _axis_field_summary(wout, B0_est)

    # ── B0 (ISS04-compatible: toroidal flux / cross-section area) ──
    B0_iss04, B0_iss04_method = _iss04_field_scale(wout)

    # ── Volume-averaged beta (DESC) ──
    beta_vol_desc = _compute_desc_volume_beta(eq_desc)

    # ── Toroidal current ──
    try:
        from desc.grid import LinearGrid
        grid1 = LinearGrid(rho=np.array([1.0]), M=0, N=0, NFP=int(nfp))
        cur_data = eq_desc.compute(["current"], grid=grid1)
        Itor_edge = float(np.asarray(cur_data["current"]).ravel()[0])
    except Exception:
        Itor_edge = np.nan

    # ── Boundary geometry (from VMEC scalars — simpler & more robust) ──
    try:
        Rmax_bnd = float(wout.rmax_surf)
        Rmin_bnd = float(wout.rmin_surf)
        Zmax_bnd = float(wout.zmax_surf)
        Zmin_bnd = -Zmax_bnd  # stellarator-symmetric
    except Exception:
        Rmax_bnd = Rmin_bnd = Zmax_bnd = Zmin_bnd = np.nan

    # ── Boundary curvature geometry (simsopt LCFS) ──
    try:
        from simsopt.geo import SurfaceRZFourier
        from ..diagnostics.boundary_geometry import boundary_geometry_metrics
        from ..objectives.pdrot_residual import (
            pdrot_area_weighted_stats,
            principal_direction_rotation_rate,
        )
        boundary_geometry = boundary_geometry_metrics(
            nc_file, ntheta=128, nphi=128, torus_range="full torus"
        )
        pdrot_surface = SurfaceRZFourier.from_wout(
            nc_file, range="full torus", ntheta=128, nphi=128
        )
        pdrot_residual_area_weighted = pdrot_area_weighted_stats(
            principal_direction_rotation_rate(pdrot_surface)
        )
    except Exception as exc:
        boundary_geometry = {
            "source": "simsopt SurfaceRZFourier LCFS geometry",
            "error": str(exc),
        }
        pdrot_residual_area_weighted = {"error": str(exc)}

    surface_area, surface_area_source = _surface_area_for_report(
        wout, boundary_geometry
    )

    # ── LCFS |B| ──
    from ..evaluation.evaluate import run_boozer as _rb, reconstruct_B as _rcB
    _, surf_lcfs = _rb(vmec, [0.99], mpol=16, ntor=16)
    if surf_lcfs:
        d = surf_lcfs[0]
        B_lcfs = _rcB(d["m"], d["n"], d["bmnc"],
                               np.linspace(0, 2*np.pi, 64),
                               np.linspace(0, 2*np.pi/nfp, 64))
        Bmin_lcfs = float(np.min(B_lcfs))
        Bmax_lcfs = float(np.max(B_lcfs))
        Bavg_lcfs = float(np.mean(B_lcfs))
    else:
        Bmin_lcfs = Bmax_lcfs = Bavg_lcfs = np.nan

    # ── nfp dangerous rational check ──
    crossed = []
    iota_profile = np.asarray(wout.iotaf, dtype=float)
    s_iota = np.linspace(0.0, 1.0, iota_profile.size)
    iota_gate = iota_profile[(s_iota >= 0.05) & (s_iota <= 0.95)]
    iota_lo = float(np.nanmin(iota_gate))
    iota_hi = float(np.nanmax(iota_gate))
    for iota_val, m, k in _nfp_dangerous_rationals(
        nfp, iota_lo, iota_hi, m_max=16
    ):
        if iota_lo <= iota_val <= iota_hi:
            crossed.append({"fraction": f"{int(k*nfp)}/{m}", "value": iota_val,
                           "m": m, "k": k})
    rational_scan = _scan_iota_rationals(
        vmec, max_denominator=12, warn_distance=0.01, s_min=0.05, s_max=0.95
    )

    return {
        # Geometry
        "nfp": nfp,
        "R0_major_radius_m": float(wout.Rmajor_p),
        "a_minor_radius_m": float(wout.Aminor_p),
        "aspect_ratio": float(wout.aspect),
        "volume_m3": float(wout.volume_p),
        "surface_area_m2": surface_area,
        "surface_area_source": surface_area_source,
        "boundary_geometry_simsopt": boundary_geometry,
        "pdrot_residual_area_weighted": pdrot_residual_area_weighted,
        "boundary_R_max": Rmax_bnd,
        "boundary_R_min": Rmin_bnd,
        "boundary_Z_max": Zmax_bnd,
        "boundary_Z_min": Zmin_bnd,
        "elongation_kappa": float((Zmax_bnd - Zmin_bnd) / (Rmax_bnd - Rmin_bnd)),
        "cross_section_area_m2": float(np.pi * float(wout.Aminor_p)**2),

        # Magnetic field
        "B0_ISS04_T": B0_iss04,
        "B0_ISS04_extraction_method": B0_iss04_method,
        "B_axis_mean_T": b_axis_mean,
        "B_axis_min_T": b_axis_min,
        "B_axis_max_T": b_axis_max,
        "Bmin_lcfs_T": Bmin_lcfs,
        "Bmax_lcfs_T": Bmax_lcfs,
        "Bavg_lcfs_T": Bavg_lcfs,
        "mirror_ratio_lcfs": (Bmax_lcfs - Bmin_lcfs) / (Bmax_lcfs + Bmin_lcfs)
            if Bmax_lcfs > 1e-10 else np.nan,
        "Phi_toroidal_flux_Tm2": float(wout.phi[-1]),

        # iota
        "iota_axis": float(wout.iotaf[0]),
        "iota_edge": float(wout.iotaf[-1]),
        "iota_min": float(np.min(wout.iotaf)),
        "iota_max": float(np.max(wout.iotaf)),
        "iota_span": float(wout.iotaf[-1] - wout.iotaf[0]),
        "nfp_dangerous_rationals_crossed": crossed,
        "low_order_rational_crossings_q_le_12": rational_scan.get("crossings", []),
        "iota_sign_flip": rational_scan.get("sign_flip", False),
        "iota_zero_crossings": rational_scan.get("zero_crossings", []),

        # MHD stability
        "dmerc_convention": mercier_facts.get("convention"),
        "dmerc_formula": mercier_facts.get("formula"),
        "dmerc_edge_toroidal_flux_wb": mercier_facts.get("edge_toroidal_flux_wb"),
        "dmerc_vmec_raw_min_gated": mercier_facts.get("dmerc_vmec_raw_min"),
        "dmerc_flux_normalized_min_gated": mercier_facts.get("dmerc_flux_normalized_min"),
        "dmerc_negative_count_gated": mercier_facts.get("dmerc_negative_count", 0),
        "dmerc_s_min": mercier_facts.get("s_min", s_gate),
        "dmerc_s_max": mercier_facts.get("s_max", 0.97),
        "magnetic_well_edge_percent": float(well_facts["edge"]) * 100,
        "ballooning_n_unstable": ballooning_facts.get("n_unstable", -1),
        "ballooning_lambda_max": ballooning_facts.get("lam_max", -1),
        "newcomb_rho": ballooning_facts.get("newcomb_rho", []),
        "newcomb_metric": ballooning_facts.get("newcomb_metric", []),
        "newcomb_metric_min": ballooning_facts.get("newcomb_min", np.nan),
        "newcomb_metric_negative_count": ballooning_facts.get(
            "newcomb_negative_count", -1),
        "newcomb_metric_note": (
            "Positive Newcomb metric indicates ideal-ballooning stability; "
            "negative indicates instability."
        ),

        # Transport
        "eps_eff_DESC_Nemov": {f"s={s:.1f}": v if np.isfinite(v) else None
                               for s, v in zip(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            eps_eff)} if eps_eff is not None else {},
        "ITG_flux_compression_proxy": itg_proxy or {},

        # Current & beta
        "Itor_edge_A": Itor_edge,
        "beta_vol_averaged_DESC": beta_vol_desc,
        "beta_total_VMEC": float(wout.betatotal),

        # SQuID
        "mirror_ratio_core_SQuID": info["mirror_ratio"],
        "f_QI": info["f_QI"],
        "f_maxJ": info["f_maxJ"],
        "f_Bmin": info["f_Bmin"],
        "maxJ_global_pass_ratio": info["maxj_global_pass_ratio"],
        "maxJ_global_violation_fraction": info["maxj_global_violation_fraction"],
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def generate_report(nc_file, output_dir="viz", nfp=4, gate_json=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    s_gate = 0.10

    print(f"\n{'='*60}")
    print(f"SQuID Diagnostic Report")
    print(f"  Input:  {nc_file}")
    print(f"  Output: {output_dir}")
    if gate_json:
        print(f"  Gate:   {gate_json}")
    print(f"{'='*60}")

    # Load gate JSON for force-balanced ballooning & ripple
    gate_ballooning = None
    gate_ripple = None
    gate_desc = None
    if gate_json:
        with open(gate_json) as f:
            gj = json.load(f)
        gate_ballooning = gj.get("ballooning")
        gate_ripple = gj.get("ripple")
        gate_desc = gj.get("desc")
        print(f"  Gate verdict: {gj.get('verdict','?')}  "
              f"ballooning n_unstable={gate_ballooning.get('n_unstable','?')}")

    # VMEC
    from simsopt.mhd import Vmec
    vmec = Vmec(nc_file)
    vmec.run()
    wout = vmec.wout

    # Core SQuID
    from ..evaluation.evaluate import evaluate_squid_detailed
    info = evaluate_squid_detailed(
        vmec, s_vals=np.linspace(0.15, 0.85, 7),
        num_alpha=8, num_pitch=50, verbose=False,
    )
    print(f"  f_QI={info['f_QI']:.4e}  f_maxJ={info['f_maxJ']:.4e}  "
          f"f_Bmin={info['f_Bmin']:.4e}")

    # Mercier facts (exclude axis s<0.1 and boundary s>0.97)
    mercier_facts = mercier_summary(wout, s_min=s_gate, s_max=0.97)

    # Well facts
    vp = np.array(wout.vp)
    well_facts = {"edge": float((vp[1] - vp[-1]) / max(abs(vp[1]), 1e-30))}

    # DESC equilibrium (fit only, no force balance)
    print("\n--- DESC Fit ---")
    eq_desc = _compute_desc_equilibrium(nc_file)
    print(f"  L={eq_desc.L}, M={eq_desc.M}, N={eq_desc.N}")

    # Generate figures
    print("\n--- Figures ---")
    plot_equilibrium_overview(vmec, eq_desc, s_gate=s_gate,
                              output_path=str(output_dir/"equilibrium_overview.png"))

    fig2, ballooning_facts = plot_mhd_stability(
        vmec, eq_desc, nfp=nfp, s_gate=s_gate,
        output_path=str(output_dir/"mhd_stability.png"),
        gate_ballooning=gate_ballooning)

    fig3, eps_eff = plot_neoclassical(
        vmec, eq_desc, nfp=nfp,
        output_path=str(output_dir/"neoclassical_transport.png"))

    fig_itg, itg_proxy = plot_itg_flux_compression(
        vmec, output_path=str(output_dir/"itg_flux_compression.png"))

    # J contours
    from ..evaluation.evaluate import plot_J_contours
    fig4 = plot_J_contours(vmec, lambda_N=0.3)
    jc_path = str(output_dir/"j_contours_polar.png")
    fig4.savefig(jc_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {jc_path}")

    # Boozer surface (classic style, 4-panel)
    from ..evaluation.evaluate import run_boozer, reconstruct_B
    boozer_s = [0.25, 0.5, 0.75, 1.0]
    safe_s = [max(0.01, min(0.99, s)) for s in boozer_s]
    _, all_surf = run_boozer(vmec, safe_s, mpol=20, ntor=20)

    th = np.linspace(0, 2*np.pi, 100)
    ze = np.linspace(0, 2*np.pi/nfp, 100)
    TH, ZE = np.meshgrid(th, ze, indexing="ij")
    B_all = [reconstruct_B(d["m"], d["n"], d["bmnc"], TH, ZE) for d in all_surf]

    fig5, axes5 = plt.subplots(2, 2, figsize=(9, 6.5))
    for i, (s_val, B2d) in enumerate(zip(boozer_s, B_all)):
        ax = axes5.flat[i]
        cs = ax.contour(ZE, TH, B2d, levels=np.linspace(B2d.min(), B2d.max(), 20),
                        cmap="plasma", linewidths=1.0)
        ax.text(0.03, 0.95, f"|B| @ s={s_val:g}" if s_val == 1.0 else f"|B| @ s={s_val}",
                transform=ax.transAxes, fontsize=10, fontweight="bold", va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        ax.set_xticks([0, 2*np.pi/nfp])
        ax.set_xticklabels(["0", r"$2\pi/N$" if nfp != 4 else r"$\pi/2$"])
        ax.set_yticks([0, 2*np.pi])
        ax.set_yticklabels(["0", r"$2\pi$"])
        ax.set_xlabel(r"$\phi$", fontweight="bold")
        ax.set_ylabel(r"$\theta$", fontweight="bold")
        fig5.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    fig5.suptitle("Boozer |B| Contours", fontsize=14, fontweight="bold")
    fig5.tight_layout()
    boozer_path = str(output_dir/"boozer_surface.png")
    fig5.savefig(boozer_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {boozer_path}")

    # JSON
    scalar = build_scalar_json(nc_file, vmec, eq_desc, info, mercier_facts, well_facts,
                               ballooning_facts, eps_eff,
                               itg_proxy=itg_proxy, s_gate=s_gate)
    json_path = output_dir / "scalar_summary.json"
    with open(json_path, "w") as f:
        json.dump(scalar, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else str(x))
    print(f"  Saved: {json_path}")

    # ── 3D boundary plot (DESC plotly, interactive HTML) ──
    print("\n--- 3D Boundary ---")
    try:
        from desc.plotting import plot_3d
        eq_3d = _compute_desc_equilibrium(nc_file)
        fig3d = plot_3d(eq_3d, "|B|", log=False)
        if isinstance(fig3d, tuple):
            fig3d = fig3d[0]
        b_status = "STABLE" if ballooning_facts.get("n_unstable", 1) == 0 else f"{ballooning_facts.get('n_unstable', '?')} unstable"
        fig3d.update_layout(
            title=dict(
                text=f"SQuID  —  nfp={nfp}  A={float(wout.aspect):.2f}  "
                     f"\u03b2={scalar.get('beta_total_VMEC', 0)*100:.2f}%  "
                     f"\u03b9=[{float(wout.iotaf[0]):.3f},{float(wout.iotaf[-1]):.3f}]  "
                     f"ballooning {b_status}",
                font=dict(size=12),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        html3d_path = output_dir / "boundary_3d.html"
        fig3d.write_html(str(html3d_path))
        print(f"  Saved: {html3d_path}")
    except Exception as exc:
        print(f"  [3D plot failed: {exc}]")

    plt.close("all")
    print(f"\n{'='*60}\nReport complete.\n{'='*60}\n")
    return scalar

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_file", required=True)
    parser.add_argument("--output_dir", default="runs/viz_latest")
    parser.add_argument("--nfp", type=int, default=4)
    parser.add_argument("--gate_json", type=str, default=None,
                        help="DESC force-balanced gate JSON for accurate ballooning data")
    args = parser.parse_args(argv)
    generate_report(args.nc_file, output_dir=args.output_dir, nfp=args.nfp,
                    gate_json=args.gate_json)
    return 0
