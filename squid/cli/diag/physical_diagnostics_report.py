#!/usr/bin/env python3
"""Generate physics-first diagnostic plots for a VMEC wout.

Reusable CLI implementation for the physics-first diagnostic report. The
scripts/diag/physical_diagnostics_report.py entry point and the legacy
scripts/physical_diagnostics_report.py wrapper both call into this module.

This complements scripts/diagnose.py.  It avoids the older SQuID target
summary plots and focuses on MHD, iota/rational placement, DESC ballooning,
effective ripple, and current/beta facts.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from ...diagnostics.mercier_normalization import (
    mercier_profiles,
    vmec_half_grid_profile,
)
from ...utils.jax_runtime import initialize_requested_jax_backend


def _scalar(ds, name, default=np.nan):
    if name not in ds.variables:
        return default
    arr = np.asarray(ds.variables[name][:])
    return float(arr.reshape(-1)[0])


def _array(ds, name):
    if name not in ds.variables:
        return None
    return np.asarray(ds.variables[name][:], dtype=float)


def _desc_ballooning(wout, rho, alpha, nturns, nzetaperturn, nzeta0, L, M, N):
    from desc.vmec import VMECIO
    from desc.objectives import BallooningStability

    eq = VMECIO.load(wout, L=L, M=M, N=N)
    zeta0 = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nzeta0)
    lam = np.zeros((len(rho), len(alpha)))
    lam_shift = 1.0
    for j, a in enumerate(alpha):
        obj = BallooningStability(
            eq,
            rho=rho,
            alpha=np.array([a]),
            nturns=nturns,
            nzetaperturn=nzetaperturn,
            zeta0=zeta0,
            Neigvals=1,
            lambda0=-lam_shift,
            w0=0.0,
            w1=1.0,
        )
        obj.build(use_jit=False, verbose=0)
        values = np.asarray(obj.compute(eq.params_dict), dtype=float).ravel()
        if values.size != len(rho):
            raise ValueError(
                f"unexpected ballooning result size {values.size}; expected {len(rho)}"
            )
        lam[:, j] = values - lam_shift
    return lam


def _desc_newcomb_metric(wout, rho, alpha, L, M, N, nzeta=81):
    """DESC Newcomb ballooning metric on a radial grid.

    Positive values indicate stability; negative values indicate instability.
    """
    from desc.vmec import VMECIO
    from desc.grid import Grid, LinearGrid

    eq = VMECIO.load(wout, L=L, M=M, N=N)
    rho = np.asarray(rho, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    nfp = int(eq.NFP)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, int(nzeta), endpoint=False)
    source = LinearGrid(rho=rho, theta=alpha, zeta=zeta,
                        NFP=nfp, sym=False, endpoint=False)
    rr, aa, zz = np.meshgrid(rho, alpha, zeta, indexing="ij")
    nodes = np.column_stack([rr.ravel(), aa.ravel(), zz.ravel()])
    grid = Grid(nodes, coordinates="raz", NFP=nfp, source_grid=source,
                sort=False, is_meshgrid=True)
    data = eq.compute(["Newcomb ballooning metric"], grid=grid)
    return np.asarray(data["Newcomb ballooning metric"], dtype=float).ravel()[:len(rho)]


def _desc_effective_ripple(wout, rho):
    from desc.vmec import VMECIO
    from desc.grid import LinearGrid
    from desc.objectives import EffectiveRipple

    eq = VMECIO.load(wout)
    values = []
    for r in rho:
        grid = LinearGrid(rho=np.array([r]), M=eq.M_grid, N=eq.N_grid,
                          NFP=eq.NFP, sym=False)
        obj = EffectiveRipple(eq, grid=grid, num_transit=10, num_pitch=31)
        obj.build(verbose=0)
        values.append(float(np.asarray(obj.compute(eq.params_dict)).reshape(-1)[0]))
    return np.asarray(values, dtype=float)


def _dangerous_rationals(nfp=4, m_max=16):
    """Signed low-order rationals, including non-NFP assembly-error modes."""
    del nfp
    from fractions import Fraction

    vals = []
    for m in range(1, m_max + 1):
        for numerator in range(1, int(np.ceil(1.4 * m)) + 1):
            rational = Fraction(numerator, m)
            value = float(rational)
            if value >= 1.4:
                continue
            for sign in (-1, 1):
                signed_num = sign * rational.numerator
                vals.append(
                    (sign * value, f"{signed_num}/{rational.denominator}",
                     rational.denominator)
                )
    unique = {}
    for val, label, m in vals:
        key = round(val, 12)
        if key not in unique or m < unique[key][2]:
            unique[key] = (val, label, m)
    return sorted(unique.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wout", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--diagnostics_json")
    ap.add_argument("--desc_force_balance_json")
    ap.add_argument("--rho", nargs="+", type=float,
                    default=[0.35, 0.5, 0.65, 0.8, 0.9])
    ap.add_argument("--nalpha", type=int, default=8)
    ap.add_argument("--nturns", type=int, default=2)
    ap.add_argument("--nzetaperturn", type=int, default=80)
    ap.add_argument("--nzeta0", type=int, default=5)
    ap.add_argument("--desc_L", type=int, default=6)
    ap.add_argument("--desc_M", type=int, default=6)
    ap.add_argument("--desc_N", type=int, default=6)
    args = ap.parse_args(argv)
    initialize_requested_jax_backend()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wout = Path(args.wout)

    ds = netCDF4.Dataset(wout)
    ns = int(_scalar(ds, "ns", 0))
    s = np.linspace(0.0, 1.0, ns)
    rho_grid = np.sqrt(s)
    mercier = mercier_profiles(ds)
    s_half, dmerc_raw = vmec_half_grid_profile(
        mercier["profiles"]["DMerc"]["vmec_raw"], ns
    )
    _, dmerc = vmec_half_grid_profile(
        mercier["profiles"]["DMerc"]["flux_normalized"], ns
    )
    dwell_raw = mercier["profiles"].get("DWell", {}).get("flux_normalized")
    dwell = (
        vmec_half_grid_profile(dwell_raw, ns)[1]
        if dwell_raw is not None else None
    )
    dgeod_raw = mercier["profiles"].get("DGeod", {}).get("flux_normalized")
    dgeod = (
        vmec_half_grid_profile(dgeod_raw, ns)[1]
        if dgeod_raw is not None else None
    )
    iotaf = _array(ds, "iotaf")
    presf = _array(ds, "presf")
    vp = _array(ds, "vp")
    jcuru = _array(ds, "jcuru")
    jcurv = _array(ds, "jcurv")
    beta = _scalar(ds, "betatotal")
    aspect = _scalar(ds, "aspect")
    ctor = _scalar(ds, "ctor", 0.0)
    nfp = int(_scalar(ds, "nfp", 1))
    ds.close()

    if vp is not None and len(vp) > 1 and abs(vp[1]) > 1e-30:
        well_depth = (float(vp[1]) - vp) / float(vp[1])
    else:
        well_depth = np.full_like(s, np.nan)

    rho = np.asarray(args.rho, dtype=float)
    alpha = np.linspace(0, np.pi, args.nalpha, endpoint=False)
    ballooning = _desc_ballooning(
        str(wout), rho, alpha, args.nturns, args.nzetaperturn, args.nzeta0,
        args.desc_L, args.desc_M, args.desc_N,
    )
    try:
        newcomb = _desc_newcomb_metric(
            str(wout), rho, alpha, args.desc_L, args.desc_M, args.desc_N)
        newcomb_error = None
    except Exception as exc:
        newcomb = np.full(len(rho), np.nan)
        newcomb_error = str(exc)
    eps_eff = _desc_effective_ripple(str(wout), rho)

    try:
        from simsopt.geo import SurfaceRZFourier
        from ...diagnostics.boundary_geometry import boundary_geometry_metrics
        from ...objectives.pdrot_residual import (
            pdrot_area_weighted_stats,
            principal_direction_rotation_rate,
        )
        boundary_geometry = boundary_geometry_metrics(
            str(wout), ntheta=128, nphi=128, torus_range="full torus"
        )
        pdrot_surface = SurfaceRZFourier.from_wout(
            str(wout), range="full torus", ntheta=128, nphi=128
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

    desc_fb = None
    if args.desc_force_balance_json:
        p = Path(args.desc_force_balance_json)
        if p.exists():
            desc_fb = json.loads(p.read_text())

    # MHD/iota plot.
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    mhd_plot_mask = (s_half >= 0.1) & (s_half <= 0.97)
    full_plot_mask = (s >= 0.1) & (s <= 0.97)
    ax[0, 0].plot(
        s_half[mhd_plot_mask], dmerc[mhd_plot_mask], marker="o",
        label=r"$\Phi_{edge}^2 D_{Merc}$",
    )
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_xlim(0.1, 0.97)
    ax[0, 0].set_xlabel("s = normalized toroidal flux")
    ax[0, 0].set_ylabel(r"$\Phi_{edge}^2 D_{Merc}$")
    ax[0, 0].set_title("Mercier stability profile")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(
        s[full_plot_mask], 100 * well_depth[full_plot_mask], marker="o",
        label="well depth",
    )
    if dwell is not None:
        ax2 = ax[0, 1].twinx()
        ax2.plot(s_half[mhd_plot_mask], dwell[mhd_plot_mask],
                 color="tab:orange", marker="s", alpha=0.75,
                 label=r"$\Phi_{edge}^2 D_{Well}$")
        ax2.set_ylabel(r"$\Phi_{edge}^2 D_{Well}$")
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_xlabel("s = normalized toroidal flux")
    ax[0, 1].set_ylabel("Magnetic well depth [%]")
    ax[0, 1].set_title("Magnetic well / interchange proxy")
    ax[0, 1].set_xlim(0.1, 0.97)

    ax[1, 0].plot(s, iotaf, marker="o", label="iota")
    for val, label, m in _dangerous_rationals(nfp, 16):
        if np.nanmin(iotaf) - 0.03 <= val <= np.nanmax(iotaf) + 0.03:
            style = "-" if m <= 8 else "--"
            color = "tab:red" if m <= 8 else "tab:orange"
            ax[1, 0].axhline(val, color=color, ls=style, lw=0.8, alpha=0.7)
            ax[1, 0].text(1.005, val, f"{label} m={m}", va="center",
                          fontsize=7, color=color)
    ax[1, 0].set_xlabel("s = normalized toroidal flux")
    ax[1, 0].set_ylabel("rotational transform iota [dimensionless]")
    ax[1, 0].set_title(f"Low-order rational placement (q<=16, NFP={nfp})")

    ax[1, 1].plot(s, presf, marker="o")
    ax[1, 1].axhline(0, color="k", lw=0.8)
    ax[1, 1].set_xlabel("s = normalized toroidal flux")
    ax[1, 1].set_ylabel("pressure p [Pa]")
    ax[1, 1].set_title("Pressure profile")
    for axi in ax.ravel():
        axi.grid(alpha=0.25)
    fig.savefig(out / "mhd_iota_pressure_profiles.png", dpi=180)

    # Ballooning and ripple plot.
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    im = ax[0, 0].imshow(
        ballooning,
        origin="lower",
        aspect="auto",
        extent=[0.0, np.pi, rho[0], rho[-1]],
        interpolation="nearest",
    )
    ax[0, 0].set_xlabel("field-line label alpha [rad]")
    ax[0, 0].set_ylabel("rho = sqrt(s)")
    ax[0, 0].set_title("DESC ideal ballooning screen (VMECIO fit)")
    fig.colorbar(im, ax=ax[0, 0], label="raw lambda [DESC normalized units]")

    ax[0, 1].plot(rho, np.nanmax(ballooning, axis=1), marker="o")
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_xlabel("rho = sqrt(s)")
    ax[0, 1].set_ylabel("max raw lambda [DESC units]")
    ax[0, 1].set_title("Worst ballooning by radius (VMECIO fit)")
    if desc_fb is not None and desc_fb.get("ballooning_lambda_max") is not None:
        ax[0, 1].text(
            0.02, 0.98,
            "DESC force-balance summary\\n"
            f"lambda_max={desc_fb['ballooning_lambda_max']:.3e}\\n"
            f"status={desc_fb.get('ballooning_status', 'n/a')}",
            transform=ax[0, 1].transAxes,
            va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
        )

    ax[1, 0].plot(rho, eps_eff, marker="o")
    ax[1, 0].set_xlabel("rho = sqrt(s)")
    ax[1, 0].set_ylabel("effective ripple epsilon_eff [dimensionless]")
    ax[1, 0].set_title("DESC EffectiveRipple")

    if np.any(np.isfinite(newcomb)):
        ax[1, 1].plot(rho, newcomb, marker="o", color="tab:blue",
                      label="Newcomb metric")
        ax[1, 1].fill_between(rho, 0, newcomb, where=newcomb >= 0,
                              color="tab:green", alpha=0.15)
        ax[1, 1].fill_between(rho, 0, newcomb, where=newcomb < 0,
                              color="tab:red", alpha=0.20)
        ax[1, 1].axhline(0, color="k", lw=0.8)
        ax[1, 1].set_xlabel("rho = sqrt(s)")
        ax[1, 1].set_ylabel("Newcomb ballooning metric")
        ax[1, 1].set_title("Newcomb ballooning proxy (>0 stable)")
    else:
        if jcuru is not None:
            ax[1, 1].plot(s, jcuru, marker="o", label="jcuru")
        if jcurv is not None:
            ax[1, 1].plot(s, jcurv, marker="s", label="jcurv")
        ax[1, 1].set_xlabel("s = normalized toroidal flux")
        ax[1, 1].set_ylabel("VMEC current-density output [VMEC units]")
        ax[1, 1].set_title("Current profile outputs")
        if newcomb_error:
            ax[1, 1].text(0.02, 0.05, f"Newcomb unavailable: {newcomb_error[:90]}",
                          transform=ax[1, 1].transAxes, fontsize=7,
                          color="0.4")
    ax[1, 1].legend(fontsize=8)
    for axi in ax.ravel():
        axi.grid(alpha=0.25)
    fig.savefig(out / "ballooning_ripple_current_profiles.png", dpi=180)

    ballooning_by_rho = [
        {"rho": float(r), "lambda_raw_max": float(v)}
        for r, v in zip(rho, np.nanmax(ballooning, axis=1))
    ]
    ripple_by_rho = [
        {"rho": float(r), "epsilon_eff": float(v)}
        for r, v in zip(rho, eps_eff)
    ]
    newcomb_by_rho = [
        {"rho": float(r), "newcomb_metric": float(v)}
        for r, v in zip(rho, newcomb)
    ]

    summary = {
        "wout": str(wout.resolve()),
        "global": {
            "aspect": float(aspect),
            "betatotal": float(beta),
            "ctor_A_VMEC": float(ctor),
        },
        "boundary_geometry_simsopt": boundary_geometry,
        "pdrot_residual_area_weighted": pdrot_residual_area_weighted,
        "mhd": {
            "dmerc_convention": mercier["convention"],
            "dmerc_formula": mercier["formula"],
            "dmerc_edge_toroidal_flux_wb": mercier["edge_toroidal_flux_wb"],
            "dmerc_vmec_raw_min_s_0p1_0p97": float(np.nanmin(dmerc_raw[(s_half >= 0.1) & (s_half <= 0.97)])),
            "dmerc_flux_normalized_min_s_0p1_0p97": float(np.nanmin(dmerc[(s_half >= 0.1) & (s_half <= 0.97)])),
            "mercier_negative_count_s_ge_0p1": int(np.sum(dmerc[(s_half >= 0.1) & (s_half <= 0.97)] < 0)),
            "well_depth_edge": float(well_depth[-1]),
            "pressure_edge_Pa": float(presf[-1]),
        },
        "iota": {
            "axis": float(iotaf[0]),
            "edge": float(iotaf[-1]),
            "min": float(np.nanmin(iotaf)),
            "max": float(np.nanmax(iotaf)),
        },
        "desc_ballooning": {
            "source": "DESC BallooningStability on VMECIO.load(wout) fit; not a re-solved DESC force-balance equilibrium unless an equilibrium file is supplied",
            "rho": rho.tolist(),
            "alpha": alpha.tolist(),
            "lambda_raw": ballooning.tolist(),
            "lambda_raw_max_by_rho": ballooning_by_rho,
            "lambda_raw_max": float(np.nanmax(ballooning)),
            "unstable_count": int(np.sum(ballooning > 0)),
        },
        "desc_newcomb": {
            "source": "DESC Newcomb ballooning metric on VMECIO.load(wout) fit; positive is stable, negative is unstable",
            "rho": rho.tolist(),
            "newcomb_metric": newcomb.tolist(),
            "newcomb_metric_by_rho": newcomb_by_rho,
            "newcomb_metric_min": (
                float(np.nanmin(newcomb)) if np.any(np.isfinite(newcomb))
                else None
            ),
            "negative_count": (
                int(np.sum(newcomb < 0)) if np.any(np.isfinite(newcomb))
                else None
            ),
            "error": newcomb_error,
        },
        "desc_effective_ripple": {
            "rho": rho.tolist(),
            "epsilon_eff": eps_eff.tolist(),
            "epsilon_eff_by_rho": ripple_by_rho,
            "epsilon_eff_max": float(np.nanmax(eps_eff)),
        },
        "desc_force_balance": desc_fb,
        "plots": {
            "mhd_iota_pressure": str((out / "mhd_iota_pressure_profiles.png").resolve()),
            "ballooning_ripple_current": str((out / "ballooning_ripple_current_profiles.png").resolve()),
        },
    }
    (out / "physical_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    if args.diagnostics_json:
        p = Path(args.diagnostics_json)
        if p.exists():
            report = json.loads(p.read_text())
            report["physical_validation"] = {
                "global": summary["global"],
                "mhd": summary["mhd"],
                "boundary_geometry_simsopt": summary["boundary_geometry_simsopt"],
                "iota": summary["iota"],
                "desc_ballooning": {
                    "source": summary["desc_ballooning"]["source"],
                    "lambda_raw_max_by_rho": ballooning_by_rho,
                    "lambda_raw_max": summary["desc_ballooning"]["lambda_raw_max"],
                    "unstable_count": summary["desc_ballooning"]["unstable_count"],
                },
                "desc_effective_ripple": {
                    "epsilon_eff_by_rho": ripple_by_rho,
                    "epsilon_eff_max": summary["desc_effective_ripple"]["epsilon_eff_max"],
                },
                "desc_newcomb": summary["desc_newcomb"],
                "desc_force_balance": desc_fb,
                "desc_force_balance_note": (
                    "Only desc_force_balance.json was found; no force-balanced DESC equilibrium file was saved, "
                    "so radial ballooning plots use the VMECIO fit screen."
                    if desc_fb is not None else None
                ),
                "plots": summary["plots"],
            }
            p.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0
