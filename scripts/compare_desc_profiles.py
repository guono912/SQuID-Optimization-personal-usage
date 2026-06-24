#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two DESC/VMEC equilibria using profile diagnostics.

This follows the user's profile-plot logic:
  - pressure p(rho)
  - rotational transform iota(rho)
  - flux-surface averaged beta(rho)
  - toroidal current profile current(rho) [A]

Inputs may be DESC .h5 files or VMEC wout .nc files. VMEC files are loaded
through DESC VMECIO so that beta/current are computed with DESC variable names.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=".*[Ll]eft handed coordinates detected.*",
    module="desc",
)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from desc.grid import LinearGrid
from desc.io import load
from desc.vmec import VMECIO


RHO = np.linspace(0.02, 1.0, 50)
RHO_BETA = np.linspace(0.0, 1.0, 11)
RHO_CURRENT = np.linspace(0.0, 1.0, 21)
MU0 = 4e-7 * np.pi


def _last_equilibrium(obj):
    eq = obj
    if isinstance(eq, (list, tuple)) and len(eq) > 0:
        eq = eq[-1]
    if hasattr(eq, "__len__") and not hasattr(eq, "compute") and len(eq) > 0:
        eq = eq[-1]
    return eq


def load_eq(path):
    path = Path(path)
    if path.suffix.lower() == ".nc":
        return VMECIO.load(str(path), L=6, M=6, N=6)
    return _last_equilibrium(load(str(path)))


def _surface_mean(values, n_rho, n_nodes):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == n_rho:
        return arr
    n_per = n_nodes // n_rho
    if n_per > 0 and n_per * n_rho == arr.size:
        return np.nanmean(arr.reshape((n_rho, n_per)), axis=1)
    return np.full(n_rho, np.nan)


def get_profiles(eq):
    nfp = int(eq.NFP)

    grid_rho = LinearGrid(rho=RHO, theta=0.0, zeta=0.0, NFP=nfp)
    p = np.asarray(eq.compute("p", grid=grid_rho)["p"], dtype=float).ravel()
    iota = np.asarray(eq.compute("iota", grid=grid_rho)["iota"], dtype=float).ravel()

    n_beta = len(RHO_BETA)
    grid_beta = LinearGrid(rho=RHO_BETA, M=10, N=10, NFP=nfp)
    n_nodes_beta = grid_beta.num_nodes
    beta_vals = None
    for key in ["<beta>", "beta", "<beta>_fs"]:
        try:
            data = eq.compute(key, grid=grid_beta)
            beta_vals = _surface_mean(data[key], n_beta, n_nodes_beta)
            if np.any(np.isfinite(beta_vals)):
                break
        except Exception:
            beta_vals = None
    if beta_vals is None:
        try:
            data = eq.compute(["p", "|B|"], grid=grid_beta)
            p_b = np.asarray(data["p"], dtype=float).ravel()
            B_b = np.asarray(data["|B|"], dtype=float).ravel()
            beta_local = 2 * MU0 * p_b / np.where(B_b > 1e-10, B_b**2, np.nan)
            beta_vals = _surface_mean(beta_local, n_beta, n_nodes_beta)
        except Exception:
            beta_vals = np.full(n_beta, np.nan)

    try:
        B = np.asarray(eq.compute("|B|", grid=grid_beta)["|B|"], dtype=float).ravel()
        B2_mean = _surface_mean(B**2, n_beta, n_nodes_beta)
    except Exception:
        B2_mean = np.full(n_beta, np.nan)

    n_cur = len(RHO_CURRENT)
    grid_current = LinearGrid(rho=RHO_CURRENT, theta=0.0, zeta=0.0, NFP=nfp)
    try:
        cur = np.asarray(eq.compute("current", grid=grid_current)["current"],
                         dtype=float).ravel()
        current = _surface_mean(cur, n_cur, grid_current.num_nodes)
    except Exception:
        current = np.full(n_cur, np.nan)

    return {
        "rho": RHO.copy(),
        "p": p,
        "iota": iota,
        "rho_beta": RHO_BETA.copy(),
        "beta": beta_vals,
        "rho_current": RHO_CURRENT.copy(),
        "current": current,
        "B2_mean": B2_mean,
    }


def get_total_current(eq):
    nfp = int(eq.NFP)
    grid = LinearGrid(rho=np.array([1.0]), M=10, N=10, NFP=nfp)
    c = np.asarray(eq.compute("current", grid=grid)["current"], dtype=float).ravel()
    return float(np.nanmean(c)) if c.size else np.nan


def get_volume_averaged_beta(eq):
    try:
        data = eq.compute("<beta>_vol")
        val = np.asarray(data["<beta>_vol"], dtype=float).ravel()
        if val.size and np.isfinite(val[0]):
            return float(val[0])
    except Exception:
        pass

    nfp = int(eq.NFP)
    grid = LinearGrid(
        rho=np.linspace(0.0, 1.0, 40),
        theta=np.linspace(0, 2 * np.pi, 32, endpoint=False),
        zeta=np.linspace(0, 2 * np.pi / nfp, 16, endpoint=False),
        NFP=nfp,
    )
    d = eq.compute(["p", "|B|", "sqrt(g)"], grid=grid)
    p = np.asarray(d["p"], dtype=float).ravel()
    B = np.asarray(d["|B|"], dtype=float).ravel()
    sg = np.asarray(d["sqrt(g)"], dtype=float).ravel()
    beta_local = 2 * MU0 * p / np.where(B > 1e-10, B**2, np.nan)
    volume = np.nansum(sg)
    return float(np.nansum(beta_local * sg) / volume) if volume > 0 else np.nan


def get_B_range(eq):
    nfp = int(eq.NFP)
    grid = LinearGrid(
        rho=np.linspace(0.0, 1.0, 40),
        theta=np.linspace(0, 2 * np.pi, 32, endpoint=False),
        zeta=np.linspace(0, 2 * np.pi / nfp, 16, endpoint=False),
        NFP=nfp,
    )
    B = np.asarray(eq.compute("|B|", grid=grid)["|B|"], dtype=float).ravel()
    return float(np.nanmax(B)), float(np.nanmin(B))


def _json_arr(arr):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(arr).ravel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs=2, action="append", metavar=("PATH", "LABEL"),
                    required=True,
                    help="Input equilibrium path and plot label. May repeat.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--title", default="Equilibrium Profile Comparison")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    summary = {}
    for path, label in args.case:
        print(f"加载: {Path(path).name} [{label}]")
        eq = load_eq(path)
        pro = get_profiles(eq)
        total_current = get_total_current(eq)
        beta_vol = get_volume_averaged_beta(eq)
        Bmax, Bmin = get_B_range(eq)
        cases.append((label, pro))
        summary[label] = {
            "path": str(path),
            "NFP": float(eq.NFP),
            "total_toroidal_current_A": total_current,
            "beta_vol": beta_vol,
            "iota_max": float(np.nanmax(pro["iota"])),
            "iota_min": float(np.nanmin(pro["iota"])),
            "B_max_T": Bmax,
            "B_min_T": Bmin,
            "rho_beta": _json_arr(pro["rho_beta"]),
            "B2_mean_T2": _json_arr(pro["B2_mean"]),
        }

    print("\n" + "=" * 60)
    print("总环向电流 (A)，由 DESC 内置 current(ρ=1) 读取")
    print("=" * 60)
    for label, data in summary.items():
        print(f"  {label}: {data['total_toroidal_current_A']:.6e} A")

    print("\n" + "=" * 60)
    print("体积平均比压 <beta>_vol")
    print("=" * 60)
    for label, data in summary.items():
        print(f"  {label}: {data['beta_vol']:.6e}")

    print("\n" + "=" * 60)
    print("旋转变换 ι 的最大值和最小值")
    print("=" * 60)
    for label, data in summary.items():
        print(f"  {label}: ι_max = {data['iota_max']:.6f}, "
              f"ι_min = {data['iota_min']:.6f}")

    print("\n" + "=" * 60)
    print("每个磁面上的平均 B² (T²)")
    print("=" * 60)
    for label, data in summary.items():
        print(f"  {label}:")
        for rho, b2 in zip(data["rho_beta"], data["B2_mean_T2"]):
            if b2 is not None:
                print(f"    ρ={rho:.2f}: <B²> = {b2:.6f} T²")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_p, ax_iota, ax_beta, ax_j = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.suptitle(args.title, fontsize=17)

    for idx, (label, pro) in enumerate(cases):
        c = colors[idx % len(colors)]
        ax_p.plot(pro["rho"], pro["p"], color=c, label=label)
        ax_iota.plot(pro["rho"], pro["iota"], color=c, label=label)
        ax_beta.plot(pro["rho_beta"], pro["beta"], color=c, label=label)
        ax_j.plot(pro["rho_current"], pro["current"], color=c, label=label)

    ax_p.set_xlabel(r"$\rho$")
    ax_p.set_ylabel(r"$p$ [Pa]")
    ax_p.set_title("Pressure")
    ax_p.legend()
    ax_p.grid(True, alpha=0.3)

    ax_iota.set_xlabel(r"$\rho$")
    ax_iota.set_ylabel(r"$\iota$")
    ax_iota.set_title("Rotational transform")
    ax_iota.legend()
    ax_iota.grid(True, alpha=0.3)

    ax_beta.set_xlabel(r"$\rho$")
    ax_beta.set_ylabel(r"$\langle\beta\rangle_{\mathrm{flux}}$")
    ax_beta.set_title("Beta (flux-surface averaged)")
    ax_beta.set_xlim(0, 1)
    ax_beta.legend()
    ax_beta.grid(True, alpha=0.3)

    ax_j.set_xlabel(r"$\rho$")
    ax_j.set_ylabel("Toroidal Current [A]")
    ax_j.set_title("Toroidal Current Profile")
    ax_j.legend()
    ax_j.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = output_dir / "equilibrium_profiles_compare.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_json = output_dir / "equilibrium_profiles_compare_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n已保存: {out_png}")
    print(f"已保存: {out_json}")
    print("完成.")


if __name__ == "__main__":
    main()
