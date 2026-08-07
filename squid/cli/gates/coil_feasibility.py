"""Coil feasibility gate CLI implementation.

REGCOIL-like triage tool for a fixed-boundary VMEC target: loads the wout with
DESC, builds approximate winding surfaces, and runs DESC's regularized
surface-current solve over offsets and regularization strengths.

The reusable physics helpers (winding-surface construction, current-spectrum
metrics) live in squid/evaluation/coil_metrics.py and
squid/objectives/coil_proxy.py; this module only orchestrates the scan,
writes the CSV/JSON/PNG report, and parses the CLI.
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desc.grid import LinearGrid
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.magnetic_fields import solve_regularized_surface_current
from desc.vmec import VMECIO

from ...evaluation.coil_metrics import make_winding_surface
from ...objectives.coil_proxy import _float, _list_item, current_spectrum_metrics


def run_gate(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eq = VMECIO.load(args.wout, L=args.desc_L, M=args.desc_M, N=args.desc_N)
    aminor = _float(eq.compute("a")["a"])

    lambdas = np.array(args.lambdas, dtype=float)
    offsets = np.array(args.offsets, dtype=float)
    records = []

    for offset in offsets:
        surface = make_winding_surface(
            eq,
            offset,
            method=args.winding_surface_method,
            M=args.winding_M,
            N=args.winding_N,
        )
        field = FourierCurrentPotentialField(
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface.R_basis.modes[:, 1:],
            modes_Z=surface.Z_basis.modes[:, 1:],
            NFP=surface.NFP,
            sym=surface.sym,
            M_Phi=args.M_Phi,
            N_Phi=args.N_Phi,
        )
        source_grid = LinearGrid(
            M=max(3 * args.M_Phi, args.source_M),
            N=max(3 * args.N_Phi, args.source_N),
            NFP=eq.NFP,
            sym=False,
        )
        eval_grid = LinearGrid(
            M=args.eval_M,
            N=args.eval_N,
            NFP=eq.NFP,
            sym=False,
            rho=np.array([1.0]),
        )
        bmag_eval = np.asarray(eq.compute("|B|", grid=eval_grid)["|B|"], dtype=float)
        print(f"offset={offset:.3f}  lambdas={lambdas}")
        fields, data = solve_regularized_surface_current(
            field,
            eq,
            lambda_regularization=lambdas,
            current_helicity=tuple(args.current_helicity),
            vacuum=args.vacuum,
            regularization_type=args.regularization_type,
            source_grid=source_grid,
            eval_grid=eval_grid,
            verbose=args.verbose,
            chunk_size=args.chunk_size,
            B_plasma_chunk_size=args.chunk_size,
        )

        for i, lam in enumerate(lambdas):
            k_key = "||K||" if "||K||" in data else "|K|"
            k_norm = np.asarray(_list_item(data, k_key, i), dtype=float)
            bn_total = np.asarray(_list_item(data, "Bn_total", i), dtype=float)
            phi = np.asarray(_list_item(data, "Phi_mn", i), dtype=float)
            spec = current_spectrum_metrics(phi, fields[i].Phi_basis.modes)
            chi_b = float(_list_item(data, "chi^2_B", i))
            chi_k = float(_list_item(data, "chi^2_K", i))
            bn_unitless = bn_total / np.maximum(np.abs(bmag_eval), 1e-300)
            rec = {
                "wout": str(Path(args.wout).resolve()),
                "offset_fraction": float(offset),
                "offset_m_est": float(offset * aminor),
                "lambda_regularization": float(lam),
                "regularization_type": args.regularization_type,
                "current_helicity_M": int(args.current_helicity[0]),
                "current_helicity_N": int(args.current_helicity[1]),
                "vacuum": bool(args.vacuum),
                "chi2_B": chi_b,
                "chi2_K": chi_k,
                "Bn_rms_T": float(np.sqrt(np.mean(bn_total**2))),
                "Bn_max_abs_T": float(np.max(np.abs(bn_total))),
                "Bn_avg_abs_T": float(np.mean(np.abs(bn_total))),
                "Bn_rms_unitless": float(np.sqrt(np.mean(bn_unitless**2))),
                "Bn_max_abs_unitless": float(np.max(np.abs(bn_unitless))),
                "Bn_avg_abs_unitless": float(np.mean(np.abs(bn_unitless))),
                "K_mean": float(np.mean(k_norm)),
                "K_rms": float(np.sqrt(np.mean(k_norm ** 2))),
                "K_max": float(np.max(k_norm)),
                "K_p95": float(np.percentile(k_norm, 95)),
            }
            rec.update(spec)
            records.append(rec)

    csv_path = out_dir / "coil_feasibility_scan.csv"
    json_path = out_dir / "coil_feasibility_scan.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    make_plots(records, out_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


def make_plots(records, out_dir):
    offsets = sorted({r["offset_fraction"] for r in records})
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for offset in offsets:
        rows = [r for r in records if r["offset_fraction"] == offset]
        rows.sort(key=lambda r: r["lambda_regularization"])
        x = np.array([r["lambda_regularization"] for r in rows])
        label = f"offset {offset:.2f}a"
        axs[0, 0].loglog(x, [r["chi2_B"] for r in rows], marker="o", label=label)
        axs[0, 1].loglog(x, [r["K_rms"] for r in rows], marker="o", label=label)
        axs[1, 0].loglog(x, [r["K_max"] for r in rows], marker="o", label=label)
        axs[1, 1].semilogx(
            x, [r["phi_high_mode_fraction"] for r in rows], marker="o", label=label
        )
    axs[0, 0].set_ylabel("chi2_B")
    axs[0, 1].set_ylabel("K_rms")
    axs[1, 0].set_ylabel("K_max")
    axs[1, 1].set_ylabel("Phi high-mode fraction")
    for ax in axs.ravel():
        ax.set_xlabel("lambda_regularization")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_dir / "coil_feasibility_scan.png", dpi=180)


def build_parser():
    p = argparse.ArgumentParser(
        description="Coil feasibility gate: regularized surface-current scan")
    p.add_argument("--wout", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--offsets", type=float, nargs="+", default=[0.25, 0.35, 0.50])
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8],
    )
    p.add_argument("--M_Phi", type=int, default=6)
    p.add_argument("--N_Phi", type=int, default=6)
    p.add_argument("--desc_L", type=int, default=6)
    p.add_argument("--desc_M", type=int, default=6)
    p.add_argument("--desc_N", type=int, default=6)
    p.add_argument("--source_M", type=int, default=24)
    p.add_argument("--source_N", type=int, default=24)
    p.add_argument("--eval_M", type=int, default=24)
    p.add_argument("--eval_N", type=int, default=24)
    p.add_argument("--current_helicity", type=int, nargs=2, default=[1, 0])
    p.add_argument("--regularization_type", choices=["regcoil", "simple"], default="regcoil")
    p.add_argument("--winding_surface_method", choices=["normal", "scaled"], default="normal")
    p.add_argument("--winding_M", type=int, default=None)
    p.add_argument("--winding_N", type=int, default=None)
    p.add_argument("--vacuum", action="store_true")
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--verbose", type=int, default=1)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_gate(args)
    return 0
