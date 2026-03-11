#!/usr/bin/env python3
"""
SQuID Diagnostic Tool — evaluate an equilibrium without optimisation.

Computes all SQuID target function components and generates
diagnostic plots.

Usage:
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --plot
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simsopt.mhd import Vmec

from squid.evaluation.evaluate import (
    evaluate_squid,
    evaluate_itg,
    plot_boozer_surface,
    plot_squash_stretch,
    plot_gradient_diagnostics,
)


def main():
    parser = argparse.ArgumentParser(
        description="SQuID diagnostic evaluation (no optimisation)"
    )
    parser.add_argument("--nc_file", type=str, required=True,
                        help="VMEC wout .nc file to evaluate")
    parser.add_argument("--num_alpha", type=int, default=8)
    parser.add_argument("--num_pitch", type=int, default=50)
    parser.add_argument("--num_surfaces", type=int, default=5)
    parser.add_argument("--s_center", type=float, default=0.5)
    parser.add_argument("--ds", type=float, default=0.1)
    parser.add_argument("--itg_method", choices=["drift_curvature", "vacuum_dBds"],
                        default="vacuum_dBds",
                        help="Bad-curvature detection method for f_nabla_s")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("SQuID Diagnostic Evaluation")
    print(f"{'=' * 60}")
    print(f"  Input: {args.nc_file}")

    vmec = Vmec(args.nc_file)
    vmec.run()
    print(f"  nfp = {vmec.wout.nfp}, ns = {vmec.wout.ns}")

    half = (args.num_surfaces - 1) * args.ds / 2
    s_vals = np.linspace(
        max(0.05, args.s_center - half),
        min(0.95, args.s_center + half),
        args.num_surfaces,
    )
    print(f"  Surfaces: {s_vals}")

    # SQuID evaluation
    print(f"\n--- SQuID Targets ---")
    info = evaluate_squid(
        vmec, s_vals=s_vals,
        num_alpha=args.num_alpha,
        num_pitch=args.num_pitch,
    )

    # ITG evaluation
    print(f"\n--- ITG Target ---")
    itg_info = evaluate_itg(
        vmec,
        snorms=np.array([0.1, 0.3, 0.5]),
        method=args.itg_method,
    )

    # Plots
    if args.plot:
        import matplotlib.pyplot as plt
        print(f"\n--- Generating plots ---")

        fig1 = plot_boozer_surface(vmec, s_val=0.5)
        fig1.savefig("boozer_surface.png", dpi=150, bbox_inches="tight")
        print("  Saved: boozer_surface.png")

        fig2 = plot_squash_stretch(vmec, s_val=0.5, alpha=0.0)
        fig2.savefig("squash_stretch.png", dpi=150, bbox_inches="tight")
        print("  Saved: squash_stretch.png")

        fig3 = plot_gradient_diagnostics(vmec, s_vals=s_vals,
                                         num_alpha=args.num_alpha,
                                         num_pitch=args.num_pitch)
        if fig3 is not None:
            fig3.savefig("gradient_diagnostics.png", dpi=150, bbox_inches="tight")
            print("  Saved: gradient_diagnostics.png")

        plt.close("all")

    print(f"\n{'=' * 60}")
    print("Diagnostic evaluation complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
