"""MHD gate CLI implementation.

Reusable entry point for the promotion gate; scripts/gate/mhd_gate.py and the
legacy scripts/mhd_gate.py both call into it. The gate decision logic lives
in squid/evaluation/gates.py.
"""

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from ...evaluation.gates import gate
from ...utils.jax_runtime import initialize_requested_jax_backend


def build_parser():
    ap = argparse.ArgumentParser(
        description="MHD Gate: DESC force-balance + ballooning + ripple")
    ap.add_argument("--wout", required=True, help="VMEC wout .nc file")
    ap.add_argument("--output_dir", default="runs/mhd_gate_latest")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--compare", type=str, default=None,
                    help="Previous gate JSON to compare against")
    ap.add_argument("--extended-edge", action="store_true",
                    help="Use R2.35 extended edge protocol: rho includes 0.975/0.99 and 12 alpha.")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    initialize_requested_jax_backend()

    rho_v = None
    alpha_v = None
    protocol = None
    if args.extended_edge:
        rho_v = np.array([0.1, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 0.975, 0.99])
        alpha_v = np.linspace(0, np.pi, 12, endpoint=False)
        protocol = "extended_edge_rho_0p975_0p99_alpha12"

    report = gate(
        args.wout, args.output_dir, L=args.L, M=args.M, N=args.N,
        rho_v=rho_v, alpha_v=alpha_v, protocol=protocol)

    if args.compare and os.path.exists(args.compare):
        with open(args.compare) as f:
            prev = json.load(f)
        print(f"\n--- Comparison with {args.compare} ---")
        for key in ["ballooning.lam_max", "ballooning.n_unstable", "ripple.peak"]:
            parts = key.split(".")
            curr_val = report[parts[0]][parts[1]]
            prev_val = prev[parts[0]][parts[1]]
            if curr_val is not None and prev_val is not None:
                delta = curr_val - prev_val
                arrow = "↓" if delta < 0 else "↑"
                print(f"  {key}: {prev_val:.2e} → {curr_val:.2e} ({arrow}{abs(delta):.1e})")
    return 0
