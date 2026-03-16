#!/usr/bin/env python3
"""
SQuID Stellarator Optimiser — unified entry point.

Complete PRX Energy target function:
    f_SQuID = w_QI * f_QI
            + w_maxJ * f_maxJ
            + w_AR * f_A
            + w_mirror * f_delta
            + w_iota * f_iota
            + w_grad_s * f_nabla_s
            + w_reg * f_reg

Backends:
  - VMEC (via simsopt): fastest, requires VMEC2000 Fortran extension
  - DESC (fallback):     pure Python, automatically used when VMEC absent

Usage:
    python scripts/optimize.py \\
        --nc_file path/to/wout_xxx.nc \\
        --maxiter 10
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Backend detection
HAS_VMEC = False
HAS_DESC = False
try:
    import vmec as _vmec_mod  # noqa: F401
    HAS_VMEC = True
except ImportError:
    pass
try:
    import desc as _desc_mod  # noqa: F401
    HAS_DESC = True
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="SQuID stellarator optimiser (complete PRX Energy target)"
    )
    parser.add_argument("--nc_file", type=str, required=True,
                        help="Initial VMEC wout .nc file")
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--aspect_target", type=float, default=None)
    parser.add_argument("--max_dofs", type=int, default=15)
    parser.add_argument("--num_alpha", type=int, default=4)
    parser.add_argument("--num_pitch", type=int, default=20)
    parser.add_argument("--num_surfaces", type=int, default=3)
    parser.add_argument("--ns_vmec", type=int, default=31)

    parser.add_argument("--w_qi", type=float, default=1.0)
    parser.add_argument("--w_maxj", type=float, default=1.0)
    parser.add_argument("--w_ar", type=float, default=100.0)
    parser.add_argument("--w_mirror", type=float, default=500.0)
    parser.add_argument("--w_iota", type=float, default=200.0)
    parser.add_argument("--w_grad_s", type=float, default=1.0)
    parser.add_argument("--w_reg", type=float, default=10.0)

    parser.add_argument("--grad_s_smin", type=float, default=0.1)
    parser.add_argument("--grad_s_smax", type=float, default=0.5)
    parser.add_argument("--grad_s_ns", type=int, default=3)

    parser.add_argument("--mirror_target", type=float, default=0.20)
    parser.add_argument("--mirror_max", type=float, default=None,
                        help="Upper mirror bound; if unset, penalty is symmetric around target")
    parser.add_argument("--iota_ax", type=float, default=None)
    parser.add_argument("--iota_edge", type=float, default=None)

    parser.add_argument("--w_well", type=float, default=0.0,
                        help="Weight for magnetic well penalty (0 = disabled)")
    parser.add_argument("--target_well", type=float, default=0.01,
                        help="Target magnetic well depth (>0 means well); 0.01 = 1%%")

    parser.add_argument("--abs_step", type=float, default=1e-4,
                        help="Absolute FD step for Jacobian (simsopt default 1e-7)")
    parser.add_argument("--rel_step", type=float, default=0.0,
                        help="Relative FD step for Jacobian")
    parser.add_argument("--perturb", type=float, default=0.0,
                        help="Random perturbation amplitude (fraction of |x|) to escape local minima")

    parser.add_argument("--backend", choices=["auto", "vmec", "desc"],
                        default="auto")
    parser.add_argument("--desc_L", type=int, default=4)
    parser.add_argument("--desc_M", type=int, default=4)
    parser.add_argument("--desc_N", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("SQuID Optimisation  --  unified squid package")
    print(f"{'=' * 60}")
    print(f"  Input : {args.nc_file}")

    use_desc = False
    if args.backend == "vmec":
        if not HAS_VMEC:
            print("  ERROR: --backend=vmec but VMEC extension not installed.")
            sys.exit(1)
    elif args.backend == "desc":
        use_desc = True
    else:
        if HAS_VMEC:
            use_desc = False
        elif HAS_DESC:
            use_desc = True
        else:
            print("  ERROR: neither VMEC nor DESC is installed.")
            sys.exit(1)

    if use_desc:
        from squid.backends.desc_backend import run_desc
        run_desc(args)
    else:
        from squid.backends.vmec_backend import run_vmec
        run_vmec(args)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
