#!/usr/bin/env python3
"""
Generate an analytic QI-flavoured VMEC input file from physical parameters.

Based on Goodman et al., PRX Energy 3, 023010 (2024).

Usage examples:

  # nfp=2 QI seed (recommended starting point)
  python scripts/generate_seed.py --nfp 2 --R0 2.0 --aspect 8 \\
      --mirror 0.20 --elongation 2 --iota_ax 0.60 --iota_edge 0.50

  # nfp=4 QI seed
  python scripts/generate_seed.py --nfp 4 --R0 2.0 --aspect 10 \\
      --mirror 0.21 --elongation 3 --iota_ax 0.85 --iota_edge 0.72

  # Scale an existing wout to different R0 / nfp
  python scripts/generate_seed.py --from_wout wout_QI_nfp3.nc \\
      --nfp 4 --R0 2.0 --iota_ax 0.80 --iota_edge 0.72
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Generate VMEC input file for QI stellarator optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--from_wout", type=str, default=None,
                      help="Scale an existing wout .nc file instead of "
                           "generating from scratch")

    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--R0", type=float, default=1.0,
                        help="Major radius [m]")
    parser.add_argument("--aspect", type=float, default=8,
                        help="Target aspect ratio")
    parser.add_argument("--elongation", type=float, default=2,
                        help="Cross-section elongation")
    parser.add_argument("--mirror", type=float, default=0.20,
                        help="Target mirror ratio Δ")
    parser.add_argument("--Z2s", type=float, default=0.2,
                        help="Axis vertical excursion")
    parser.add_argument("--iota_ax", type=float, default=0.80,
                        help="Rotational transform at axis")
    parser.add_argument("--iota_edge", type=float, default=0.60,
                        help="Rotational transform at edge")
    parser.add_argument("--phiedge", type=float, default=None,
                        help="Boundary toroidal flux [T·m²]")
    parser.add_argument("--ns", type=int, default=51,
                        help="Number of VMEC radial surfaces")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: input.qi_seed_nfpN)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"input.qi_seed_nfp{args.nfp}"

    if args.from_wout:
        _from_wout(args)
    else:
        _from_analytic(args)


def _from_analytic(args):
    """Generate seed from analytic boundary parametrisation."""
    from squid.utils.generate_initial import write_vmec_input

    write_vmec_input(
        args.output,
        nfp=args.nfp,
        aspect=args.aspect,
        elongation=args.elongation,
        mirror=args.mirror,
        Z2s=args.Z2s,
        R0=args.R0,
        iota_axis=args.iota_ax,
        iota_edge=args.iota_edge,
        phiedge=args.phiedge,
        ns=args.ns,
    )
    print(f"  Generated: {args.output}")
    print(f"    nfp={args.nfp}, R0={args.R0}, A={args.aspect}, "
          f"E={args.elongation}, Δ={args.mirror}")
    print(f"    iota: {args.iota_ax:.3f} → {args.iota_edge:.3f}")
    print(f"\n  Next: run VMEC to get equilibrium, then diagnose andoptimize:")
    print(f"    xvmec {args.output}")


def _from_wout(args):
    """Scale an existing wout to new R0 / nfp / iota."""
    import numpy as np

    try:
        import netCDF4
    except ImportError:
        print("ERROR: netCDF4 required for --from_wout")
        sys.exit(1)

    ds = netCDF4.Dataset(args.from_wout, "r")
    nfp_old = int(ds.variables["nfp"][:])
    mpol = int(ds.variables["mpol"][:])
    ntor = int(ds.variables["ntor"][:])
    xm = np.array(ds.variables["xm"][:], dtype=int)
    xn_full = np.array(ds.variables["xn"][:], dtype=int)
    rmnc = np.array(ds.variables["rmnc"][:])
    zmns = np.array(ds.variables["zmns"][:])
    phi_arr = np.array(ds.variables["phi"][:])
    raxis_cc = np.array(ds.variables["raxis_cc"][:])
    zaxis_cs = np.array(ds.variables["zaxis_cs"][:])
    ds.close()

    xn_in = xn_full // nfp_old

    R0_old = float(rmnc[-1, np.where((xm == 0) & (xn_full == 0))[0][0]])
    scale = args.R0 / R0_old if R0_old > 1e-10 else 1.0
    phiedge_old = float(phi_arr[-1])
    phiedge_new = args.phiedge if args.phiedge else phiedge_old * scale**2

    rbc, zbs = {}, {}
    for i in range(len(xm)):
        m, n = int(xm[i]), int(xn_in[i])
        if m >= mpol or abs(n) > ntor:
            continue
        rc = float(rmnc[-1, i]) * scale
        zs = float(zmns[-1, i]) * scale
        if abs(rc) > 1e-16 or abs(zs) > 1e-16:
            rbc[(n, m)] = rc
            zbs[(n, m)] = zs

    nfp_new = args.nfp

    with open(args.output, "w") as f:
        f.write("&INDATA\n")
        f.write(f"! Scaled from {args.from_wout}: "
                f"nfp {nfp_old}→{nfp_new}, R {R0_old:.3f}→{args.R0:.3f}\n")
        f.write("  DELT = 0.9\n  TCON0 = 1.0\n  NSTEP = 200\n")
        f.write(f"  NFP = {nfp_new}\n  MPOL = {mpol}\n  NTOR = {ntor}\n")
        f.write(f"  NS_ARRAY = {args.ns}\n  NITER_ARRAY = 10000\n")
        f.write("  FTOL_ARRAY = 1.0E-12\n")
        f.write(f"  PHIEDGE = {phiedge_new:.15e}\n")
        f.write("  GAMMA = 0.0\n  LFREEB = F\n")
        f.write("  NCURR = 0\n")
        f.write("  PIOTA_TYPE = 'power_series'\n")
        f.write(f"  AI(0) = {args.iota_ax:.15e}\n")
        f.write(f"  AI(1) = {args.iota_edge - args.iota_ax:.15e}\n")
        f.write("  PMASS_TYPE = 'power_series'\n")
        f.write("  PRES_SCALE = 0.0\n  AM(0) = 0.0\n")

        rax_new = raxis_cc * scale
        zax_new = zaxis_cs * scale
        rax_str = "  ".join(f"{v:.15e}" for v in rax_new)
        zax_str = "  ".join(f"{v:.15e}" for v in zax_new)
        f.write(f"  RAXIS_CC = {rax_str}\n")
        f.write(f"  ZAXIS_CS = {zax_str}\n")

        for (n, m) in sorted(rbc.keys()):
            f.write(f"  RBC({n:d},{m:d}) = {rbc[(n, m)]:.15e}\n")
            zs = zbs.get((n, m), 0.0)
            if abs(zs) > 1e-16:
                f.write(f"  ZBS({n:d},{m:d}) = {zs:.15e}\n")
        f.write("/\n")

    print(f"  Scaled: {args.from_wout} → {args.output}")
    print(f"    nfp: {nfp_old} → {nfp_new},  R0: {R0_old:.3f} → {args.R0:.3f}")
    print(f"    iota: {args.iota_ax:.3f} → {args.iota_edge:.3f}")
    print(f"    phiedge: {phiedge_new:.6e}")


if __name__ == "__main__":
    main()
