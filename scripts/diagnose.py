#!/usr/bin/env python3
"""
SQuID Diagnostic Tool — evaluate an equilibrium without optimisation.

Computes all SQuID target function components and generates
diagnostic plots, including stability and transport metrics.

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
    plot_J_contours,
)
from squid.evaluation.axis_geometry import axis_geometry_from_vmec, plot_axis_geometry
from squid.evaluation.available_energy import ae_diagnostics
from squid.core.boozer_utils import run_boozer


def _evaluate_effective_ripple(vmec, s_val, nc_file_path):
    """
    Evaluate effective ripple eps_eff at normalised flux *s_val*.

    Graceful degradation:
      1. NEO-RT — run Boozer transform (→ boozmn_*.nc), convert to
         NEO-RT in_file via nc_to_neort, run neo_rt.x, parse D11
         transport coefficient from output.
      2. Boozer proxy fallback — (B_max - B_min) / (2*B00), a ripple
         amplitude proxy (not the neoclassical eps_eff; good QI has
         true eps_eff < 1%).

    Returns
    -------
    (value, source_str) : (float | None, str)
    """
    import glob

    # ── Priority 1: NEO-RT ──────────────────────────────────────────
    try:
        from simsopt.mhd.boozer import Boozer
        import nc_to_neort
        import subprocess

        # Register enough surfaces for stable cubic spline interpolation.
        # Particle drift orbits can wander radially; a narrow [s±0.1]
        # range causes spline extrapolation → NaN.
        raw = [s_val - 0.2, s_val - 0.1, s_val, s_val + 0.1, s_val + 0.2]
        s_surfaces = sorted(set(max(0.01, min(0.99, s)) for s in raw))
        print(f"      [NEO-RT] Running Boozer transform at s={s_surfaces} ...")
        boozer = Boozer(vmec, mpol=16, ntor=16)
        for s in s_surfaces:
            boozer.register(s)
        boozer.run()

        latest_boozmn = "boozmn_squid_diag.nc"
        try:
            boozer.write_boozmn(latest_boozmn)
        except AttributeError:
            boozer.bx.write_boozmn(latest_boozmn)
        if not os.path.isfile(latest_boozmn):
            raise FileNotFoundError(
                f"write_boozmn() did not produce {latest_boozmn}")
        print(f"      [NEO-RT] Using Boozer file: {latest_boozmn}")

        # Convert boozmn netCDF → NEO-RT ASCII "in_file"
        # phi_b and aspect_b are "not implemented" in this booz_xform,
        # so we supply the real values from VMEC.
        phi_edge = float(vmec.wout.phi[-1])         # total toroidal flux [Tm²]
        a_minor  = float(vmec.wout.Aminor_p)        # minor radius [m]
        result = nc_to_neort.convert_boozmn_to_neort(
            latest_boozmn, output_path="in_file", s_values=s_surfaces,
            flux_override=phi_edge, a_override=a_minor,
        )
        if result is None:
            raise RuntimeError("boozmn → in_file conversion failed")
        in_file_path, epsmn, pert_m0, pert_mph = result

        # Locate neo_rt.x
        neort_exe = os.environ.get("NEORT_EXECUTABLE", "")
        if not neort_exe and os.environ.get("NEO_RT_ROOT"):
            neort_exe = os.path.join(
                os.environ["NEO_RT_ROOT"], "build", "neo_rt.x")
        if not neort_exe or not os.path.isfile(neort_exe):
            raise FileNotFoundError(
                f"neo_rt.x not found (NEORT_EXECUTABLE={neort_exe!r}). "
                "Set NEORT_EXECUTABLE or NEO_RT_ROOT.")

        # Write a minimal NEO-RT namelist for transport evaluation.
        # NOTE: run_driftorbit.run_single_flux_surface() is NOT used here
        # because it prepends './' to the executable path, which breaks
        # absolute paths.  We call neo_rt.x via subprocess directly.
        runname = "squid_diag"
        with open(f"{runname}.in", "w") as fh:
            fh.write(
                "&params\n"
                f"    s = {s_val}\n"
                "    m_t = 1.0d-2\n"
                "    qs = 1.0\n"
                "    ms = 2.014\n"
                "    vth = 1.0d8\n"
                f"    epsmn = {epsmn}\n"
                f"    m0 = {pert_m0}\n"
                f"    mph = {max(pert_mph, 1)}\n"
                "    magdrift = .true.\n"
                "    nopassing = .false.\n"
                "    noshear = .false.\n"
                "    pertfile = .false.\n"
                "    nonlin = .false.\n"
                "    bfac = 1.0\n"
                "    efac = 1.0\n"
                "    inp_swi = 9\n"
                "    vsteps = 512\n"
                "    log_level = -1\n"
                "/\n"
            )

        print(f"      [NEO-RT] Running {os.path.basename(neort_exe)} {runname} ...")
        result = subprocess.run(
            [neort_exe, runname],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"neo_rt.x exited with code {result.returncode}: "
                f"{result.stderr[:300]}")

        # Parse D11 from {runname}.out
        # Header: "# M_t D11co D11ctr D11t D11 D12co D12ctr D12t D12"
        out_file = f"{runname}.out"
        if not os.path.isfile(out_file):
            raise FileNotFoundError(f"{out_file} not written by neo_rt.x")
        with open(out_file) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                cols = stripped.split()
                if len(cols) >= 5:
                    D11 = float(cols[4])
                    if not np.isfinite(D11):
                        raise ValueError(
                            "NEO-RT returned NaN/Inf — likely spline "
                            "extrapolation or extreme ripple "
                            f"(eps≈{epsmn:.2f})")
                    return (D11,
                            "NEO-RT D11 (neoclassical transport coeff)")
        raise RuntimeError(f"Could not parse D11 from {out_file}")

    except ImportError as e:
        print(f"      [Fallback] NEO-RT 依赖不可用: {e}")
    except FileNotFoundError as e:
        print(f"      [Fallback] 文件未找到: {e}")
    except Exception as e:
        print(f"      [Fallback] NEO-RT 评估失败: {e}")

    # ── Priority 2: Boozer proxy (fallback) ─────────────────────────
    try:
        _, surface_data = run_boozer(vmec, [s_val], mpol=16, ntor=16)
        data = surface_data[0]
        B_min, B_max = data["B_min"], data["B_max"]
        m_arr = np.array(data["m"])
        n_arr = np.array(data["n"])
        bmnc = np.array(data["bmnc"])
        idx_00 = np.where((m_arr == 0) & (n_arr == 0))[0]
        B00 = float(np.abs(bmnc[idx_00[0]])) if len(idx_00) > 0 else (B_max + B_min) / 2.0
        if B00 > 1e-30:
            eps_proxy = (B_max - B_min) / (2.0 * B00)
            return (float(eps_proxy), "Boozer proxy (B_max-B_min)/(2*B00)")
    except Exception as e:
        return (None, f"Boozer proxy failed: {e}")

    return (None, "No method available")


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
    parser.add_argument("--eps_eff_surface", type=float, default=0.5,
                        help="Normalised flux surface for effective ripple evaluation (default 0.5)")
    parser.add_argument("--ae", action="store_true",
                        help="Compute Available Energy (TEM turbulence metric)")
    parser.add_argument("--ae_omn", type=float, default=1.0,
                        help="AE: -d ln n / d s")
    parser.add_argument("--ae_omt", type=float, default=3.0,
                        help="AE: -d ln T / d s")
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

    # ---------------------------------------------------------
    # NEW: Stability & Transport Diagnostics
    # ---------------------------------------------------------
    print(f"\n--- Stability & Transport Diagnostics ---")
    
    # 1. Mercier Stability Criterion
    try:
        dmerc = vmec.wout.Dmerc[1:]
        min_dmerc = np.min(dmerc)
        if min_dmerc > 0:
            print(f"  Mercier Stability: STABLE (Min D_Merc = {min_dmerc:.4e} > 0)")
        else:
            print(f"  Mercier Stability: UNSTABLE (Min D_Merc = {min_dmerc:.4e} < 0)")
    except Exception:
        print("  Mercier Stability: [Not available in this nc file]")
        
    

    # 2. Magnetic Well Depth
    try:
        vp = vmec.wout.vp[1:] # Specific volume on half mesh
        well_depth = (vp[0] - vp[-1]) / vp[0]
        print(f"  Magnetic Well Depth: {well_depth * 100:.2f}% (>0 means Well, <0 means Hill)")
    except Exception:
        print("  Magnetic Well Depth: [Not available in this nc file]")


    # 3. Effective Ripple (eps_eff)
    print("  Effective Ripple (eps_eff / 有效波纹度):")
    eps_eff_surface = getattr(args, "eps_eff_surface", 0.5)
    eps_eff_value, eps_eff_source = _evaluate_effective_ripple(vmec, eps_eff_surface, args.nc_file)
    if eps_eff_value is not None:
        print(f"      eps_eff at s={eps_eff_surface}: {eps_eff_value:.4e}  ({eps_eff_source})")
        if "Boozer proxy" in eps_eff_source:
            print("      [注] 上述为波纹幅度代理 (B_max-B_min)/(2*B00)，非新经典 ε_eff。"
                  "真实 ε_eff 需 NEO-RT：在同目录放 in_file 与 diagnose_eps.in 后重跑；良好 QI 位型 ε_eff 通常 < 1%。")
    else:
        print(f"      [Could not evaluate. {eps_eff_source}]")

    # 4. Axis Geometry
    print(f"\n--- Axis Geometry ---")
    try:
        ax_info = axis_geometry_from_vmec(vmec)
        kappa = ax_info["curvature"]
        tau = ax_info["torsion"]
        print(f"  Axis length:    {ax_info['axis_length']:.4f} m")
        print(f"  Curvature κ:    min={np.min(kappa):.4f}, "
              f"max={np.max(kappa):.4f}, mean={np.mean(kappa):.4f}")
        print(f"  Torsion   τ:    min={np.min(tau):.4f}, "
              f"max={np.max(tau):.4f}, mean={np.mean(tau):.4f}")
    except Exception as e:
        print(f"  [Failed] {e}")

    # 5. Available Energy (TEM turbulence)
    if args.ae:
        print(f"\n--- Available Energy (TEM) ---")
        try:
            ae_info = ae_diagnostics(
                vmec, s_vals=s_vals,
                omn=args.ae_omn, omt=args.ae_omt,
                n_alpha=min(args.num_alpha, 4),
                n_turns=3, lam_res=200, gridpoints=512,
            )
        except Exception as e:
            print(f"  [Failed] {e}")

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

        print("  Computing J contour polar plot (Fig. 9) ...")
        fig4 = plot_J_contours(vmec, lambda_N=0.3)
        fig4.savefig("j_contours_polar.png", dpi=150, bbox_inches="tight")
        print("  Saved: j_contours_polar.png")

        try:
            fig5 = plot_axis_geometry(vmec)
            fig5.savefig("axis_geometry.png", dpi=150, bbox_inches="tight")
            print("  Saved: axis_geometry.png")
        except Exception as e:
            print(f"  [axis_geometry plot failed: {e}]")

        plt.close("all")

    print(f"\n{'=' * 60}")
    print("Diagnostic evaluation complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
