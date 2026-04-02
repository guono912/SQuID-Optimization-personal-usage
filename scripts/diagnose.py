#!/usr/bin/env python3
"""
SQuID Diagnostic Tool — evaluate an equilibrium without optimisation.

Computes all SQuID target function components and generates
diagnostic plots, including stability and transport metrics.

Usage:
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --plot
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --extended
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --extended \
        --extended_surfaces 0.1 0.3 0.5
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --extended \
        --ae --ae_surfaces 0.2 0.5 0.8 --plot

Tiers:
    default      -> core SQuID + equilibrium sanity + geometry/stability preview
    --extended   -> add third-tier transport diagnostics (ITG proxy)
    --ae         -> add Available Energy on top of --extended
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simsopt.mhd import Vmec

from squid.evaluation.evaluate import (
    evaluate_squid_detailed,
    evaluate_itg,
    plot_J_contours,
    plot_squid_core_diagnostics,
    plot_transport_diagnostics,
)
from squid.evaluation.axis_geometry import axis_geometry_from_vmec
from squid.evaluation.available_energy import ae_diagnostics
from squid.core.boozer_utils import run_boozer


def _evaluate_effective_ripple(vmec, s_val, nc_file_path):
    """
    Evaluate effective ripple eps_eff at normalised flux *s_val*.

    Graceful degradation:
      1. DESC (Nemov formula via bounce integrals) — most reliable.
      2. NEO-RT — Boozer transform + Fortran solver.
      3. Boozer proxy fallback — (B_max - B_min) / (2*B00).

    Returns
    -------
    (value, source_str) : (float | None, str)
    """
    import glob

    # ── Priority 1: DESC effective ripple (Nemov formula) ────────────
    try:
        import numpy as _np
        from desc.vmec import VMECIO
        from desc.grid import LinearGrid
        from desc.objectives import EffectiveRipple

        print(f"      [DESC] Computing effective ripple at s={s_val} ...")
        eq = VMECIO.load(nc_file_path)
        rho = _np.array([_np.sqrt(s_val)])
        grid = LinearGrid(
            rho=rho, M=eq.M_grid, N=eq.N_grid,
            NFP=eq.NFP, sym=False,
        )
        obj = EffectiveRipple(eq, grid=grid, num_transit=10, num_pitch=31)
        obj.build(verbose=0)
        f = obj.compute(eq.params_dict)
        eps_eff = float(f[0])
        if not _np.isfinite(eps_eff):
            raise ValueError("DESC returned NaN/Inf")
        return (eps_eff, "DESC (Nemov bounce-integral)")

    except ImportError:
        print("      [DESC] Not available, trying NEO-RT ...")
    except Exception as e:
        print(f"      [DESC] Failed: {e}, trying NEO-RT ...")

    # ── Priority 2: NEO-RT ──────────────────────────────────────────
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
        print(f"      [Fallback] NEO-RT dependencies not available: {e}")
    except FileNotFoundError as e:
        print(f"      [Fallback] File not found: {e}")
    except Exception as e:
        print(f"      [Fallback] NEO-RT evaluation failed: {e}")

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


def _evaluate_effective_ripple_series(vmec, s_vals, nc_file_path):
    """Evaluate the ripple metric on several representative surfaces."""
    results = []
    for s_val in s_vals:
        value, source = _evaluate_effective_ripple(vmec, float(s_val), nc_file_path)
        results.append({
            "s": float(s_val),
            "value": value,
            "source": source,
        })
    return results


def _compute_equilibrium_sanity(vmec, squid_info):
    """Basic numerical sanity checks for a single equilibrium."""
    fails, warns = [], []
    wout = vmec.wout

    scalar_checks = {
        "aspect": float(getattr(wout, "aspect", np.nan)),
        "Aminor_p": float(getattr(wout, "Aminor_p", np.nan)),
        "phi_edge": float(np.array(getattr(wout, "phi", [np.nan]))[-1]),
    }
    for name, value in scalar_checks.items():
        if not np.isfinite(value):
            fails.append(f"{name} is not finite")
    if np.isfinite(scalar_checks["Aminor_p"]) and scalar_checks["Aminor_p"] <= 0:
        fails.append("Aminor_p <= 0")

    iotaf = np.array(getattr(wout, "iotaf", []), dtype=float)
    if iotaf.size < 2:
        fails.append("iota profile missing or too short")
    elif not np.all(np.isfinite(iotaf[[0, -1]])):
        fails.append("iota axis/edge is not finite")

    for key in ("f_QI", "f_maxJ", "mirror_ratio"):
        value = float(squid_info.get(key, np.nan))
        if not np.isfinite(value):
            fails.append(f"{key} is not finite")

    if not squid_info.get("common_B_valid", True):
        fails.append("no common B* range across requested diagnostic surfaces")

    if not np.all(np.isfinite(iotaf)):
        warns.append("interior iota profile contains non-finite values")

    status = "OK"
    if fails:
        status = "FAIL"
    elif warns:
        status = "WARN"
    return {"status": status, "fails": fails, "warns": warns}


def _print_equilibrium_sanity(sanity):
    print(f"  Status: {sanity['status']}")
    for msg in sanity["fails"]:
        print(f"    FAIL: {msg}")
    for msg in sanity["warns"]:
        print(f"    WARN: {msg}")


def _sorted_surface_dict(metric_dict):
    """Return sorted (s, value) arrays from a {surface: value} dict."""
    items = [
        (float(k), float(v))
        for k, v in metric_dict.items()
        if np.isscalar(k)
    ]
    items.sort(key=lambda kv: kv[0])
    s = np.array([float(k) for k, _ in items], dtype=float)
    values = np.array([float(v) for _, v in items], dtype=float)
    return s, values


def _grade_qi(qi_surface_rms):
    if len(qi_surface_rms) == 0 or not np.any(np.isfinite(qi_surface_rms)):
        return "unknown"
    worst = float(np.nanmax(qi_surface_rms))
    if worst < 6e-3:
        return "good"
    if worst < 1.2e-2:
        return "watch"
    return "poor"


def _grade_maxj(pass_ratio):
    if not np.isfinite(pass_ratio):
        return "unknown"
    if pass_ratio >= 0.8:
        return "strong"
    if pass_ratio >= 0.6:
        return "mixed"
    return "weak"


def _grade_ripple(ripple_results):
    vals = [
        float(item["value"]) for item in ripple_results
        if item["value"] is not None and np.isfinite(item["value"])
    ]
    if not vals:
        return "unknown"
    worst = max(vals)
    if worst < 5e-3:
        return "good"
    if worst < 1e-2:
        return "watch"
    return "poor"


def _grade_well(well_depth):
    if not np.isfinite(well_depth):
        return "unknown"
    if well_depth > 0.01:
        return "good"
    if well_depth >= 0.0:
        return "watch"
    return "poor"


def _main_issue_summary(info, ripple_results=None, itg_info=None, ae_info=None):
    notes = []

    if len(info["s_vals"]) > 0:
        worst_qi_s = float(info["s_vals"][int(info["qi_worst_surface_idx"])])
        notes.append(f"QI worst at s={worst_qi_s:.2f}")

    if len(info["interval_centers"]) > 0 and info["maxj_worst_interval_idx"] >= 0:
        worst_interval = float(info["interval_centers"][int(info["maxj_worst_interval_idx"])])
        notes.append(
            f"max-J weakest near s~{worst_interval:.2f}, "
            f"shallow trapped depth λ_N~{info['maxj_worst_lambda']:.03f}"
        )

    if ripple_results:
        finite = [item for item in ripple_results
                  if item["value"] is not None and np.isfinite(item["value"])]
        if finite:
            worst = max(finite, key=lambda item: float(item["value"]))
            notes.append(f"ripple rises outward; worst checked at s={worst['s']:.2f}")

    if itg_info is not None and len(itg_info.get("per_surface", {})) > 0:
        itg_s, itg_vals = _sorted_surface_dict(itg_info["per_surface"])
        if len(itg_vals) > 0 and np.any(np.isfinite(itg_vals)):
            worst_idx = int(np.nanargmax(itg_vals))
            notes.append(f"ITG proxy peaks at s={itg_s[worst_idx]:.2f}")

    if ae_info is not None:
        ae_s, ae_vals = _sorted_surface_dict({k: v for k, v in ae_info.items() if k != "total"})
        if len(ae_vals) > 0 and np.any(np.isfinite(ae_vals)):
            worst_idx = int(np.nanargmax(ae_vals))
            notes.append(f"AE peaks at s={ae_s[worst_idx]:.2f}")

    return notes


def _overall_verdict(sanity, qi_grade, maxj_grade, ripple_grade):
    if sanity["status"] == "FAIL":
        return "fail"
    if maxj_grade == "weak" or ripple_grade == "poor":
        return "needs work"
    if qi_grade == "good" and maxj_grade in {"strong", "mixed"} and ripple_grade != "poor":
        return "promising"
    return "mixed"


def _plot_axis_geometry_summary(ax_info, mercier_data=None, well_data=None,
                                ripple_results=None):
    """Compact axis/stability summary figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    phi = ax_info["phi"]
    curvature = ax_info["curvature"]
    torsion = ax_info["torsion"]

    ax = axes[0, 0]
    ax.plot(phi, curvature, "k-", lw=1.5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("Axis curvature")

    ax = axes[0, 1]
    ax.plot(phi, torsion, "k-", lw=1.5)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\tau$")
    ax.grid(True, alpha=0.3)
    ax.set_title("Axis torsion")

    ax = axes[1, 0]
    handles, labels = [], []
    if mercier_data is not None:
        line = ax.plot(
            mercier_data["s"],
            mercier_data["values"],
            marker="o",
            lw=1.8,
            color="tab:blue",
            label=r"$D_{\mathrm{Merc}}$",
        )[0]
        ax.axhline(0, color="tab:blue", ls=":", alpha=0.4)
        handles.append(line)
        labels.append(line.get_label())
    ax.set_xlabel("s")
    ax.set_ylabel(r"$D_{\mathrm{Merc}}$")
    ax.grid(True, alpha=0.3)
    ax.set_title("Mercier and magnetic well")

    if well_data is not None:
        ax2 = ax.twinx()
        line2 = ax2.plot(
            well_data["s"],
            100.0 * well_data["values"],
            marker="s",
            lw=1.5,
            ls="--",
            color="tab:orange",
            label="well depth [%]",
        )[0]
        ax2.axhline(0, color="tab:orange", ls=":", alpha=0.4)
        ax2.set_ylabel("Well depth [%]")
        handles.append(line2)
        labels.append(line2.get_label())
    if handles:
        ax.legend(handles, labels, loc="best")

    ax = axes[1, 1]
    if ripple_results:
        s = np.array([item["s"] for item in ripple_results], dtype=float)
        y = np.array([
            np.nan if item["value"] is None else float(item["value"])
            for item in ripple_results
        ])
        ax.plot(s, y, marker="o", lw=1.8, color="tab:green")
        for idx, item in enumerate(ripple_results):
            src = item["source"]
            short = "DESC" if "DESC" in src else (
                "NEO-RT" if "NEO-RT" in src else (
                    "proxy" if "proxy" in src else "n/a"
                )
            )
            ax.annotate(short, (s[idx], y[idx]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)
    ax.set_xlabel("s")
    ax.set_ylabel("Ripple metric")
    ax.grid(True, alpha=0.3)
    ax.set_title("Effective ripple / proxy")

    fig.suptitle(
        f"Axis and stability summary (L_axis={ax_info['axis_length']:.3f} m)"
    )
    plt.tight_layout()
    return fig


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
    parser.add_argument("--extended", action="store_true",
                        help="Run third-tier transport diagnostics (ITG proxy, optional AE)")
    parser.add_argument("--extended_surfaces", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5],
                        help="Flux surfaces for third-tier ITG diagnostics")
    parser.add_argument("--eps_eff_surface", type=float, default=None,
                        help="Deprecated single-surface effective ripple evaluation")
    parser.add_argument("--eps_eff_surfaces", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75],
                        help="Normalised flux surfaces for effective ripple evaluation")
    parser.add_argument("--ae_surfaces", type=float, nargs="+",
                        default=[0.2, 0.5, 0.8],
                        help="Flux surfaces for AE diagnostics when --ae is enabled")
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

    print(f"\n--- Core SQuID Targets ---")
    info = evaluate_squid_detailed(
        vmec, s_vals=s_vals,
        num_alpha=args.num_alpha,
        num_pitch=args.num_pitch,
        verbose=False,
    )

    qi_grade = _grade_qi(info["qi_surface_rms"])
    maxj_grade = _grade_maxj(info["maxj_global_pass_ratio"])

    print(f"  f_QI             = {info['f_QI']:.6e}")
    print(f"  f_maxJ           = {info['f_maxJ']:.6e}")
    print(f"  mirror ratio     = {info['mirror_ratio']:.4f}")
    print(f"  QI grade         = {qi_grade}")
    print(f"  QI worst surface = s={info['s_vals'][info['qi_worst_surface_idx']]:.2f}")
    for s_val, rms, p95 in zip(info["s_vals"], info["qi_surface_rms"], info["qi_surface_p95"]):
        print(f"    QI @ s={s_val:.2f}: RMS={rms:.3e}, p95={p95:.3e}")

    if len(info["interval_centers"]) > 0:
        worst_interval_idx = int(info["maxj_worst_interval_idx"])
        print(f"  max-J grade      = {maxj_grade}")
        print(f"  max-J pass ratio = {info['maxj_global_pass_ratio']:.3f}")
        print(f"  max-J violation  = {info['maxj_global_violation_fraction']:.3f}")
        print("  max-J per interval:")
        for center, pass_ratio, frac, worst_lambda in zip(
            info["interval_centers"],
            info["maxj_interval_pass_ratio"],
            info["maxj_interval_violation_fraction"],
            info["maxj_interval_worst_lambda"],
        ):
            print(
                f"    s~{center:.2f}: pass={pass_ratio:.3f}, "
                f"violation={frac:.3f}, worst λ_N≈{worst_lambda:.3f}"
            )
        print(
            "  max-J worst interval = "
            f"s~{info['interval_centers'][worst_interval_idx]:.2f}"
        )

    print(f"\n--- Equilibrium Sanity ---")
    sanity = _compute_equilibrium_sanity(vmec, info)
    _print_equilibrium_sanity(sanity)

    print(f"\n--- Basic Geometry ---")
    print(f"  Aspect ratio = {vmec.wout.aspect:.4f}")
    print(f"  Iota (core)  = {vmec.wout.iotaf[0]:.4f}")
    print(f"  Iota (edge)  = {vmec.wout.iotaf[-1]:.4f}")

    if sanity["status"] == "FAIL":
        print("\nAborting further diagnostics: equilibrium failed sanity checks.")
        raise SystemExit(1)

    # ---------------------------------------------------------
    # NEW: Stability & Transport Diagnostics
    # ---------------------------------------------------------
    print(f"\n--- Stability & Transport Diagnostics ---")
    mercier_data = None
    well_data = None
    well_depth = float("nan")

    # 1. Mercier Stability Criterion
    try:
        dmerc = np.array(vmec.wout.Dmerc[1:], dtype=float)
        s_half = np.array(vmec.s_half_grid, dtype=float)
        mercier_data = {"s": s_half, "values": dmerc}
        min_idx = int(np.argmin(dmerc))
        min_dmerc = float(dmerc[min_idx])
        if min_dmerc > 0:
            print(
                "  Mercier Stability: "
                f"STABLE (min D_Merc = {min_dmerc:.4e} at s={s_half[min_idx]:.2f})"
            )
        else:
            print(
                "  Mercier Stability: "
                f"UNSTABLE (min D_Merc = {min_dmerc:.4e} at s={s_half[min_idx]:.2f})"
            )
    except Exception:
        print("  Mercier Stability: [Not available in this nc file]")

    # 2. Magnetic Well Depth
    try:
        vp = np.array(vmec.wout.vp[1:], dtype=float)
        well_profile = (vp[0] - vp) / max(vp[0], 1e-30)
        well_data = {"s": np.array(vmec.s_half_grid, dtype=float), "values": well_profile}
        well_depth = float(well_profile[-1])
        min_well_idx = int(np.argmin(well_profile))
        print(
            "  Magnetic Well Depth: "
            f"edge={well_depth * 100:.2f}%, "
            f"min={well_profile[min_well_idx] * 100:.2f}% at "
            f"s={well_data['s'][min_well_idx]:.2f}"
        )
    except Exception:
        print("  Magnetic Well Depth: [Not available in this nc file]")

    # 3. Effective Ripple (eps_eff)
    print("  Effective Ripple / Ripple Proxy:")
    eps_eff_surfaces = args.eps_eff_surfaces
    if args.eps_eff_surface is not None:
        eps_eff_surfaces = [args.eps_eff_surface]
    eps_eff_surfaces = sorted(set(max(0.01, min(0.99, float(s))) for s in eps_eff_surfaces))
    eps_eff_results = _evaluate_effective_ripple_series(vmec, eps_eff_surfaces, args.nc_file)
    for item in eps_eff_results:
        if item["value"] is None:
            print(f"    s={item['s']:.2f}: [Could not evaluate. {item['source']}]")
        else:
            print(f"    s={item['s']:.2f}: {item['value']:.4e}  ({item['source']})")
            if "Boozer proxy" in item["source"]:
                print("      [Note] This value is a ripple amplitude proxy, not the true ε_eff.")
            if "NEO-RT D11" in item["source"]:
                print("      [Note] This value is the NEO-RT transport coefficient D11, not the direct ε_eff.")
    ripple_grade = _grade_ripple(eps_eff_results)
    print(f"  Ripple grade: {ripple_grade}")

    # 4. Axis Geometry
    print(f"\n--- Axis Geometry ---")
    ax_info = None
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

    # ---------------------------------------------------------
    # Third-tier / extended diagnostics
    # ---------------------------------------------------------
    run_extended = args.extended or args.ae
    itg_info = None
    ae_info = None

    if run_extended:
        print(f"\n--- Extended Transport Diagnostics (reference only) ---")

        itg_surfaces = np.array(
            [max(0.01, min(0.99, float(s))) for s in args.extended_surfaces],
            dtype=float,
        )
        itg_surfaces = np.unique(itg_surfaces)
        itg_info = evaluate_itg(
            vmec,
            snorms=itg_surfaces,
            method=args.itg_method,
            verbose=False,
        )
        itg_s, itg_vals = _sorted_surface_dict(itg_info["per_surface"])
        print(f"  ITG proxy method = {args.itg_method}")
        print(f"  Total f_nabla_s  = {itg_info['total']:.6e}")
        if len(itg_vals) > 0 and np.any(np.isfinite(itg_vals)):
            worst_itg_idx = int(np.nanargmax(itg_vals))
            print(f"  Worst ITG surface = s={itg_s[worst_itg_idx]:.2f} ({itg_vals[worst_itg_idx]:.6e})")
        for s_itg, val_itg in zip(itg_s, itg_vals):
            print(f"    ITG @ s={s_itg:.2f}: {val_itg:.6e}")

        if args.ae:
            ae_surfaces = np.array(
                [max(0.01, min(0.99, float(s))) for s in args.ae_surfaces],
                dtype=float,
            )
            ae_surfaces = np.unique(ae_surfaces)
            print(f"\n  Available Energy (gradient-dependent reference diagnostic)")
            try:
                ae_info = ae_diagnostics(
                    vmec, s_vals=ae_surfaces,
                    omn=args.ae_omn, omt=args.ae_omt,
                    n_alpha=min(args.num_alpha, 4),
                    n_turns=3, lam_res=200, gridpoints=512,
                    verbose=False,
                )
                ae_s, ae_vals = _sorted_surface_dict({
                    k: v for k, v in ae_info.items() if k != "total"
                })
                if np.any(np.isfinite(ae_vals)):
                    worst_ae_idx = int(np.nanargmax(ae_vals))
                    print(
                        f"  Mean AE         = {ae_info['total']:.6e}\n"
                        f"  Worst AE surface = s={ae_s[worst_ae_idx]:.2f} ({ae_vals[worst_ae_idx]:.6e})"
                    )
                for s_ae, val_ae in zip(ae_s, ae_vals):
                    print(f"    AE @ s={s_ae:.2f}: {val_ae:.6e}")
            except Exception as e:
                print(f"  [AE failed] {e}")
        else:
            print("  AE skipped. Use --ae to include TEM-oriented diagnostics.")

    well_grade = _grade_well(well_depth)
    verdict = _overall_verdict(sanity, qi_grade, maxj_grade, ripple_grade)
    notes = _main_issue_summary(info, ripple_results=eps_eff_results,
                                itg_info=itg_info, ae_info=ae_info)

    print(f"\n--- Diagnostic Summary ---")
    print(f"  Overall verdict = {verdict}")
    print(f"  Core grades     = QI:{qi_grade}, max-J:{maxj_grade}, ripple:{ripple_grade}, well:{well_grade}")
    if sanity['status'] != 'OK':
        print(f"  Sanity status    = {sanity['status']}")
    for note in notes:
        print(f"  Note            = {note}")

    # Plots
    if args.plot:
        import matplotlib.pyplot as plt
        print(f"\n--- Generating plots ---")

        # ---------------------------------------------------------
        # 替换后的 Boozer Surface 画图代码
        # ---------------------------------------------------------
        boozer_s_vals = [0.25, 0.5, 0.75, 1.0] # 对应图中的4个面
        fig1, axes1 = plt.subplots(2, 2, figsize=(8, 5.5)) # 改为 2x2 布局
        axes_flat = axes1.flatten()
        from squid.evaluation.evaluate import run_boozer, reconstruct_B
        
        nfp = int(vmec.wout.nfp)
        safe_s = [max(0.01, min(0.99, s)) for s in boozer_s_vals]
        _, all_surf = run_boozer(vmec, safe_s, mpol=20, ntor=20)
        
        ntheta, nphi = 100, 100
        th = np.linspace(0, 2 * np.pi, ntheta)
        ze = np.linspace(0, 2 * np.pi / nfp, nphi)
        TH, ZE = np.meshgrid(th, ze, indexing="ij")
        b_all = [reconstruct_B(d["m"], d["n"], d["bmnc"], TH, ZE) for d in all_surf]
        
        for i, (s, data, B_2d) in enumerate(zip(boozer_s_vals, all_surf, b_all)):
            ax = axes_flat[i]
            
            # 动态计算 level，大约18条线，防止过于密集
            levels = np.linspace(B_2d.min(), B_2d.max(), 18)
            
            # 使用 contour 代替 contourf，cmap 采用 'plasma' 以匹配原图紫色到黄色的渐变
            cs = ax.contour(ZE, TH, B_2d, levels=levels, cmap="plasma", linewidths=1.2)
            
            # 添加左上角的标签文本框 (例如: |B| @ s=0.25)
            # 根据图中s=1没有小数位的情况做一点格式化
            s_label = f"{s:g}" if s == 1.0 else f"{s}"
            ax.text(0.03, 0.95, f"|B| @ s={s_label}", transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
            # 设置刻度以匹配原图
            ax.set_xticks([0, 2 * np.pi / nfp])
            # 图中 x 轴右侧刻度为 \pi/2，这通常对应 nfp=4 的情况。这里做个动态适配
            if nfp == 4:
                ax.set_xticklabels(['0', r'$\pi/2$'], fontsize=11)
            else:
                ax.set_xticklabels(['0', rf'$2\pi/{nfp}$'], fontsize=11)
                
            ax.set_yticks([0, 2 * np.pi])
            ax.set_yticklabels(['0', r'$2\pi$'], fontsize=11)
            
            # 坐标轴标签
            ax.set_xlabel(r"$\phi$", fontsize=12, labelpad=-8, fontweight='bold')
            ax.set_ylabel(r"$\theta$", fontsize=12, labelpad=-5, fontweight='bold')
            
            # 为每个子图添加单独的 colorbar
            cbar = fig1.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            
        fig1.tight_layout()
        fig1.savefig("boozer_surface.png", dpi=150, bbox_inches="tight")
        print("  Saved: boozer_surface.png")

        fig2 = plot_squid_core_diagnostics(
            info,
            metadata=dict(
                num_surfaces=len(s_vals),
                num_alpha=args.num_alpha,
                num_pitch=args.num_pitch,
            ),
        )
        fig2.savefig("squid_core_diagnostics.png", dpi=150, bbox_inches="tight")
        print("  Saved: squid_core_diagnostics.png")

        if run_extended:
            fig_ext = plot_transport_diagnostics(
                itg_info=itg_info,
                ae_info=ae_info,
                itg_method=args.itg_method,
                metadata=dict(
                    itg_surfaces=",".join(f"{s:.2f}" for s in itg_surfaces),
                    ae_surfaces=",".join(f"{s:.2f}" for s in ae_surfaces) if args.ae else "off",
                ),
            )
            if fig_ext is not None:
                fig_ext.savefig("transport_diagnostics.png", dpi=150, bbox_inches="tight")
                print("  Saved: transport_diagnostics.png")

        print("  Computing J contour polar plot (Fig. 9) ...")
        fig3 = plot_J_contours(vmec, lambda_N=0.3)
        fig3.savefig("j_contours_polar.png", dpi=150, bbox_inches="tight")
        print("  Saved: j_contours_polar.png")

        try:
            if ax_info is None:
                raise RuntimeError("axis geometry unavailable")
            fig4 = _plot_axis_geometry_summary(
                ax_info,
                mercier_data=mercier_data,
                well_data=well_data,
                ripple_results=eps_eff_results,
            )
            fig4.savefig("axis_geometry.png", dpi=150, bbox_inches="tight")
            print("  Saved: axis_geometry.png")
        except Exception as e:
            print(f"  [axis_geometry plot failed: {e}]")

        plt.close("all")

    print(f"\n{'=' * 60}")
    print("Diagnostic evaluation complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
