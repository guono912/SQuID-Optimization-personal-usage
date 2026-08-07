"""External MHD gate: DESC force-balance + ballooning + effective ripple.

The promotion gate for SQuID campaigns (see skill/skill_04_gates_promotion.md).
This module owns the gate decision logic; CLI entry points in
``squid/cli/gates/mhd.py`` and the legacy ``scripts/mhd_gate.py`` wrapper call
into it.
"""

import json
import time
from pathlib import Path

import numpy as np

from ..diagnostics.mercier_normalization import mercier_summary


def _compute_ballooning_envelope(eq, nfp, rho_v=None, alpha_v=None, nzeta0=5, shift=1.0):
    """DESC ballooning λ(ρ,α) with raw eigenvalues via ReLU(λ+shift) trick."""
    from desc.objectives import BallooningStability

    if rho_v is None:
        rho_v = np.array([0.1, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95])
    if alpha_v is None:
        alpha_v = np.linspace(0, np.pi, 8, endpoint=False)
    zeta0 = np.linspace(-0.5*np.pi, 0.5*np.pi, nzeta0)

    lam_grid = np.zeros((len(rho_v), len(alpha_v)))
    t0 = time.time()
    for j, a in enumerate(alpha_v):
        obj = BallooningStability(
            eq, rho=rho_v, alpha=np.array([a]),
            nturns=2, nzetaperturn=80, zeta0=zeta0,
            Neigvals=1, lambda0=-shift, w0=0.0, w1=1.0,
        )
        obj.build(use_jit=False, verbose=0)
        val = np.asarray(obj.compute(eq.params_dict), dtype=float)
        lam_grid[:, j] = val.ravel()[:len(rho_v)] - shift

    elapsed = time.time() - t0
    lam_env = np.max(lam_grid, axis=1)
    n_pos = int(np.sum(lam_grid > 0))
    lam_max = float(np.max(lam_grid))
    lam_min = float(np.min(lam_grid))

    return {
        "rho": rho_v.tolist(),
        "alpha": alpha_v.tolist(),
        "lam_grid": lam_grid.tolist(),
        "lam_envelope": lam_env.tolist(),
        "n_unstable": n_pos,
        "lam_max": lam_max,
        "lam_min": lam_min,
        "elapsed_s": elapsed,
    }, lam_grid


def _compute_effective_ripple(eq, s_vals=None):
    """DESC ε_eff via Nemov bounce-integral."""
    from desc.grid import LinearGrid
    from desc.objectives import EffectiveRipple

    if s_vals is None:
        s_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    results = {}
    failed_surfaces = []
    t0 = time.time()
    for s in s_vals:
        try:
            rho = np.array([np.sqrt(s)])
            grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid,
                             NFP=int(eq.NFP), sym=False)
            obj = EffectiveRipple(eq, grid=grid, num_transit=10, num_pitch=31)
            obj.build(verbose=0)
            eps = float(obj.compute(eq.params_dict)[0])
            results[f"s={s:.1f}"] = eps if np.isfinite(eps) else None
        except Exception as exc:
            results[f"s={s:.1f}"] = None
            failed_surfaces.append({"s": float(s), "error": repr(exc)})

    elapsed = time.time() - t0
    peak = max((v for v in results.values() if v is not None), default=None)
    return {
        "per_surface": results,
        "peak": peak,
        "failed_surfaces": failed_surfaces,
        "failed": bool(failed_surfaces),
        "elapsed_s": elapsed,
    }


def gate(wout_path, output_dir="runs/mhd_gate_latest", L=6, M=6, N=6,
         rho_v=None, alpha_v=None, protocol=None):
    """Run full MHD gate on a VMEC wout file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wout_path).stem

    print(f"\n{'='*60}")
    print(f"MHD Gate: {stem}")
    print(f"{'='*60}")

    # ── VMEC quick check ──
    import netCDF4
    ds = netCDF4.Dataset(wout_path, 'r')
    vmec_info = {
        "ier_flag": int(ds.variables["ier_flag"][:]),
        "nfp": int(ds.variables["nfp"][:]),
        "ns": int(ds.variables["ns"][:]),
        "aspect": float(ds.variables["aspect"][:]),
        "betatotal": float(ds.variables["betatotal"][:]),
        "Rmajor_p": float(ds.variables["Rmajor_p"][:]),
        "Aminor_p": float(ds.variables["Aminor_p"][:]),
        "fsqr": float(ds.variables["fsqr"][:]),
        "fsqz": float(ds.variables["fsqz"][:]),
        "iotaf": [float(ds.variables["iotaf"][0]), float(ds.variables["iotaf"][-1])],
    }
    nfp = vmec_info["nfp"]

    # Mercier
    mercier = mercier_summary(ds, s_min=0.1, s_max=0.97)
    vmec_info.update({
        "dmerc_convention": mercier["convention"],
        "dmerc_formula": mercier["formula"],
        "dmerc_edge_toroidal_flux_wb": mercier["edge_toroidal_flux_wb"],
        "dmerc_vmec_raw_min_gated": mercier["dmerc_vmec_raw_min"],
        "dmerc_flux_normalized_min_gated": mercier["dmerc_flux_normalized_min"],
        "dmerc_min_s": mercier["minimum_s"],
        "dmerc_s_min": mercier["s_min"],
        "dmerc_s_max": mercier["s_max"],
        "dmerc_negative_count": mercier["dmerc_negative_count"],
    })

    # Well
    vp = np.array(ds.variables["vp"][:])
    vmec_info["well_depth"] = float((vp[1] - vp[-1]) / max(abs(vp[1]), 1e-30))

    ds.close()
    print(f"  VMEC: ier={vmec_info['ier_flag']}, β={vmec_info['betatotal']*100:.2f}%, "
          f"ι=[{vmec_info['iotaf'][0]:.3f},{vmec_info['iotaf'][1]:.3f}], "
          f"Phi_edge^2*DMerc_min={vmec_info['dmerc_flux_normalized_min_gated']:.3e}, "
          f"raw={vmec_info['dmerc_vmec_raw_min_gated']:.3e}, "
          f"neg={vmec_info['dmerc_negative_count']}, well={vmec_info['well_depth']*100:.2f}%")

    # ── DESC force balance ──
    print(f"\n  DESC force-balance (L={L},M={M},N={N})...")
    t0 = time.time()
    from desc.vmec import VMECIO
    from desc.objectives import get_fixed_boundary_constraints, ForceBalance

    eq = VMECIO.load(wout_path, L=L, M=M, N=N)

    # Initial force error
    fb = ForceBalance(eq=eq)
    fb.build(verbose=0)
    fb_init = np.array(fb.compute(eq.params_dict))
    fb_rms_init = float(np.sqrt(np.mean(fb_init**2)))

    # Solve
    eq.solve(objective="force",
             constraints=get_fixed_boundary_constraints(eq=eq, profiles=True, normalize=True),
             optimizer="lsq-exact", maxiter=100, verbose=0)

    fb = ForceBalance(eq=eq)
    fb.build(verbose=0)
    fb_final = np.array(fb.compute(eq.params_dict))
    fb_rms_final = float(np.sqrt(np.mean(fb_final**2)))
    fb_elapsed = time.time() - t0
    print(f"    FB: {fb_rms_init:.2e} → {fb_rms_final:.2e}  ({fb_elapsed:.0f}s)")

    # Iota from DESC
    iota_desc = eq.compute("iota")["iota"]
    iota_desc_arr = np.asarray(iota_desc).ravel()
    print(f"    DESC ι: [{iota_desc_arr[0]:.4f}, {iota_desc_arr[-1]:.4f}]")

    # ── Ballooning ──
    print(f"\n  Ballooning...")
    bal_info, lam_grid = _compute_ballooning_envelope(
        eq, nfp, rho_v=rho_v, alpha_v=alpha_v)
    if protocol:
        bal_info["protocol"] = protocol
    status = "PASS" if bal_info["n_unstable"] == 0 else f"{bal_info['n_unstable']} unstable"
    print(f"    λ_max={bal_info['lam_max']:.2e}, λ_min={bal_info['lam_min']:.2e}, "
          f"{status}  ({bal_info['elapsed_s']:.0f}s)")
    print(f"    λ(ρ) envelope: {[f'{v:.1e}' for v in bal_info['lam_envelope']]}")

    # ── Effective ripple ──
    print(f"\n  Effective ripple...")
    ripple_info = _compute_effective_ripple(eq)
    peak = ripple_info["peak"]
    print(f"    ε_eff peak: {peak:.4f}" if peak else "    ε_eff: FAILED")
    print(f"    ({ripple_info['elapsed_s']:.0f}s)")

    # ── Gate verdict ──
    issues = []
    if vmec_info["ier_flag"] != 0:
        issues.append(f"VMEC ier_flag={vmec_info['ier_flag']}")
    for name in ("fsqr", "fsqz"):
        if not np.isfinite(vmec_info[name]):
            issues.append(f"VMEC {name} is non-finite")
    if not np.isfinite(fb_rms_final):
        issues.append("DESC force-balance residual is non-finite")
    if (np.isfinite(fb_rms_init) and np.isfinite(fb_rms_final)
            and fb_rms_final > fb_rms_init * (1 + 1e-6)):
        issues.append(
            f"DESC force-balance worsened: {fb_rms_init:.3e} -> {fb_rms_final:.3e}"
        )
    if vmec_info["dmerc_negative_count"] > 0:
        issues.append(f"DMerc has {vmec_info['dmerc_negative_count']} negative points")
    if bal_info["n_unstable"] > 10:
        issues.append(f"Ballooning: {bal_info['n_unstable']} unstable points")
    elif bal_info["n_unstable"] > 0:
        issues.append(f"Ballooning: {bal_info['n_unstable']} unstable (marginal)")
    if peak and peak > 0.05:
        issues.append(f"ε_eff peak {peak:.3f} > 0.05")
    if ripple_info["failed"]:
        issues.append(
            f"Effective ripple failed on {len(ripple_info['failed_surfaces'])} surfaces"
        )
    if vmec_info["well_depth"] < 0.03:
        issues.append(f"Well depth {vmec_info['well_depth']*100:.1f}% < 3%")

    verdict = "PASS" if not issues else "WARN"
    print(f"\n  Gate: {verdict}")
    for issue in issues:
        print(f"    - {issue}")

    # ── Report ──
    report = {
        "wout": wout_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "issues": issues,
        "vmec": vmec_info,
        "desc": {
            "L": L, "M": M, "N": N,
            "fb_rms_init": fb_rms_init,
            "fb_rms_final": fb_rms_final,
            "fb_elapsed_s": fb_elapsed,
            "iota": [float(iota_desc_arr[0]), float(iota_desc_arr[-1])],
        },
        "ballooning": bal_info,
        "ripple": ripple_info,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{stem}_{ts}_gate.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else float(x))
    print(f"\n  Saved: {report_path}")

    # Save DESC equilibrium as .h5
    try:
        from desc.equilibrium import Equilibrium
        h5_path = output_dir / f"{stem}_{ts}_desc.h5"
        eq.save(h5_path)
        print(f"  DESC eq saved: {h5_path}")
    except Exception as e:
        print(f"  DESC .h5 save failed: {e}")

    return report
