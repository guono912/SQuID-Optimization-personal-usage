#!/usr/bin/env python3
"""Iterative QI refinement CLI implementation.

Reusable entry point: scripts/opt/refine_qi.py and the legacy
scripts/refine_qi.py wrapper both call into this module. The optimization
loop (random hill-climbing with DESC force balance) runs inside main();
it used to execute at module import time.
"""

import os, sys, time, json, shutil, csv, argparse
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from desc.vmec import VMECIO
from desc.objectives import (
    get_fixed_boundary_constraints, ForceBalance, BallooningStability,
)
from simsopt.mhd import Vmec
from ..objectives.maxj_residual import _evaluate_squid
from ..objectives.penalties import hinge_loss
from ..diagnostics.mercier_normalization import (
    edge_toroidal_flux_wb,
    flux_normalize_mercier,
    mercier_interior_profile,
)

def validate_settings(settings):
    fields = {
        "weights": {"w_qi", "w_maxj", "w_bmin", "w_ar", "w_reg", "w_mirror",
                    "w_beta", "w_iota", "w_well", "w_mercier_margin",
                    "w_ballooning", "w_force_balance"},
        "targets": {"aspect_target", "mirror_target", "beta_target", "iota_ax",
                    "iota_edge", "iota_tolerance", "mercier_flux_normalized_margin_target",
                    "target_well", "ballooning_lambda_target", "maxj_target", "bmin_slope_target"},
    }
    if not isinstance(settings, dict) or settings.keys() != fields.keys():
        raise ValueError("refinement config requires weights and targets")
    for group, keys in fields.items():
        values = settings[group]
        if not isinstance(values, dict) or values.keys() != keys:
            raise ValueError(f"{group} requires exactly: {', '.join(sorted(keys))}")
        for key, value in values.items():
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if group == "weights" and value < 0:
                raise ValueError(f"{key} must be nonnegative")


# ── Config ──
def main(argv=None):
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--nc-file", default=os.environ.get("SQUID_REFINE_NC_FILE"),
                         help="input equilibrium; required unless SQUID_REFINE_NC_FILE is set")
    _parser.add_argument("--run-dir", default=os.environ.get("SQUID_REFINE_RUN_DIR"),
                         help="new output directory; required unless SQUID_REFINE_RUN_DIR is set")
    _parser.add_argument("--config", required=True, help="JSON refinement weights and targets")
    _parser.add_argument("--n-trials", type=int, default=6)
    _parser.add_argument("--n-iter", type=int, default=6)
    _args = _parser.parse_args(argv)
    if not _args.nc_file:
        _parser.error("--nc-file is required")
    if not _args.run_dir:
        _parser.error("--run-dir is required")

    NC_FILE = os.path.abspath(_args.nc_file)
    RUN_DIR = os.path.abspath(_args.run_dir)
    N_TRIALS = _args.n_trials       # random perturbations per iteration
    N_ITER = _args.n_iter           # number of iterations
    DOF_FRAC = 0.025    # max fractional change per DoF
    L = 6; M = 6; N = 6

    try:
        with open(_args.config) as handle:
            settings = json.load(handle)
        validate_settings(settings)
    except (OSError, ValueError) as exc:
        _parser.error(str(exc))
    W, TGT = settings["weights"], settings["targets"]

    S_VALS = np.linspace(0.2, 0.8, 4)
    ALPHAS = np.linspace(0, 2*np.pi, 6, endpoint=False)
    B_RHOS = np.array([0.5, 0.65, 0.8, 0.9])
    B_ALPHAS = np.linspace(0, np.pi, 6, endpoint=False)
    B_ZETA0 = np.linspace(-0.5*np.pi, 0.5*np.pi, 5)

    if N_ITER < 0 or N_TRIALS < 0:
        _parser.error("iteration and trial counts must be nonnegative")
    os.makedirs(RUN_DIR, exist_ok=True)
    if os.listdir(RUN_DIR):
        _parser.error("choose an empty refinement directory")

    with open(os.path.join(RUN_DIR, "refinement_settings.json"), "x") as handle:
        json.dump(settings, handle, indent=2)

    # ── Load initial equilibrium ──
    print(f"Loading: {NC_FILE}")
    eq0 = VMECIO.load(NC_FILE, L=L, M=M, N=N)
    surf = eq0.surface
    R_modes, Z_modes = surf.R_basis.modes, surf.Z_basis.modes

    # Free DoFs (same order as desc_backend)
    free_idx_R, free_idx_Z, free_names = [], [], []
    max_dofs = 20
    for m in [1, 0, 2, 3]:
        for n in [0, 1, -1, 2, -2, 3, -3]:
            if m == 0 and n == 0: continue
            if len(free_names) >= max_dofs: break
            for modes, idx_list, prefix in [(R_modes, free_idx_R, "R"), (Z_modes, free_idx_Z, "Z")]:
                if len(free_names) >= max_dofs: break
                for idx, mode in enumerate(modes):
                    if int(mode[1]) == m and int(mode[2]) == n:
                        idx_list.append(idx)
                        free_names.append(f"{prefix}({m},{n})")
                        break

    free_idx_R = np.array(free_idx_R, dtype=int)
    free_idx_Z = np.array(free_idx_Z, dtype=int)
    nR = len(free_idx_R)
    print(f"Free DoFs: {len(free_names)}")

    R0_arr = np.asarray(surf.R_lmn).copy()
    Z0_arr = np.asarray(surf.Z_lmn).copy()
    x0 = np.concatenate([R0_arr[free_idx_R], Z0_arr[free_idx_Z]])
    n_dof = len(x0)

    wout_tmp = os.path.join(RUN_DIR, "wout_trial.nc")

    # ── History ──
    history_path = os.path.join(RUN_DIR, "history.csv")
    with open(history_path, "x", newline="") as fh:
        csv.DictWriter(fh, fieldnames=["iter","trial","f_QI","f_maxJ","f_Bmin",
            "balloon_n","balloon_lam","dmerc_edge_toroidal_flux_wb",
            "dmerc_vmec_raw_min","dmerc_flux_normalized_min","dmerc_neg","fb_rms",
            "delta","beta","iota_ax","iota_ed","well","total","accept","dt"]).writeheader()

    def evaluate(x, label=""):
        """Full DESC force-balance + SQuID + MHD evaluation."""
        t0 = time.time()
        new_R = R0_arr.copy(); new_Z = Z0_arr.copy()
        new_R[free_idx_R] = x[:nR]; new_Z[free_idx_Z] = x[nR:]
        surf.R_lmn = new_R; surf.Z_lmn = new_Z

        # DESC solve
        try:
            constraints = get_fixed_boundary_constraints(eq=eq0, profiles=True, normalize=True)
            _, solve_result = eq0.solve(objective="force", constraints=constraints,
                      optimizer="lsq-exact", maxiter=50, verbose=0)
            if not solve_result["success"]:
                return None
            fb_obj = ForceBalance(eq=eq0); fb_obj.build(use_jit=False, verbose=0)
            fb_rms = float(np.sqrt(np.mean(np.asarray(fb_obj.compute(eq0.params_dict))**2)))
            if not np.isfinite(fb_rms):
                return None
            if fb_rms > 1e6 and hasattr(evaluate, 'fb_rms0'):
                return None  # reject
            if not hasattr(evaluate, 'fb_rms0'):
                evaluate.fb_rms0 = max(fb_rms, 1.0)
            elif fb_rms > 500 * evaluate.fb_rms0:
                return None
        except Exception as exc:
            print(f"  {label} DESC solve FAILED: {exc}")
            return None

        # VMEC bridge + SQuID
        try:
            VMECIO.save(eq0, wout_tmp, surfs=31, verbose=0)
            vmec = Vmec(wout_tmp); vmec.run()
            info = _evaluate_squid(vmec, S_VALS, ALPHAS, 20, TGT["maxj_target"], 8, 8)
        except Exception as exc:
            print(f"  {label} SQuID FAILED: {exc}")
            return None

        # Ballooning
        bal_n, bal_lam = -1, np.nan
        f_bal = 0.0
        try:
            lam_res = []; bal_n = 0; bal_lam = -999
            for a in B_ALPHAS:
                obj = BallooningStability(eq0, rho=B_RHOS, alpha=np.array([a]),
                    nturns=2, nzetaperturn=80, zeta0=B_ZETA0,
                    Neigvals=1, lambda0=-1.0, w0=0.0, w1=1.0)
                obj.build(use_jit=False, verbose=0)
                v = np.asarray(obj.compute(eq0.params_dict), dtype=float).ravel() - 1.0
                bal_lam = max(bal_lam, float(np.max(v)))
                bal_n += int(np.sum(v > 0))
                lam_res.extend(np.maximum(v - TGT["ballooning_lambda_target"], 0.0).tolist())
            f_bal = float(np.mean(np.array(lam_res)**2))
        except Exception:
            f_bal = 1e4

        # Mercier
        dm_raw_min, dm_fluxnorm_min, dm_phi_edge, dm_neg = np.nan, np.nan, np.nan, 0
        try:
            dm_raw_full = np.array(vmec.wout.DMerc, dtype=float)
            dm_full = flux_normalize_mercier(dm_raw_full, vmec.wout)
            dm_phi_edge = edge_toroidal_flux_wb(vmec.wout)
            s_dm, dm_raw = mercier_interior_profile(dm_raw_full)
            _, dm = mercier_interior_profile(dm_full, ns=dm_raw_full.size)
            mask = (s_dm >= 0.1) & (s_dm <= 0.95) & np.isfinite(dm_raw)
            vals = dm[mask]
            dm_raw_min = float(np.min(dm_raw[mask])) if vals.size else np.nan
            dm_fluxnorm_min = float(np.min(vals)) if vals.size else np.nan
            dm_neg = int(np.sum(vals < 0)) if vals.size else 0
            mm_res = np.maximum(
                TGT['mercier_flux_normalized_margin_target'] - vals, 0.0
            )
            f_mm = float(np.mean(mm_res**2)) if vals.size else 1e6
        except Exception:
            f_mm = 1e6

        # Other penalties
        f_QI = info["f_QI"]; f_maxJ = info["f_maxJ"]
        bmin_res = np.zeros(1)
        try:
            from ..objectives.penalties import bmin_slope_residuals
            bmin_res = bmin_slope_residuals(S_VALS, info["surface_Bmin"], TGT["bmin_slope_target"])
        except: pass
        f_Bmin = float(np.sum(bmin_res**2))
        delta = info["mirror_ratio"]
        iota_ax, iota_ed = info["iota_axis"], info["iota_edge"]
        try: aspect = float(vmec.aspect())
        except: aspect = 10.0
        try: beta = float(vmec.wout.betatotal)
        except: beta = np.inf
        f_mirror = hinge_loss(delta, TGT['mirror_target'])
        f_beta = hinge_loss(beta, TGT['beta_target']) if np.isfinite(beta) else 1e6
        f_iota = hinge_loss(abs(iota_ax - TGT['iota_ax']), TGT['iota_tolerance']) + \
                 hinge_loss(abs(iota_ed - TGT['iota_edge']), TGT['iota_tolerance'])
        try:
            vp = np.array(vmec.wout.vp); well = float((vp[1]-vp[-1])/max(abs(vp[1]),1e-30))
            f_well = hinge_loss(TGT['target_well'] - well, 0.0)
        except: well = np.nan; f_well = 1e6
        f_fb = max(fb_rms / evaluate.fb_rms0 - 1.0, 0.0) if hasattr(evaluate, 'fb_rms0') else 0.0
        f_reg = float(np.sum((x - x0)**2))

        total = (W['w_qi']*f_QI + W['w_maxj']*f_maxJ + W['w_bmin']*f_Bmin
                 + W['w_ar']*hinge_loss(aspect, TGT['aspect_target'])
                 + W['w_mirror']*f_mirror + W['w_beta']*f_beta
                 + W['w_iota']*f_iota + W['w_well']*f_well
                 + W['w_mercier_margin']*f_mm + W['w_ballooning']*f_bal
                 + W['w_force_balance']*f_fb + W['w_reg']*f_reg)

        dt = time.time() - t0
        row = dict(iter=0, trial=0, f_QI=f_QI, f_maxJ=f_maxJ, f_Bmin=f_Bmin,
                   balloon_n=bal_n, balloon_lam=bal_lam,
                   dmerc_edge_toroidal_flux_wb=dm_phi_edge,
                   dmerc_vmec_raw_min=dm_raw_min,
                   dmerc_flux_normalized_min=dm_fluxnorm_min,
                   dmerc_neg=dm_neg, fb_rms=fb_rms, delta=delta, beta=beta,
                   iota_ax=iota_ax, iota_ed=iota_ed, well=well, total=total,
                   accept=0, dt=dt)
        return (row, x) if np.isfinite(total) else None

    # ── Initial evaluation ──
    print("\n=== Initial evaluation ===")
    def finish(state, best=None, candidate=None):
        status = {"state": state, "accepted": state == "improved",
                  "best_objective": best, "output_file": candidate}
        with open(os.path.join(RUN_DIR, "run_status.json"), "w") as fh:
            json.dump(status, fh, indent=2, allow_nan=False)
        print(f"Refinement finished: {state}")
        return 1 if state == "initial_failed" else 0

    initial = evaluate(x0, "init")
    if initial is None:
        return finish("initial_failed")
    res0, x_cur = initial
    best_path = os.path.join(RUN_DIR, "wout_refinement_best.nc")
    shutil.copyfile(wout_tmp, best_path)
    evaluate.fb_rms0 = max(res0['fb_rms'], 1.0)
    res0['iter'] = 0; res0['trial'] = 0; res0['accept'] = 1
    print(f"  f_QI={res0['f_QI']:.4e}  f_maxJ={res0['f_maxJ']:.4e}  "
          f"bal_n={res0['balloon_n']}  PhiEdge^2*DMerc="
          f"{res0['dmerc_flux_normalized_min']:.2e}({res0['dmerc_neg']}n)  "
          f"total={res0['total']:.4e}")
    with open(history_path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=list(res0.keys())).writerow(res0)

    best_total = res0['total']
    x_best = x_cur.copy()

    # ── Iterative random search ──
    rng = np.random.default_rng(42)
    for it in range(1, N_ITER + 1):
        print(f"\n--- Iteration {it}/{N_ITER} (best total={best_total:.4e}) ---")
        improved = False
        for trial in range(1, N_TRIALS + 1):
            # Random perturbation within DOF_FRAC
            dx_max = DOF_FRAC * np.maximum(np.abs(x_best), 1e-4)
            dx = rng.uniform(-1, 1, n_dof) * dx_max
            x_try = x_best + dx

            result = evaluate(x_try, f"it{it}t{trial}")
            if result is None:
                print(f"  [{it}.{trial}] REJECTED (bad solve)")
                continue
            res, _ = result

            res['iter'] = it; res['trial'] = trial
            accept = 1 if res['total'] < best_total else 0
            res['accept'] = accept
            with open(history_path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=list(res.keys())).writerow(res)

            status = "ACCEPT" if accept else "reject"
            print(f"  [{it}.{trial}] {status}  f_QI={res['f_QI']:.4e}  "
                  f"f_maxJ={res['f_maxJ']:.4e}  bal_n={res['balloon_n']}  "
                  f"PhiEdge^2*DMerc={res['dmerc_flux_normalized_min']:.2e}  "
                  f"total={res['total']:.4e}  ({res['dt']:.1f}s)")

            if accept:
                os.replace(wout_tmp, best_path)
                best_total = res['total']
                x_best = result[1].copy()
                improved = True

        if not improved:
            print(f"  No improvement this iteration.")

        checkpoint = os.path.join(RUN_DIR, f"wout_desc_iter{it:02d}.nc")
        shutil.copyfile(best_path, checkpoint)

    state = "improved" if best_total < res0["total"] else "baseline_only"
    return finish(state, float(best_total), os.path.basename(best_path))

if __name__ == "__main__":
    sys.exit(main())
