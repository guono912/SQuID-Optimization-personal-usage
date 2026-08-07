"""
DESC backend for SQuID optimisation — force-balanced equilibrium.

Flow per eval:
  1. DESC force-balance solve  (~25 s)
  2. VMECIO.save → temp wout
  3. Vmec(wout) → VMEC run → SQuID core (QI / maxJ / Bmin)
  4. Ballooning + Mercier + FB evaluated directly on DESC eq
  5. Least-squares objective assembled from all residual vectors

Uses scipy Nelder-Mead (derivative-free, robust for noisy DESC solves).
"""

import os, sys, time, json, csv, tempfile, shutil
import numpy as np

from simsopt.mhd import Vmec
from scipy.optimize import minimize

from ..objectives.maxj_residual import _evaluate_squid
from ..objectives.qi_residual import (
    compute_highB_contour_residual,
    compute_qi_residual_r2,
    QIResidual,
)
from ..objectives.itg_residual import ITGResidual
from ..objectives.penalties import bmin_slope_residuals, hinge_loss
from ..diagnostics.mercier_normalization import (
    MERCIER_CONVENTION_VERSION,
    edge_toroidal_flux_wb,
    flux_normalize_mercier,
    vmec_half_grid_profile,
)

INVALID = 1.0e6
INVALID_TOTAL = 1.0e10


def _compute_mirror_penalty(mirror_ratio, mirror_target):
    return hinge_loss(mirror_ratio, mirror_target)


def _compute_beta_penalty(beta, beta_target):
    if not np.isfinite(beta):
        return INVALID
    return hinge_loss(beta, beta_target)


def _compute_iota_penalty(iota_axis, iota_edge, iota_ax_target,
                          iota_edge_target, tolerance=0.01,
                          edge_mode="target"):
    axis_penalty = 0.0
    if iota_ax_target is not None:
        axis_penalty = hinge_loss(abs(iota_axis - iota_ax_target), tolerance)
    if edge_mode == "upper":
        edge_penalty = hinge_loss(iota_edge - iota_edge_target, tolerance)
    elif edge_mode == "lower":
        edge_penalty = hinge_loss(iota_edge_target - iota_edge, tolerance)
    else:
        edge_penalty = hinge_loss(abs(iota_edge - iota_edge_target), tolerance)
    return axis_penalty + edge_penalty


def run_desc(args):
    from desc.vmec import VMECIO

    print(f"\n  Backend: DESC (force-balance + VMEC bridge for SQuID)")

    # ── Load initial equilibrium ──
    print(f"  Loading: {args.nc_file}")
    eq = VMECIO.load(args.nc_file, L=args.desc_L, M=args.desc_M, N=args.desc_N)
    print(f"  Resolution: L={eq.L} M={eq.M} N={eq.N} NFP={int(eq.NFP)}")

    # ── Free boundary DoFs ──
    surf = eq.surface
    R_modes = surf.R_basis.modes
    Z_modes = surf.Z_basis.modes

    free_idx_R, free_idx_Z, free_names = [], [], []
    for m in [1, 0, 2, 3]:
        for n in [0, 1, -1, 2, -2, 3, -3]:
            if m == 0 and n == 0:
                continue
            if len(free_names) >= args.max_dofs:
                break
            for basis, modes, prefix in [
                ("R", R_modes, "R"),
                ("Z", Z_modes, "Z"),
            ]:
                if len(free_names) >= args.max_dofs:
                    break
                for idx_mode, mode in enumerate(modes):
                    if int(mode[1]) == m and int(mode[2]) == n:
                        if prefix == "R":
                            free_idx_R.append(idx_mode)
                        else:
                            free_idx_Z.append(idx_mode)
                        free_names.append(f"{prefix}({m},{n})")
                        break

    nR = len(free_idx_R)
    free_idx_R = np.array(free_idx_R, dtype=int)
    free_idx_Z = np.array(free_idx_Z, dtype=int)
    print(f"\n  Free DoFs ({len(free_names)}):")
    for name in free_names:
        print(f"    {name}")

    R0 = surf.R_lmn.copy()
    Z0 = surf.Z_lmn.copy()
    x0 = np.concatenate([np.asarray(R0)[free_idx_R],
                         np.asarray(Z0)[free_idx_Z]])

    # ── Physics grids ──
    s_vals = np.linspace(0.2, 0.8, args.num_surfaces)
    alphas = np.linspace(0, 2 * np.pi, args.num_alpha, endpoint=False)
    s_grad = np.linspace(getattr(args, 'grad_s_smin', 0.35),
                         getattr(args, 'grad_s_smax', 0.85),
                         getattr(args, 'grad_s_ns', 4))

    # ── Ballooning setup ──
    b_rhos = np.array(getattr(args, 'ballooning_rhos', [0.5, 0.65, 0.8, 0.9]))
    b_alphas = np.linspace(0, np.pi, 6, endpoint=False)
    b_zeta0 = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)

    # ── Run directory ──
    run_dir = os.path.abspath(getattr(args, "run_dir", "") or os.getcwd())
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if getattr(args, "checkpoint_every", 0) > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── History CSV ──
    history_path = os.path.join(run_dir, "history.csv")
    history_fields = [
        "eval", "f_QI", "f_maxJ", "f_Bmin", "mirror_ratio",
        "iota_axis", "iota_edge", "beta", "f_grad_s", "f_highB",
        "well_depth", "dmerc_convention", "dmerc_edge_toroidal_flux_wb",
        "dmerc_vmec_raw_min", "dmerc_flux_normalized_min",
        "dmerc_negative_count",
        "balloon_n_pos", "balloon_lam_max", "aspect", "fb_rms", "elapsed_s",
    ]
    with open(history_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=history_fields).writeheader()

    def _append_history(row):
        with open(history_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=history_fields)
            writer.writerow({k: row.get(k, "") for k in history_fields})

    # ── Temp wout ──
    wout_tmp = os.path.join(tempfile.gettempdir(), "wout_desc_squid.nc")

    # ── State ──
    n_eval = [0]
    iota_targets_set = [False]
    _fb_rms0 = [None]

    # ── Aspect ratio target ──
    if args.aspect_target is None:
        try:
            A0_approx = float(eq.compute("R0")["R0"]) / float(eq.compute("a")["a"])
        except Exception:
            A0_approx = 10.0
        args.aspect_target = round(A0_approx, 1)

    def _append_history_row(**kw):
        _append_history(dict(
            eval=n_eval[0],
            f_QI=kw.get("f_QI"), f_maxJ=kw.get("f_maxJ"),
            f_Bmin=kw.get("f_Bmin"), mirror_ratio=kw.get("delta"),
            iota_axis=kw.get("iota_ax"), iota_edge=kw.get("iota_ed"),
            beta=kw.get("beta"), f_grad_s=kw.get("f_gs", 0.0),
            f_highB=kw.get("f_highB", 0.0),
            well_depth=kw.get("well"),
            dmerc_convention=MERCIER_CONVENTION_VERSION,
            dmerc_edge_toroidal_flux_wb=kw.get("dmerc_edge_toroidal_flux_wb"),
            dmerc_vmec_raw_min=kw.get("dmerc_vmec_raw_min"),
            dmerc_flux_normalized_min=kw.get("dmerc_flux_normalized_min"),
            dmerc_negative_count=kw.get("dmerc_neg"),
            balloon_n_pos=kw.get("balloon_n"),
            balloon_lam_max=kw.get("balloon_lam"),
            aspect=kw.get("aspect"), fb_rms=kw.get("fb_rms"),
            elapsed_s=kw.get("dt"),
        ))

    # ═══════════════════════════════════════════════════════════════
    # Objective function
    # ═══════════════════════════════════════════════════════════════

    def objective(x):
        n_eval[0] += 1
        t0 = time.time()

        # ── 1. Set boundary DoFs ──
        new_R = np.asarray(R0).copy()
        new_Z = np.asarray(Z0).copy()
        new_R[free_idx_R] = x[:nR]
        new_Z[free_idx_Z] = x[nR:]
        surf.R_lmn = new_R
        surf.Z_lmn = new_Z

        # ── 2. DESC force-balance solve ──
        try:
            from desc.objectives import get_fixed_boundary_constraints, ForceBalance
            constraints = get_fixed_boundary_constraints(eq=eq, profiles=True, normalize=True)
            eq.solve(objective="force", constraints=constraints,
                     optimizer="lsq-exact", maxiter=50, verbose=0)
            # Quick FB check: reject if force balance exploded
            _fb_check = ForceBalance(eq=eq)
            _fb_check.build(use_jit=False, verbose=0)
            _fb_post = np.asarray(_fb_check.compute(eq.params_dict), dtype=float)
            _fb_rms_post = float(np.sqrt(np.mean(_fb_post**2)))
            if _fb_rms0[0] is None:
                _fb_rms0[0] = max(_fb_rms_post, 1.0)
            elif _fb_rms_post > 500.0 * _fb_rms0[0] and _fb_rms_post > 1e6:
                return INVALID_TOTAL + _fb_rms_post
        except Exception as exc:
            print(f"    [DESC #{n_eval[0]}] solve FAILED: {exc}")
            return INVALID_TOTAL

        # ── 3. Save DESC eq as VMEC wout, run VMEC for SQuID core ──
        try:
            VMECIO.save(eq, wout_tmp, surfs=args.ns_vmec, verbose=0)
            vmec_ro = Vmec(wout_tmp)
            vmec_ro.run()
        except Exception as exc:
            print(f"    [DESC #{n_eval[0]}] VMEC bridge FAILED: {exc}")
            return INVALID_TOTAL

        # ── 4. SQuID core evaluation (QI, maxJ, Bmin) ──
        try:
            info = _evaluate_squid(
                vmec_ro, s_vals, alphas,
                args.num_pitch, T_J=-0.06, mboz=args.mboz, nboz=args.nboz,
            )
            f_maxJ = info["f_maxJ"]
            f_QI = info["f_QI"]
            delta = info["mirror_ratio"]
            iota_ax = info["iota_axis"]
            iota_ed = info["iota_edge"]
        except Exception as exc:
            print(f"    [DESC #{n_eval[0]}] SQuID core FAILED: {exc}")
            return INVALID_TOTAL

        # ── 5. Bmin penalty ──
        try:
            bmin_res = bmin_slope_residuals(s_vals, info["surface_Bmin"],
                                            getattr(args, 'bmin_slope_target', 0.01))
            f_Bmin = float(np.sum(bmin_res ** 2))
        except Exception:
            f_Bmin = INVALID

        # ── 6. R2 QI (optional) ──
        f_QI_R2 = 0.0
        if getattr(args, 'w_qi_r2', 0.0) > 0:
            try:
                _, qi_r2_res = compute_qi_residual_r2(
                    vmec_ro, s_vals,
                    nphi=getattr(args, 'qi_r2_nphi', 101),
                    nalpha=getattr(args, 'qi_r2_nalpha', 8),
                    nBj=getattr(args, 'qi_r2_nbj', 101),
                    mpol=getattr(args, 'qi_r2_mpol', 8),
                    ntor=getattr(args, 'qi_r2_ntor', 8),
                    arr_out=getattr(args, 'qi_r2_arr_out', False),
                )
                f_QI_R2 = float(np.sum(qi_r2_res ** 2))
            except Exception:
                f_QI_R2 = INVALID

        # ── 6b. High-B Boozer contour topology (optional) ──
        f_highB = 0.0
        w_highB = getattr(args, 'w_highB_topology', 0.0)
        if w_highB > 0:
            try:
                highB_s = np.linspace(
                    getattr(args, 'highB_s_min', 0.25),
                    getattr(args, 'highB_s_max', 0.90),
                    getattr(args, 'highB_ns', 4),
                )
                highB_thresholds = np.linspace(
                    getattr(args, 'highB_threshold_min', 0.75),
                    getattr(args, 'highB_threshold_max', 0.98),
                    getattr(args, 'highB_n_thresholds', 6),
                )
                highB_res = compute_highB_contour_residual(
                    vmec_ro, highB_s, thresholds=highB_thresholds,
                    mpol=getattr(args, 'highB_mpol', 20),
                    ntor=getattr(args, 'highB_ntor', 20),
                    ntheta=getattr(args, 'highB_ntheta', 96),
                    nphi=getattr(args, 'highB_nphi', 96),
                    target_phi_coverage=getattr(args, 'highB_target_phi_coverage', 0.85),
                    max_phi_gap=getattr(args, 'highB_max_phi_gap', 0.20),
                )
                f_highB = float(np.sum(highB_res ** 2))
            except Exception as exc:
                f_highB = INVALID
                print(f"    [DESC #{n_eval[0]}] High-B topology FAILED: {exc}")

        # ── 7. Aspect ratio ──
        try:
            aspect = float(vmec_ro.aspect())
        except Exception:
            aspect = args.aspect_target

        # ── 8. Beta ──
        try:
            beta = float(vmec_ro.wout.betatotal)
        except Exception:
            beta = np.inf

        # ── 9. Iota targets (auto-detect once) ──
        if not iota_targets_set[0]:
            if args.iota_ax is None:
                args.iota_ax = round(iota_ax, 3)
            if args.iota_edge is None:
                args.iota_edge = round(iota_ed, 3)
            iota_targets_set[0] = True

        # ── 10. Mirror, beta, iota penalties ──
        f_mirror = _compute_mirror_penalty(delta, getattr(args, 'mirror_target', 0.20))
        f_beta = _compute_beta_penalty(beta, getattr(args, 'beta_target', 0.02))
        f_iota = _compute_iota_penalty(
            iota_ax, iota_ed, args.iota_ax, args.iota_edge,
            getattr(args, 'iota_tolerance', 0.01),
            getattr(args, 'iota_edge_mode', 'target'),
        )

        # ── 11. ITG (optional) ──
        f_gs = 0.0
        if getattr(args, 'w_grad_s', 0.0) > 0:
            try:
                f_gs = ITGResidual(
                    vmec_ro, s_grad,
                    method=getattr(args, 'itg_method', 'drift_curvature')).total()
            except Exception:
                f_gs = INVALID

        # ── 12. Mercier margin (from VMEC wout — DESC-saved) ──
        dmerc_vmec_raw_min = np.nan
        dmerc_flux_normalized_min = np.nan
        dmerc_edge_toroidal_flux_wb = np.nan
        dmerc_neg = 0
        f_mercier_margin = 0.0
        w_mm = getattr(args, 'w_mercier_margin', 0.0)
        if w_mm > 0 or getattr(args, 'hard_gate_mhd', False):
            try:
                dmerc_raw_full = np.array(vmec_ro.wout.DMerc, dtype=float)
                dmerc_full = flux_normalize_mercier(dmerc_raw_full, vmec_ro.wout)
                dmerc_edge_toroidal_flux_wb = edge_toroidal_flux_wb(vmec_ro.wout)
                s_dm, dmerc_raw = vmec_half_grid_profile(dmerc_raw_full)
                _, dmerc = vmec_half_grid_profile(
                    dmerc_full, ns=dmerc_raw_full.size
                )
                mask = ((s_dm >= getattr(args, 'mercier_s_min', 0.1))
                        & (s_dm <= getattr(args, 'mercier_s_max', 0.95))
                        & np.isfinite(dmerc_raw)
                        & np.isfinite(dmerc))
                vals = dmerc[mask]
                if vals.size > 0:
                    target = float(getattr(
                        args, 'mercier_flux_normalized_margin_target', 0.0
                    ))
                    mm_res = np.maximum(target - vals, 0.0)
                    f_mercier_margin = float(np.mean(mm_res ** 2))
                    dmerc_flux_normalized_min = float(np.nanmin(vals))
                    dmerc_vmec_raw_min = float(np.nanmin(dmerc_raw[mask]))
                    dmerc_neg = int(np.sum(vals < 0))
            except Exception:
                f_mercier_margin = INVALID

        # ── 13. Well depth ──
        well_depth = np.nan
        f_well = 0.0
        if getattr(args, 'w_well', 0.0) > 0:
            try:
                vp = np.array(vmec_ro.wout.vp)
                vp_ax, vp_ed = float(vp[1]), float(vp[-1])
                well_depth = (vp_ax - vp_ed) / max(abs(vp_ax), 1e-30)
                f_well = hinge_loss(getattr(args, 'target_well', 0.01) - well_depth, 0.0)
            except Exception:
                f_well = INVALID

        # ── 14. Ballooning (evaluated on DESC eq — already force-balanced) ──
        balloon_n = -1
        balloon_lam = np.nan
        f_ballooning = 0.0
        w_bal = getattr(args, 'w_ballooning', 0.0)
        if w_bal > 0 or getattr(args, 'hard_gate_mhd', False):
            try:
                from desc.objectives import BallooningStability
                lam_target = float(getattr(args, "ballooning_lambda_target", 0.0))
                lam_shift = 1.0
                lam_residuals = []
                balloon_n = 0
                balloon_lam = -999.0
                for a in b_alphas:
                    obj_b = BallooningStability(
                        eq, rho=b_rhos, alpha=np.array([a]),
                        nturns=2, nzetaperturn=80, zeta0=b_zeta0,
                        Neigvals=1, lambda0=-lam_shift, w0=0.0, w1=1.0,
                    )
                    obj_b.build(use_jit=False, verbose=0)
                    v = np.asarray(obj_b.compute(eq.params_dict), dtype=float)
                    lam_raw = v.ravel() - lam_shift
                    balloon_lam = max(balloon_lam, float(np.max(lam_raw)))
                    balloon_n += int(np.sum(lam_raw > 0))
                    lam_residuals.extend(
                        np.maximum(lam_raw - lam_target, 0.0).tolist())
                f_ballooning = float(np.mean(np.array(lam_residuals, dtype=float) ** 2))
            except Exception as exc:
                f_ballooning = 1e4
                balloon_n = -1
                balloon_lam = np.nan
                print(f"    [DESC #{n_eval[0]}] Ballooning FAILED: {exc}")

        # ── 15. Force-balance RMS (diagnostic, cheap) ──
        fb_rms = np.nan
        w_fb = getattr(args, 'w_force_balance', 0.0)
        f_fb = 0.0
        hard_fb_limit = float(getattr(args, 'hard_fb_rms_max', np.inf))
        if (
            w_fb > 0
            or (
                getattr(args, 'hard_gate_mhd', False)
                and np.isfinite(hard_fb_limit)
            )
        ):
            try:
                from desc.objectives import ForceBalance as _ForceBalance
                _fb_obj = _ForceBalance(eq=eq)
                _fb_obj.build(use_jit=False, verbose=0)
                _fb_vals = np.asarray(_fb_obj.compute(eq.params_dict), dtype=float)
                fb_rms = float(np.sqrt(np.mean(_fb_vals ** 2)))
                if _fb_rms0[0] is None:
                    _fb_rms0[0] = max(fb_rms, 1.0)
                f_fb = max(fb_rms / _fb_rms0[0] - 1.0, 0.0)
            except Exception:
                f_fb = INVALID

        # ── 16. Regularisation ──
        f_reg = float(np.sum((x - x0) ** 2))

        # ── 17. Soft DoF bounds: penalize changes > frac of |x0| ──
        dof_frac = float(getattr(args, 'dof_bound_frac', 0.0))
        f_bound = 0.0
        if dof_frac > 0:
            max_dx = dof_frac * np.maximum(np.abs(x0), 1e-4)
            excess = np.maximum(np.abs(x - x0) - max_dx, 0.0)
            f_bound = float(np.sum(excess ** 2))

        # ── 18. Optional hard MHD gates ──
        hard_gate_penalty = 0.0
        hard_gate_reasons = []
        if getattr(args, 'hard_gate_mhd', False):
            dmerc_floor = float(getattr(
                args, 'hard_dmerc_flux_normalized_min', 0.0
            ))
            dmerc_neg_max = int(getattr(args, 'hard_dmerc_neg_max', 0))
            if ((not np.isfinite(dmerc_flux_normalized_min))
                    or dmerc_flux_normalized_min < dmerc_floor):
                current = dmerc_flux_normalized_min if np.isfinite(dmerc_flux_normalized_min) else -1.0
                hard_gate_penalty += INVALID_TOTAL * (1.0 + max(dmerc_floor - current, 0.0))
                hard_gate_reasons.append("DMerc")
            if dmerc_neg > dmerc_neg_max:
                hard_gate_penalty += INVALID_TOTAL * (1.0 + dmerc_neg - dmerc_neg_max)
                hard_gate_reasons.append("DMerc_neg")

            bal_n_max = int(getattr(args, 'hard_ballooning_n_max', 0))
            bal_lam_max = float(getattr(args, 'hard_ballooning_lambda_max', 0.0))
            if balloon_n < 0:
                hard_gate_penalty += INVALID_TOTAL
                hard_gate_reasons.append("balloon_failed")
            elif balloon_n > bal_n_max:
                hard_gate_penalty += INVALID_TOTAL * (1.0 + balloon_n - bal_n_max)
                hard_gate_reasons.append("balloon_n")
            if not np.isfinite(balloon_lam):
                hard_gate_penalty += INVALID_TOTAL
                hard_gate_reasons.append("balloon_lam_failed")
            elif balloon_lam > bal_lam_max:
                hard_gate_penalty += INVALID_TOTAL * (1.0 + balloon_lam - bal_lam_max)
                hard_gate_reasons.append("balloon_lam")

            beta_max = float(getattr(args, 'hard_beta_max', np.inf))
            if np.isfinite(beta_max) and (
                (not np.isfinite(beta)) or beta > beta_max
            ):
                excess = beta - beta_max if np.isfinite(beta) else 1.0
                hard_gate_penalty += INVALID_TOTAL * (1.0 + max(excess, 0.0))
                hard_gate_reasons.append("beta")

            fb_max = hard_fb_limit
            if np.isfinite(fb_max) and (
                (not np.isfinite(fb_rms)) or fb_rms > fb_max
            ):
                ratio = fb_rms / max(fb_max, 1.0) if np.isfinite(fb_rms) else 1.0
                hard_gate_penalty += INVALID_TOTAL * (1.0 + ratio)
                hard_gate_reasons.append("FB")

        # ── Assemble total ──
        total = (
            getattr(args, 'w_qi', 1.0) * f_QI
            + getattr(args, 'w_qi_r2', 0.0) * f_QI_R2
            + w_highB * f_highB
            + getattr(args, 'w_maxj', 1.0) * f_maxJ
            + getattr(args, 'w_bmin', 1.0) * f_Bmin
            + getattr(args, 'w_ar', 100.0) * hinge_loss(aspect, args.aspect_target)
            + getattr(args, 'w_mirror', 0.0) * f_mirror
            + getattr(args, 'w_beta', 0.0) * f_beta
            + getattr(args, 'w_iota', 0.0) * f_iota
            + getattr(args, 'w_grad_s', 0.0) * f_gs
            + getattr(args, 'w_well', 0.0) * f_well
            + w_mm * f_mercier_margin
            + w_bal * f_ballooning
            + w_fb * f_fb
            + getattr(args, 'w_reg', 10.0) * f_reg
            + 1.0e6 * f_bound
            + hard_gate_penalty
        )

        dt = time.time() - t0

        # ── Log ──
        parts = [
            f"f_QI={f_QI:.3e}", f"f_maxJ={f_maxJ:.3e}", f"f_Bmin={f_Bmin:.3e}",
            f"delta={delta:.4f}", f"iota=[{iota_ax:.3f},{iota_ed:.3f}]",
            f"beta={beta:.4f}",
        ]
        if w_bal > 0:
            parts.append(f"bal_n={balloon_n} lam={balloon_lam:.2e}")
        if w_highB > 0:
            parts.append(f"f_highB={f_highB:.3e}")
        if w_mm > 0:
            parts.append(
                f"PhiEdge^2*DMerc={dmerc_flux_normalized_min:.2e}"
                f"({dmerc_neg}n)"
            )
        if not np.isnan(fb_rms):
            parts.append(f"FB={fb_rms:.1e}")
        if hard_gate_reasons:
            parts.append("HARD_GATE=" + ",".join(hard_gate_reasons))
        parts.append(f"({dt:.1f}s)")
        print(f"    [DESC #{n_eval[0]}]  {'  '.join(parts)}")

        _append_history_row(
            f_QI=f_QI, f_maxJ=f_maxJ, f_Bmin=f_Bmin,
            delta=delta, iota_ax=iota_ax, iota_ed=iota_ed,
            beta=beta, f_gs=f_gs, well=well_depth,
            f_highB=f_highB,
            dmerc_edge_toroidal_flux_wb=dmerc_edge_toroidal_flux_wb,
            dmerc_vmec_raw_min=dmerc_vmec_raw_min,
            dmerc_flux_normalized_min=dmerc_flux_normalized_min,
            dmerc_neg=dmerc_neg,
            balloon_n=balloon_n, balloon_lam=balloon_lam,
            aspect=aspect, fb_rms=fb_rms, dt=dt,
        )

        # ── Checkpoint ──
        cp_every = int(getattr(args, "checkpoint_every", 0) or 0)
        if cp_every > 0 and n_eval[0] % cp_every == 0:
            try:
                cp_wout = os.path.join(checkpoint_dir,
                                       f"wout_desc_eval_{n_eval[0]:06d}.nc")
                VMECIO.save(eq, cp_wout, surfs=args.ns_vmec, verbose=0)
            except Exception:
                pass

        return total

    # ═══════════════════════════════════════════════════════════════
    # Run optimisation
    # ═══════════════════════════════════════════════════════════════

    print(f"\n  Evaluating initial state (with force-balance solve) ...")
    obj0 = objective(x0)
    x_save = x0.copy()

    if args.maxiter > 0:
        max_total_evals = int(getattr(args, "max_total_evals", 0) or 0)
        cap_msg = f", max_evals={max_total_evals}" if max_total_evals > 0 else ""
        print(f"\n  Starting Nelder-Mead (maxiter={args.maxiter}{cap_msg}) ...")
        t_start = time.time()

        options = dict(maxiter=args.maxiter, xatol=1e-5, fatol=1e-4, adaptive=True)
        if max_total_evals > 0:
            options["maxfev"] = max_total_evals

        result = minimize(
            objective, x0, method="Nelder-Mead", options=options,
        )
        x_save = np.asarray(result.x, dtype=float).copy()
        t_total = time.time() - t_start
        print(f"\n  Finished in {t_total / 60:.1f} min  ({n_eval[0]} evals)")
        print(f"  Optimizer message: {result.message}")
    else:
        print("\n  maxiter <= 0: initial evaluation only; optimisation skipped.")

    print(f"\n  Initial objective = {obj0:.4e}")

    # ── Save final equilibrium ──
    out_wout = os.path.join(run_dir, "wout_desc_optimized.nc")
    try:
        new_R = np.asarray(R0).copy()
        new_Z = np.asarray(Z0).copy()
        new_R[free_idx_R] = x_save[:nR]
        new_Z[free_idx_Z] = x_save[nR:]
        surf.R_lmn = new_R
        surf.Z_lmn = new_Z
        from desc.objectives import get_fixed_boundary_constraints
        constraints = get_fixed_boundary_constraints(eq=eq, profiles=True, normalize=True)
        eq.solve(objective="force", constraints=constraints,
                 optimizer="lsq-exact", maxiter=50, verbose=0)
        VMECIO.save(eq, out_wout, surfs=args.ns_vmec, verbose=0)
        print(f"  Saved: {out_wout}")
    except Exception as e:
        print(f"  Could not save DESC wout: {e}")

    # Cleanup
    if os.path.exists(wout_tmp):
        os.remove(wout_tmp)

    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}\n")
