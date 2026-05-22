"""
DESC backend for SQuID optimisation.

Uses DESC (pure Python) as the equilibrium solver with Nelder-Mead
optimisation. Falls back to this when VMEC2000 is not installed.
"""

import os
import time
import glob
import tempfile
import numpy as np

from simsopt.mhd import Vmec

from ..objectives.maxj_residual import _evaluate_squid
from ..objectives.qi_residual import compute_qi_residual_r2
from ..objectives.itg_residual import ITGResidual
from ..objectives.penalties import bmin_slope_residuals, hinge_loss


INVALID_OBJECTIVE = 1.0e6
INVALID_TOTAL = 1.0e10


def _compute_mirror_penalty(mirror_ratio, mirror_target):
    """Penalise mirror ratio above the configured upper bound."""
    return hinge_loss(mirror_ratio, mirror_target)


def _compute_beta_penalty(beta, beta_target):
    """Penalise total beta above the configured upper bound."""
    if not np.isfinite(beta):
        return INVALID_OBJECTIVE
    return hinge_loss(beta, beta_target)


def _compute_iota_penalty(iota_axis, iota_edge, iota_ax_target,
                          iota_edge_target, tolerance=0.01):
    axis_penalty = hinge_loss(abs(iota_axis - iota_ax_target), tolerance)
    edge_penalty = hinge_loss(abs(iota_edge - iota_edge_target), tolerance)
    return axis_penalty + edge_penalty


def run_desc(args):
    """Optimisation using DESC as the equilibrium solver."""
    from desc.vmec import VMECIO
    from scipy.optimize import minimize
    import desc as _desc_mod

    print(f"\n  Backend: DESC (v{_desc_mod.__version__})")

    eq = VMECIO.load(args.nc_file)
    print(f"  Original resolution: L={eq.L}  M={eq.M}  N={eq.N}")

    L_new = min(args.desc_L, eq.L)
    M_new = min(args.desc_M, eq.M)
    N_new = min(args.desc_N, eq.N)
    if L_new < eq.L or M_new < eq.M or N_new < eq.N:
        eq.change_resolution(L=L_new, M=M_new, N=N_new)
        eq.surface = eq.get_surface_at(rho=1.0)
        print(f"  Reduced to:          L={L_new}  M={M_new}  N={N_new}")

    surf = eq.surface
    R_modes = surf.R_basis.modes
    Z_modes = surf.Z_basis.modes

    free_idx_R, free_idx_Z, free_names = [], [], []
    for m in [1, 2, 3, 0]:
        for n in [0, 1, -1, 2, -2]:
            if m == 0 and n == 0:
                continue
            if len(free_names) >= args.max_dofs:
                break
            for basis, modes, lmn_arr, prefix in [
                ("R", R_modes, surf.R_lmn, "R"),
                ("Z", Z_modes, surf.Z_lmn, "Z"),
            ]:
                if len(free_names) >= args.max_dofs:
                    break
                for idx_mode, mode in enumerate(modes):
                    if int(mode[1]) == m and int(mode[2]) == n:
                        if prefix == "R":
                            free_idx_R.append(idx_mode)
                        else:
                            free_idx_Z.append(idx_mode)
                        free_names.append(f"{prefix}(l={int(mode[0])},m={m},n={n})")
                        break

    print(f"\n  Free DoFs ({len(free_names)}):")
    for name in free_names:
        print(f"    {name}")

    R0 = surf.R_lmn.copy()
    Z0 = surf.Z_lmn.copy()

    nR = len(free_idx_R)
    x0 = np.concatenate([R0[np.array(free_idx_R)], Z0[np.array(free_idx_Z)]])

    s_vals = np.linspace(0.2, 0.8, args.num_surfaces)
    alphas = np.linspace(0, 2 * np.pi, args.num_alpha, endpoint=False)
    s_grad = np.linspace(args.grad_s_smin, args.grad_s_smax, args.grad_s_ns)

    wout_tmp = os.path.join(tempfile.gettempdir(), "wout_squid_tmp.nc")
    n_eval = [0]
    first_eval = [True]
    history = []
    iota_targets_set = [False]

    if args.aspect_target is None:
        try:
            A0_approx = float(eq.compute("R0")["R0"]) / float(eq.compute("a")["a"])
        except Exception:
            A0_approx = 10.0
        args.aspect_target = round(A0_approx, 1)

    def objective(x):
        n_eval[0] += 1
        t0 = time.time()

        new_R = R0.copy()
        new_Z = Z0.copy()
        new_R = new_R.at[np.array(free_idx_R)].set(x[:nR])
        new_Z = new_Z.at[np.array(free_idx_Z)].set(x[nR:])
        surf.R_lmn = new_R
        surf.Z_lmn = new_Z

        if not first_eval[0]:
            try:
                eq.solve(verbose=0, ftol=1e-6, maxiter=50)
            except Exception as exc:
                print(f"    [#{n_eval[0]}] solve failed: {exc}")
                return INVALID_TOTAL
        first_eval[0] = False

        try:
            VMECIO.save(eq, wout_tmp, surfs=args.ns_vmec, verbose=0)
            vmec_ro = Vmec(wout_tmp)
            vmec_ro.run()
        except Exception as exc:
            print(f"    [#{n_eval[0]}] VMEC load failed: {exc}")
            return INVALID_TOTAL

        try:
            info = _evaluate_squid(
                vmec_ro, s_vals, alphas,
                args.num_pitch, T_J=-0.06, mboz=8, nboz=8,
            )
        except Exception as exc:
            print(f"    [#{n_eval[0]}] SQuID failed: {exc}")
            return INVALID_TOTAL

        f_maxJ = info["f_maxJ"]
        f_QI = info["f_QI"]
        try:
            bmin_residuals = bmin_slope_residuals(
                s_vals, info["surface_Bmin"], args.bmin_slope_target)
            f_Bmin = float(np.sum(bmin_residuals ** 2))
        except Exception as exc:
            print(f"    [#{n_eval[0]}] B_min target failed: {exc}")
            f_Bmin = INVALID_OBJECTIVE
        f_QI_R2 = 0.0
        if args.w_qi_r2 > 0:
            try:
                _, qi_r2_residuals = compute_qi_residual_r2(
                    vmec_ro, s_vals,
                    nphi=args.qi_r2_nphi,
                    nalpha=args.qi_r2_nalpha,
                    nBj=args.qi_r2_nbj,
                    mpol=args.qi_r2_mpol,
                    ntor=args.qi_r2_ntor,
                    arr_out=args.qi_r2_arr_out,
                )
                f_QI_R2 = float(np.sum(qi_r2_residuals ** 2))
            except Exception as exc:
                print(f"    [#{n_eval[0]}] R2 QI target failed: {exc}")
                f_QI_R2 = INVALID_OBJECTIVE
        delta = info["mirror_ratio"]
        iota_ax = info["iota_axis"]
        iota_ed = info["iota_edge"]

        try:
            A = vmec_ro.aspect()
        except Exception:
            A = args.aspect_target

        try:
            beta = float(vmec_ro.wout.betatotal)
        except Exception:
            beta = np.inf

        if not iota_targets_set[0]:
            if args.iota_ax is None:
                args.iota_ax = round(iota_ax, 3)
            if args.iota_edge is None:
                args.iota_edge = round(iota_ed, 3)
            iota_targets_set[0] = True

        f_mirror = _compute_mirror_penalty(delta, args.mirror_target)
        f_beta = _compute_beta_penalty(beta, args.beta_target)
        f_iota = _compute_iota_penalty(iota_ax, iota_ed,
                                       args.iota_ax, args.iota_edge,
                                       args.iota_tolerance)

        f_gs = 0.0
        if getattr(args, 'w_grad_s', 0.0) > 0:
            try:
                f_gs = ITGResidual(
                    vmec_ro, s_grad, method=args.itg_method).total()
            except Exception as exc:
                print(f"    [#{n_eval[0]}] ITG target failed: {exc}")
                f_gs = INVALID_OBJECTIVE

        f_reg = float(np.sum((x - x0) ** 2))

        total = (args.w_maxj * f_maxJ
                 + args.w_qi * f_QI
                 + args.w_qi_r2 * f_QI_R2
                 + args.w_bmin * f_Bmin
                 + args.w_ar * hinge_loss(A, args.aspect_target)
                 + args.w_mirror * f_mirror
                 + args.w_beta * f_beta
                 + args.w_iota * f_iota
                 + args.w_grad_s * f_gs
                 + args.w_reg * f_reg)

        dt = time.time() - t0
        parts = [f"f_maxJ={f_maxJ:.3e}", f"f_QI={f_QI:.3e}",
                 f"f_Bmin={f_Bmin:.3e}",
                 f"A={A:.2f}", f"delta={delta:.4f}",
                 f"iota=[{iota_ax:.3f},{iota_ed:.3f}]",
                 f"beta={beta:.4f}"]
        if getattr(args, 'w_grad_s', 0.0) > 0:
            parts.append(f"f_nabla_s={f_gs:.2e}")
        if args.w_qi_r2 > 0:
            parts.append(f"f_QI_R2={f_QI_R2:.3e}")
        parts.extend([f"total={total:.3e}", f"({dt:.1f}s)"])
        print(f"    [#{n_eval[0]}]  {'  '.join(parts)}")
        history.append(dict(
            f_maxJ=f_maxJ, f_QI=f_QI, f_QI_R2=f_QI_R2,
            f_Bmin=f_Bmin, A=A, total=total,
            mirror=delta, iota_ax=iota_ax, iota_ed=iota_ed,
            beta=beta, f_mirror=f_mirror, f_beta=f_beta, f_iota=f_iota,
            f_grad_s=f_gs, f_reg=f_reg,
            B_min=info["B_min"], B_max=info["B_max"],
        ))
        return total

    print(f"\n  Evaluating initial state ...")
    obj0 = objective(x0)

    if args.maxiter > 0:
        print(f"\n  Starting Nelder-Mead (maxiter={args.maxiter}) ...")
        t_start = time.time()
        result = minimize(
            objective, x0, method="Nelder-Mead",
            options=dict(maxiter=args.maxiter, xatol=1e-5, fatol=1e-4, adaptive=True),
        )
        t_total = time.time() - t_start

        print(f"\n  Finished in {t_total / 60:.1f} min  ({n_eval[0]} evals)")
        print(f"  Optimizer message: {result.message}")
    else:
        print("\n  maxiter <= 0: initial evaluation only; optimisation skipped.")

    if history:
        h0, hf = history[0], history[-1]
        print(f"\n  Initial total = {h0['total']:.4e}")
        print(f"  Final total   = {hf['total']:.4e}")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(args.nc_file)),
        "wout_squid_optimized.nc",
    )
    try:
        VMECIO.save(eq, out_path, surfs=args.ns_vmec, verbose=0)
        print(f"\n  Saved: {out_path}")
    except Exception as e:
        print(f"\n  Could not save: {e}")

    if os.path.exists(wout_tmp):
        os.remove(wout_tmp)
