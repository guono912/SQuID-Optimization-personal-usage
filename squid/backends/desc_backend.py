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

from simsopt.mhd import Vmec, Boozer

from ..objectives.maxj_residual import _evaluate_squid
from ..objectives.penalties import hinge_loss


def _compute_mirror_penalty(mirror_ratio, mirror_target):
    return hinge_loss(mirror_target - mirror_ratio, 0.0)


def _compute_iota_penalty(iota_axis, iota_edge, iota_ax_target, iota_edge_target):
    return (iota_axis - iota_ax_target) ** 2 + (iota_edge - iota_edge_target) ** 2


def _compute_f_grad_s(vmec_ro, s_targets, nu=80, nv=80):
    """Vacuum-proxy ITG target."""
    wout = vmec_ro.wout
    ns = int(wout.ns)
    nfp = int(wout.nfp)
    xm = np.ravel(np.array(wout.xm, dtype=int))
    xn = np.ravel(np.array(wout.xn, dtype=int))
    xm_nyq = np.ravel(np.array(wout.xm_nyq, dtype=int))
    xn_nyq = np.ravel(np.array(wout.xn_nyq, dtype=int))
    rmnc = np.array(wout.rmnc)
    zmns = np.array(wout.zmns)
    bmnc = np.array(wout.bmnc)
    gmnc = np.array(wout.gmnc)
    a_min = float(wout.Aminor_p)

    theta = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / nfp, nv, endpoint=False)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")

    m_geom = xm[:, None, None]
    n_geom = xn[:, None, None]
    ang_geom = m_geom * th[None, :, :] - n_geom * ze[None, :, :]
    cos_geom = np.cos(ang_geom)
    sin_geom = np.sin(ang_geom)

    m_nyq = xm_nyq[:, None, None]
    n_nyq = xn_nyq[:, None, None]
    ang_nyq = m_nyq * th[None, :, :] - n_nyq * ze[None, :, :]
    cos_nyq = np.cos(ang_nyq)

    s_grid = np.linspace(0, 1, ns)

    def _rg(fmnc, k):
        return np.sum(fmnc[:, k][:, None, None] * cos_geom, axis=0)

    def _rn(fmnc, k):
        return np.sum(fmnc[:, k][:, None, None] * cos_nyq, axis=0)

    f_total = 0.0
    for s_t in s_targets:
        js = int(np.argmin(np.abs(s_grid - s_t)))
        js = max(1, min(js, ns - 2))
        ds = s_grid[js + 1] - s_grid[js - 1]
        R = _rg(rmnc, js)
        R_u = np.sum(-m_geom * rmnc[:, js][:, None, None] * sin_geom, axis=0)
        R_v = np.sum(n_geom * rmnc[:, js][:, None, None] * sin_geom, axis=0)
        Z_u = np.sum(m_geom * zmns[:, js][:, None, None] * cos_geom, axis=0)
        Z_v = np.sum(-n_geom * zmns[:, js][:, None, None] * cos_geom, axis=0)
        sqrtg = _rn(gmnc, js)
        sqrtg = np.where(np.abs(sqrtg) < 1e-30, 1e-30, sqrtg)
        cross_sq = R ** 2 * (R_u ** 2 + Z_u ** 2) + (Z_u * R_v - R_u * Z_v) ** 2
        grad_s = np.sqrt(np.maximum(cross_sq / sqrtg ** 2, 1e-30))
        dBds = (_rn(bmnc, js + 1) - _rn(bmnc, js - 1)) / ds
        bad_mask = np.where(dBds < 0, 1.0, 0.0)
        xi = (a_min * bad_mask * grad_s) ** 2
        xi_pos = xi[xi > 0]
        if len(xi_pos) == 0:
            continue
        xi_95 = float(np.percentile(xi_pos, 95))
        integrand = xi * np.maximum(xi_95 - xi, 0.0)
        dth = 2 * np.pi / nu
        dze = 2 * np.pi / (nfp * nv)
        f_total += float(np.sum(integrand)) * dth * dze
    return f_total


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
                return 1e10
        first_eval[0] = False

        try:
            VMECIO.save(eq, wout_tmp, surfs=args.ns_vmec, verbose=0)
            vmec_ro = Vmec(wout_tmp)
            vmec_ro.run()
        except Exception as exc:
            print(f"    [#{n_eval[0]}] VMEC load failed: {exc}")
            return 1e10

        try:
            info = _evaluate_squid(
                vmec_ro, s_vals, alphas,
                args.num_pitch, T_J=-0.06, mboz=8, nboz=8,
            )
        except Exception as exc:
            print(f"    [#{n_eval[0]}] SQuID failed: {exc}")
            return 1e10

        f_maxJ = info["f_maxJ"]
        f_QI = info["f_QI"]
        delta = info["mirror_ratio"]
        iota_ax = info["iota_axis"]
        iota_ed = info["iota_edge"]

        try:
            A = vmec_ro.aspect()
        except Exception:
            A = args.aspect_target

        if not iota_targets_set[0]:
            if args.iota_ax is None:
                args.iota_ax = round(iota_ax, 3)
            if args.iota_edge is None:
                args.iota_edge = round(iota_ed, 3)
            iota_targets_set[0] = True

        f_mirror = _compute_mirror_penalty(delta, args.mirror_target)
        f_iota = _compute_iota_penalty(iota_ax, iota_ed,
                                       args.iota_ax, args.iota_edge)

        try:
            f_gs = _compute_f_grad_s(vmec_ro, s_grad)
        except Exception:
            f_gs = 0.0

        f_reg = float(np.sum((x - x0) ** 2))

        total = (args.w_maxj * f_maxJ
                 + args.w_qi * f_QI
                 + args.w_ar * (A - args.aspect_target) ** 2
                 + args.w_mirror * f_mirror
                 + args.w_iota * f_iota
                 + args.w_grad_s * f_gs
                 + args.w_reg * f_reg)

        dt = time.time() - t0
        print(f"    [#{n_eval[0]}]  f_maxJ={f_maxJ:.3e}  f_QI={f_QI:.3e}  "
              f"A={A:.2f}  delta={delta:.4f}  "
              f"iota=[{iota_ax:.3f},{iota_ed:.3f}]  "
              f"f_nabla_s={f_gs:.2e}  total={total:.3e}  ({dt:.1f}s)")
        history.append(dict(
            f_maxJ=f_maxJ, f_QI=f_QI, A=A, total=total,
            mirror=delta, iota_ax=iota_ax, iota_ed=iota_ed,
            f_mirror=f_mirror, f_iota=f_iota,
            f_grad_s=f_gs, f_reg=f_reg,
            B_min=info["B_min"], B_max=info["B_max"],
        ))
        return total

    print(f"\n  Evaluating initial state ...")
    obj0 = objective(x0)

    print(f"\n  Starting Nelder-Mead (maxiter={args.maxiter}) ...")
    t_start = time.time()
    result = minimize(
        objective, x0, method="Nelder-Mead",
        options=dict(maxiter=args.maxiter, xatol=1e-5, fatol=1e-4, adaptive=True),
    )
    t_total = time.time() - t_start

    print(f"\n  Finished in {t_total / 60:.1f} min  ({n_eval[0]} evals)")
    print(f"  Optimizer message: {result.message}")

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
