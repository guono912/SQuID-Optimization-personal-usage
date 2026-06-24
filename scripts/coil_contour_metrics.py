#!/usr/bin/env python3
"""Cut coils from a DESC current potential solve and report geometry metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from desc.grid import Grid, LinearGrid
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.magnetic_fields import solve_regularized_surface_current
from desc.vmec import VMECIO
from desc.utils import rpz2xyz

from coil_feasibility_gate import make_winding_surface


def curve_metrics_xyz(x):
    x = np.asarray(x, dtype=float)
    if np.linalg.norm(x[0] - x[-1]) > 1e-10:
        x = np.vstack([x, x[0]])
    ds_vec = np.diff(x, axis=0)
    ds = np.linalg.norm(ds_vec, axis=1)
    length = float(np.sum(ds))
    # finite-difference curvature on closed curve
    xp = np.roll(x[:-1], -1, axis=0)
    xm = np.roll(x[:-1], 1, axis=0)
    xc = x[:-1]
    d1 = 0.5 * (xp - xm)
    d2 = xp - 2 * xc + xm
    denom = np.linalg.norm(d1, axis=1) ** 3
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(denom, 1e-300)
    return {
        "length": length,
        "curvature_mean": float(np.mean(kappa)),
        "curvature_rms": float(np.sqrt(np.mean(kappa**2))),
        "curvature_max": float(np.max(kappa)),
        "curvature_p95": float(np.percentile(kappa, 95)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wout", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--offset", type=float, default=0.50)
    p.add_argument("--lambda_regularization", type=float, default=1e-14)
    p.add_argument("--num_coils", type=int, default=4)
    p.add_argument("--M_Phi", type=int, default=4)
    p.add_argument("--N_Phi", type=int, default=4)
    p.add_argument("--desc_L", type=int, default=5)
    p.add_argument("--desc_M", type=int, default=5)
    p.add_argument("--desc_N", type=int, default=5)
    p.add_argument("--source_M", type=int, default=16)
    p.add_argument("--source_N", type=int, default=16)
    p.add_argument("--eval_M", type=int, default=16)
    p.add_argument("--eval_N", type=int, default=16)
    p.add_argument("--contour_npts", type=int, default=192)
    p.add_argument("--sample_npts", type=int, default=256)
    p.add_argument("--winding_surface_method", choices=["normal", "scaled"], default="normal")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eq = VMECIO.load(args.wout, L=args.desc_L, M=args.desc_M, N=args.desc_N)
    surface = make_winding_surface(eq, args.offset, method=args.winding_surface_method)
    field = FourierCurrentPotentialField(
        R_lmn=surface.R_lmn,
        Z_lmn=surface.Z_lmn,
        modes_R=surface.R_basis.modes[:, 1:],
        modes_Z=surface.Z_basis.modes[:, 1:],
        NFP=surface.NFP,
        sym=surface.sym,
        M_Phi=args.M_Phi,
        N_Phi=args.N_Phi,
    )
    source_grid = LinearGrid(
        M=max(3 * args.M_Phi, args.source_M),
        N=max(3 * args.N_Phi, args.source_N),
        NFP=eq.NFP,
        sym=False,
    )
    eval_grid = LinearGrid(
        M=args.eval_M, N=args.eval_N, NFP=eq.NFP, sym=False, rho=np.array([1.0])
    )
    fields, data = solve_regularized_surface_current(
        field,
        eq,
        lambda_regularization=np.array([args.lambda_regularization]),
        current_helicity=(1, 0),
        source_grid=source_grid,
        eval_grid=eval_grid,
        verbose=0,
    )
    solved = fields[0]
    coilset = solved.to_CoilSet(
        num_coils=args.num_coils,
        npts=args.contour_npts,
        stell_sym=False,
        show_plots=False,
    )

    # plasma boundary point cloud for minimum distance
    plasma_grid = LinearGrid(M=48, N=48, NFP=eq.NFP, sym=False, rho=np.array([1.0]))
    plasma_rpz = np.asarray(eq.compute("x", grid=plasma_grid)["x"], dtype=float)
    plasma_xyz = np.asarray(rpz2xyz(plasma_rpz), dtype=float)
    plasma_tree = cKDTree(plasma_xyz)

    coil_points = []
    records = []
    s = np.linspace(0, 2 * np.pi, args.sample_npts, endpoint=False)
    curve_grid = Grid(np.column_stack([np.zeros_like(s), np.zeros_like(s), s]), sort=False)
    for i, coil in enumerate(coilset):
        x = np.asarray(coil.compute("x", grid=curve_grid, basis="xyz")["x"], dtype=float)
        coil_points.append(x)
        rec = {"coil_index": i}
        rec.update(curve_metrics_xyz(x))
        rec["min_coil_plasma_distance"] = float(np.min(plasma_tree.query(x)[0]))
        records.append(rec)

    # coil-coil distances
    for i, x in enumerate(coil_points):
        mins = []
        for j, y in enumerate(coil_points):
            if i == j:
                continue
            mins.append(float(np.min(cKDTree(y).query(x)[0])))
        records[i]["min_coil_coil_distance_unique"] = min(mins) if mins else float("nan")

    aggregate = {
        "wout": str(Path(args.wout).resolve()),
        "offset_fraction": args.offset,
        "lambda_regularization": args.lambda_regularization,
        "num_coils_per_period": args.num_coils,
        "NFP": int(eq.NFP),
        "chi2_B": float(np.asarray(data["chi^2_B"]).reshape(-1)[0]),
        "chi2_K": float(np.asarray(data["chi^2_K"]).reshape(-1)[0]),
        "coils": records,
        "summary": {
            "length_mean": float(np.mean([r["length"] for r in records])),
            "curvature_rms_max": float(max(r["curvature_rms"] for r in records)),
            "curvature_max": float(max(r["curvature_max"] for r in records)),
            "min_coil_plasma_distance": float(
                min(r["min_coil_plasma_distance"] for r in records)
            ),
            "min_coil_coil_distance_unique": float(
                min(r["min_coil_coil_distance_unique"] for r in records)
            ),
        },
    }
    (out / "coil_contour_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for x in coil_points:
        ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=1.2)
    ax.set_box_aspect([1, 1, 0.35])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.savefig(out / "coil_contours.png", dpi=180)
    print(json.dumps(aggregate["summary"], indent=2))
    print(out / "coil_contour_metrics.json")


if __name__ == "__main__":
    main()
