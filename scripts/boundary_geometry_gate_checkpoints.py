#!/usr/bin/env python3
"""Summarize boundary-geometry coil-hardness proxies for checkpoint wouts."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from squid.diagnostics.boundary_geometry import boundary_geometry_metrics


def _eval_index(path: Path) -> int:
    match = re.search(r"eval_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _stat(metrics: dict, name: str, key: str) -> float:
    return float(metrics[name][key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pdrot/principal-curvature metrics for checkpoint wouts."
    )
    parser.add_argument("--run_dir", required=True, help="Optimization run directory")
    parser.add_argument("--output_csv", help="CSV output path")
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--ntheta", type=int, default=128)
    parser.add_argument("--pdrot-aw-limit", type=float, default=1.49)
    parser.add_argument("--pdrot-max-limit", type=float, default=16.5)
    parser.add_argument("--k2-abs-limit", type=float, default=102.0)
    parser.add_argument("--h-abs-limit", type=float, default=51.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    wouts = sorted((run_dir / "checkpoints").glob("wout_squid_eval_*.nc"), key=_eval_index)
    if not wouts:
        raise FileNotFoundError(f"No checkpoint wouts found under {run_dir / 'checkpoints'}")

    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "boundary_geometry_gate_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for wout in wouts:
        metrics = boundary_geometry_metrics(wout, nphi=args.nphi, ntheta=args.ntheta)
        pdrot_aw = _stat(metrics, "pdrot_1_per_m", "mean_area_weighted")
        pdrot_max = _stat(metrics, "pdrot_1_per_m", "max")
        k2_min = _stat(metrics, "k2_1_per_m", "min")
        h_min = _stat(metrics, "H_1_per_m", "min")
        verdict = "PASS"
        reasons = []
        if pdrot_aw > args.pdrot_aw_limit:
            reasons.append("PDROT_AW")
        if pdrot_max > args.pdrot_max_limit:
            reasons.append("PDROT_MAX")
        if -k2_min > args.k2_abs_limit:
            reasons.append("K2_ABS")
        if -h_min > args.h_abs_limit:
            reasons.append("H_ABS")
        if reasons:
            verdict = "FAIL_" + "_".join(reasons)
        rows.append(
            {
                "eval": _eval_index(wout),
                "wout": wout.name,
                "pdrot_aw": pdrot_aw,
                "pdrot_max": pdrot_max,
                "k2_min": k2_min,
                "H_min": h_min,
                "surface_area_m2": float(metrics["surface_area_m2_simsopt"]),
                "verdict": verdict,
            }
        )

    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"eval{row['eval']:03d}: {row['verdict']} "
            f"pdrot_aw={row['pdrot_aw']:.4f} pdrot_max={row['pdrot_max']:.2f} "
            f"k2_min={row['k2_min']:.2f} H_min={row['H_min']:.2f}"
        )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
