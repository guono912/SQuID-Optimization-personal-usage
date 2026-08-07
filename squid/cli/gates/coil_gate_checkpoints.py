"""
Batch coil feasibility and contour gate on SQuID checkpoint wout files.

This complements gate_checkpoints.py: it first ensures checkpoint VMEC wouts
exist, then runs the current-potential coil feasibility proxy and a contour
geometry check for each checkpoint. The heavy steps run as subprocesses via
the scripts/ entry points (scripts/gate/coil_feasibility_gate.py and
scripts/util/coil_contour_metrics.py), exactly as the legacy driver did.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import squid

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(squid.__file__).resolve().parents[1]
COIL_FEASIBILITY_SCRIPT = REPO_ROOT / "scripts" / "gate" / "coil_feasibility_gate.py"
COIL_CONTOUR_SCRIPT = REPO_ROOT / "scripts" / "util" / "coil_contour_metrics.py"


def _ensure_checkpoint_wouts(cp_dir):
    inputs = sorted(cp_dir.glob("input.squid_eval_*"))
    for inp in inputs:
        suffix = inp.name.replace("input.", "")
        wout = cp_dir / f"wout_{suffix}.nc"
        if wout.exists():
            continue
        print(f"  VMEC checkpoint solve: {inp.name}")
        subprocess.run(["xvmec", inp.name], cwd=str(cp_dir), check=True)


def _latest_row(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    return rows[-1]


def _load_summary(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["summary"]


def gate_checkpoints(
    run_dir,
    output_dir=None,
    watch=False,
    min_interval=60,
    offset=0.35,
    lambda_regularization=1e-8,
    bn_limit=4.5e-3,
    curvature_max_limit=8.0,
    curvature_rms_limit=4.0,
    min_plasma_distance=0.06,
    min_coil_distance=0.70,
):
    run_dir = Path(run_dir)
    cp_dir = run_dir / "checkpoints"
    if output_dir is None:
        output_dir = run_dir / "coil_gate_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "coil_gate_summary.csv"
    seen = set()
    if summary_path.exists():
        with open(summary_path) as f:
            for line in f:
                if line.strip() and not line.startswith("wout,"):
                    seen.add(line.split(",", 1)[0])

    header = (
        "wout,offset,lambda_regularization,Bn_max_unitless,Bn_avg_unitless,"
        "K_rms_MApm,K_max_MApm,phi_high_mode_fraction,length_mean,"
        "curvature_max,curvature_rms_max,min_coil_plasma_distance,"
        "min_coil_coil_distance_unique,verdict\n"
    )
    if not summary_path.exists():
        with open(summary_path, "w") as f:
            f.write(header)

    while True:
        _ensure_checkpoint_wouts(cp_dir)
        wouts = sorted(cp_dir.glob("wout_*.nc"), key=lambda p: p.stat().st_mtime)
        new_count = 0

        for wout in wouts:
            stem = wout.name
            if stem in seen:
                continue
            print(f"\n{'=' * 60}\nCoil gate: {stem}")
            case_dir = output_dir / stem.replace(".nc", "")
            feasibility_dir = case_dir / "feasibility"
            contour_dir = case_dir / f"contours_offset{offset:g}_lam{lambda_regularization:g}"
            feasibility_dir.mkdir(parents=True, exist_ok=True)
            contour_dir.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                [
                    sys.executable,
                    str(COIL_FEASIBILITY_SCRIPT),
                    "--wout",
                    str(wout),
                    "--output_dir",
                    str(feasibility_dir),
                    "--offsets",
                    str(offset),
                    "--lambdas",
                    str(lambda_regularization),
                    "--M_Phi",
                    "4",
                    "--N_Phi",
                    "4",
                    "--desc_L",
                    "4",
                    "--desc_M",
                    "4",
                    "--desc_N",
                    "4",
                    "--source_M",
                    "16",
                    "--source_N",
                    "16",
                    "--eval_M",
                    "16",
                    "--eval_N",
                    "16",
                    "--winding_surface_method",
                    "normal",
                ],
                cwd=str(REPO_ROOT),
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(COIL_CONTOUR_SCRIPT),
                    "--wout",
                    str(wout),
                    "--output_dir",
                    str(contour_dir),
                    "--offset",
                    str(offset),
                    "--lambda_regularization",
                    str(lambda_regularization),
                    "--num_coils",
                    "4",
                    "--M_Phi",
                    "4",
                    "--N_Phi",
                    "4",
                    "--desc_L",
                    "4",
                    "--desc_M",
                    "4",
                    "--desc_N",
                    "4",
                    "--source_M",
                    "16",
                    "--source_N",
                    "16",
                    "--eval_M",
                    "16",
                    "--eval_N",
                    "16",
                    "--winding_surface_method",
                    "normal",
                ],
                cwd=str(REPO_ROOT),
                check=True,
            )

            coil_row = _latest_row(feasibility_dir / "coil_feasibility_scan.csv")
            contour = _load_summary(contour_dir / "coil_contour_metrics.json")
            bn = float(coil_row["Bn_max_abs_unitless"])
            bn_avg = float(coil_row["Bn_avg_abs_unitless"])
            k_rms = float(coil_row["K_rms"]) / 1e6
            k_max = float(coil_row["K_max"]) / 1e6
            phi_hi = float(coil_row["phi_high_mode_fraction"])
            curv = float(contour["curvature_max"])
            curv_rms = float(contour["curvature_rms_max"])
            d_plasma = float(contour["min_coil_plasma_distance"])
            d_coil = float(contour["min_coil_coil_distance_unique"])

            verdict = "PASS"
            if bn > bn_limit:
                verdict = "FAIL_BN"
            elif curv > curvature_max_limit:
                verdict = "FAIL_CURVATURE_MAX"
            elif curv_rms > curvature_rms_limit:
                verdict = "FAIL_CURVATURE_RMS"
            elif d_plasma < min_plasma_distance:
                verdict = "FAIL_PLASMA_DISTANCE"
            elif d_coil < min_coil_distance:
                verdict = "FAIL_COIL_DISTANCE"

            with open(summary_path, "a") as f:
                f.write(
                    f"{stem},{offset},{lambda_regularization},{bn:.8g},{bn_avg:.8g},"
                    f"{k_rms:.8g},{k_max:.8g},{phi_hi:.8g},"
                    f"{contour['length_mean']:.8g},{curv:.8g},{curv_rms:.8g},"
                    f"{d_plasma:.8g},{d_coil:.8g},{verdict}\n"
                )
            print(
                f"  {verdict}: Bn={bn:.3e}, curv={curv:.2f}, "
                f"curv_rms={curv_rms:.2f}, d_plasma={d_plasma:.3f}, d_coil={d_coil:.3f}"
            )
            seen.add(stem)
            new_count += 1

        if not watch:
            break
        if new_count == 0:
            print(f"[{time.strftime('%H:%M:%S')}] No new checkpoints. Waiting {min_interval}s ...")
            time.sleep(min_interval)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Processed {new_count} checkpoint(s).")

    print(f"\nWrote {summary_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--min_interval", type=int, default=60)
    ap.add_argument("--offset", type=float, default=0.35)
    ap.add_argument("--lambda_regularization", type=float, default=1e-8)
    ap.add_argument("--bn_limit", type=float, default=4.5e-3)
    ap.add_argument("--curvature_max_limit", type=float, default=8.0)
    ap.add_argument("--curvature_rms_limit", type=float, default=4.0)
    ap.add_argument("--min_plasma_distance", type=float, default=0.06)
    ap.add_argument("--min_coil_distance", type=float, default=0.70)
    args = ap.parse_args(argv)
    gate_checkpoints(**vars(args))
    return 0
