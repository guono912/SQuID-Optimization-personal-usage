#!/usr/bin/env python3
"""
Batch gate evaluation on SQuID checkpoint wout files.

Runs DESC force-balance + ballooning + ripple on every wout in the
checkpoints directory, producing a summary CSV and individual gate JSONs.

Usage:
    python scripts/gate_checkpoints.py --run_dir runs/edge_bal_repair
    python scripts/gate_checkpoints.py --run_dir runs/edge_bal_repair --watch
"""

import os, sys, json, time, argparse, glob, subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def gate_checkpoints(run_dir, output_dir=None, watch=False, min_interval=60):
    """Run MHD gate on all checkpoint wout files."""
    from scripts.mhd_gate import gate

    run_dir = Path(run_dir)
    cp_dir = run_dir / "checkpoints"
    if output_dir is None:
        output_dir = run_dir / "gate_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gate_summary.csv"
    seen = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    seen.add(line.split(",")[0])

    header = ("wout,ier_flag,aspect,betatotal,iota_axis,iota_edge,"
              "DMerc_min,DMerc_neg,well_depth,fb_rms_final,"
              "lam_max,lam_min,n_unstable,ripple_peak,verdict\n")

    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(header)

    def _ensure_checkpoint_wouts():
        inputs = sorted(cp_dir.glob("input.squid_eval_*"))
        for inp in inputs:
            suffix = inp.name.replace("input.", "")
            wout = cp_dir / f"wout_{suffix}.nc"
            if wout.exists():
                continue
            print(f"  VMEC checkpoint solve: {inp.name}")
            subprocess.run(["xvmec", inp.name], cwd=str(cp_dir), check=True)

    while True:
        _ensure_checkpoint_wouts()
        wout_files = sorted(
            glob.glob(str(cp_dir / "wout_*.nc")),
            key=os.path.getmtime,
        )

        new_count = 0
        for wf in wout_files:
            stem = os.path.basename(wf)
            if stem in seen:
                continue
            print(f"\n{'='*60}")
            print(f"Gate: {stem}")
            try:
                report = gate(wf, str(output_dir))
                verdict = report["verdict"]
                v = report["vmec"]
                d = report["desc"]
                b = report["ballooning"]
                r = report["ripple"]

                row = (f"{stem},{v['ier_flag']},{v['aspect']:.4f},{v['betatotal']:.6f},"
                       f"{v['iotaf'][0]:.6f},{v['iotaf'][1]:.6f},"
                       f"{v['DMerc_min_gated']:.4f},{v['DMerc_negative_count']},"
                       f"{v['well_depth']:.4f},{d['fb_rms_final']:.2e},"
                       f"{b['lam_max']:.2e},{b['lam_min']:.2e},{b['n_unstable']},"
                       f"{r['peak']:.4f},{verdict}\n")
                with open(csv_path, "a") as f:
                    f.write(row)
                seen.add(stem)
                new_count += 1

                # Ballooning PASS indicator
                if b["n_unstable"] == 0:
                    print(f"  *** BALLOONING PASS: λ_max={b['lam_max']:.2e} < 0 ***")
                elif b["lam_max"] < 0:
                    print(f"  NOTE: lam_max < 0 but n_unstable > 0 — check grid")
            except Exception as e:
                print(f"  GATE FAILED: {e}")
                import traceback
                traceback.print_exc()

        if not watch:
            break
        if new_count == 0:
            print(f"[{time.strftime('%H:%M:%S')}] No new checkpoints. "
                  f"Waiting {min_interval}s ...")
            time.sleep(min_interval)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Processed {new_count} new "
                  f"checkpoint(s).")

    # Final summary
    if csv_path.exists():
        print(f"\n--- Gate Summary ---")
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if len(parts) >= 15:
                print(f"  {parts[0]}: lam_max={parts[10]} n_unstable={parts[12]} "
                      f"ripple={parts[13]} verdict={parts[14]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--min_interval", type=int, default=60)
    args = ap.parse_args()
    gate_checkpoints(args.run_dir, args.output_dir, args.watch, args.min_interval)
