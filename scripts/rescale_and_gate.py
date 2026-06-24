#!/usr/bin/env python3
"""
Rescale a VMEC configuration from one R0 to another, then run MHD gate.

Usage:
    python scripts/rescale_and_gate.py \
        --wout path/to/wout.nc --target_rmajor 2.5 \
        --output_dir gate_results/
"""
import os, sys, json, argparse, subprocess, shutil, glob
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def rescale(wout_src, target_rmajor, output_dir, ns=31):
    """Rescale wout to new R0 and run VMEC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wout_src).stem
    name = f"input.{stem}_R{str(target_rmajor).replace('.','p')}"

    cmd = [
        sys.executable,
        "/home/guozx/w7x_squid_nfp4/scripts/wout_to_nfp4_input.py",
        "--wout", wout_src,
        "--output", str(output_dir / name),
        "--nfp-out", "4",
        "--target-rmajor", str(target_rmajor),
        "--keep-pressure",
        "--free-iota",
        "--ns", str(ns),
    ]
    subprocess.run(cmd, check=True)

    from simsopt.mhd import Vmec
    vmec = Vmec(str(output_dir / name))
    vmec.run()

    # simsopt VMEC names wout as "wout_{input_basename}_{run}.nc"
    base = Path(name).stem
    wout_files = glob.glob(os.path.join(os.getcwd(), f"wout_{base}_*.nc"))
    if not wout_files:
        # try broader pattern
        wout_files = glob.glob(os.path.join(os.getcwd(), "wout_*.nc"))
    if wout_files:
        latest = max(wout_files, key=os.path.getmtime)
        wout_out = output_dir / f"{stem}_R{str(target_rmajor).replace('.','p')}.nc"
        shutil.copy(latest, wout_out)
        return str(wout_out)
    raise FileNotFoundError("VMEC wout not found after rescale")


def gate(wout_path, output_dir):
    """Run MHD gate on wout."""
    from scripts.mhd_gate import gate as mhd_gate
    return mhd_gate(wout_path, output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wout", required=True)
    ap.add_argument("--target_rmajor", type=float, required=True)
    ap.add_argument("--output_dir", default="gate_results")
    ap.add_argument("--ns", type=int, default=31)
    args = ap.parse_args()

    print(f"\n=== Rescale: {Path(args.wout).stem} → R0={args.target_rmajor}m ===")

    wout_r2p5 = rescale(args.wout, args.target_rmajor, args.output_dir, ns=args.ns)

    import netCDF4, numpy as np
    ds = netCDF4.Dataset(wout_r2p5, 'r')
    dmerc = np.array(ds.variables['DMerc'][:])
    s = np.linspace(0, 1, len(dmerc))
    mask = (s >= 0.1) & (s <= 0.97)
    print(f"  Rescale VMEC: β={float(ds.variables['betatotal'][:])*100:.2f}%, "
          f"ι=[{float(ds.variables['iotaf'][0]):.4f},{float(ds.variables['iotaf'][-1]):.4f}], "
          f"DMerc_min={float(np.min(dmerc[mask])):.3f}, neg={int(np.sum(dmerc[mask]<0))}")
    ds.close()

    print(f"\n=== Gate at R0={args.target_rmajor}m ===")
    report = gate(wout_r2p5, args.output_dir)
    print(f"  Verdict: {report['verdict']}")
