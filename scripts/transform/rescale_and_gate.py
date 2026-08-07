#!/usr/bin/env python3
"""
Rescale a VMEC configuration from one R0 to another, then run MHD gate.

Usage:
    python scripts/transform/rescale_and_gate.py \
        --wout path/to/wout.nc --target_rmajor 2.5 \
        --output_dir gate_results/
"""
import os, sys, json, argparse, subprocess, shutil, glob
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.diagnostics.mercier_normalization import mercier_summary
from squid.evaluation.gates import gate as mhd_gate


@contextmanager
def _run_in_directory(path):
    """Keep legacy VMEC relative outputs inside the requested run directory."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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
    subprocess.run(cmd, check=True, cwd=str(output_dir))

    from simsopt.mhd import Vmec
    vmec = Vmec(str(output_dir / name))
    with _run_in_directory(output_dir):
        vmec.run()

    # simsopt VMEC names wout as "wout_{input_basename}_{run}.nc"
    base = Path(name).stem
    wout_files = glob.glob(os.path.join(str(output_dir), f"wout_{base}_*.nc"))
    if not wout_files:
        # try broader pattern
        wout_files = glob.glob(os.path.join(str(output_dir), "wout_*.nc"))
    if wout_files:
        latest = max(wout_files, key=os.path.getmtime)
        wout_out = output_dir / f"{stem}_R{str(target_rmajor).replace('.','p')}.nc"
        shutil.copy(latest, wout_out)
        return str(wout_out)
    raise FileNotFoundError("VMEC wout not found after rescale")


def gate(wout_path, output_dir):
    """Run MHD gate on wout."""
    return mhd_gate(wout_path, output_dir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wout", required=True)
    ap.add_argument("--target_rmajor", type=float, required=True)
    ap.add_argument("--output_dir", default="runs/transform_gate_latest")
    ap.add_argument("--ns", type=int, default=31)
    args = ap.parse_args(argv)

    print(f"\n=== Rescale: {Path(args.wout).stem} → R0={args.target_rmajor}m ===")

    wout_r2p5 = rescale(args.wout, args.target_rmajor, args.output_dir, ns=args.ns)

    import netCDF4
    ds = netCDF4.Dataset(wout_r2p5, 'r')
    mercier = mercier_summary(ds, s_min=0.1, s_max=0.97)
    print(f"  Rescale VMEC: β={float(ds.variables['betatotal'][:])*100:.2f}%, "
          f"ι=[{float(ds.variables['iotaf'][0]):.4f},{float(ds.variables['iotaf'][-1]):.4f}], "
          f"Phi_edge^2*DMerc_min={mercier['dmerc_flux_normalized_min']:.3e}, "
          f"raw={mercier['dmerc_vmec_raw_min']:.3e}, "
          f"neg={mercier['dmerc_negative_count']}")
    ds.close()

    print(f"\n=== Gate at R0={args.target_rmajor}m ===")
    report = gate(wout_r2p5, args.output_dir)
    print(f"  Verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
