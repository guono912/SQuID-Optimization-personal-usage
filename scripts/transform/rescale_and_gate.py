#!/usr/bin/env python3
"""
Rescale a VMEC configuration from one R0 to another, then run MHD gate.

Usage:
    python scripts/transform/rescale_and_gate.py \
        --wout path/to/wout.nc --target_rmajor 2.5 \
        --output_dir gate_results/
"""
import argparse
import os
import shutil
import sys
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
    from simsopt.mhd import Vmec
    from squid.utils.vmec_input import wout_to_input

    source = Path(wout_src).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(wout_src).stem}_R{str(target_rmajor).replace('.', 'p')}"
    input_path = output_dir / f"input.{stem}"
    # The historical NFP4 transform uses free iota with zero toroidal current.
    wout_to_input(source, input_path, ns=ns, nfp_out=4,
                  target_rmajor=target_rmajor, zero_current=True)
    with _run_in_directory(output_dir):
        vmec = Vmec(str(input_path))
        vmec.run()
        output = output_dir / f"{stem}.nc"
        shutil.copyfile(vmec.output_file, output)
    return str(output)


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
