#!/usr/bin/env python3
"""
Rescale a VMEC configuration from one R0 to another, then run MHD gate.

Usage:
    python scripts/transform/rescale_and_gate.py \
        --wout path/to/wout.nc --target_rmajor 2.5 \
        --output_dir gate_results/
"""
import argparse
import json
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


def rescale(wout_src, target_rmajor, output_dir, ns=31, *, nfp_out=None, current="preserve"):
    """Rescale wout to new R0 and run VMEC."""
    from simsopt.mhd import Vmec
    from squid.utils.vmec_input import wout_to_input

    source = Path(wout_src).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(wout_src).stem}_R{str(target_rmajor).replace('.', 'p')}"
    input_path = output_dir / f"input.{stem}"
    if current not in ("preserve", "zero") or (nfp_out is not None and nfp_out < 1):
        raise ValueError("select a positive NFP and current policy preserve or zero")
    print(f"  Field periods: {nfp_out or 'source'}; current policy: {current}")
    wout_to_input(source, input_path, ns=ns, nfp_out=nfp_out,
                  target_rmajor=target_rmajor, zero_current=(current == "zero"))
    with _run_in_directory(output_dir):
        vmec = Vmec(str(input_path))
        vmec.run()
        output = output_dir / f"{stem}.nc"
        shutil.copyfile(vmec.output_file, output)
    return str(output)


def gate(wout_path, output_dir, limits=None):
    """Run MHD gate on wout."""
    return mhd_gate(wout_path, output_dir, limits=limits)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wout", required=True)
    ap.add_argument("--target_rmajor", type=float, required=True)
    ap.add_argument("--output_dir", default="runs/transform_gate_latest")
    ap.add_argument("--ns", type=int, default=31)
    ap.add_argument("--preset", choices=["nfp4-zero-current"], help="historical four-period transform")
    ap.add_argument("--limits", help="MHD screening limits JSON")
    ap.add_argument("--nfp-out", type=int, help="output field periods (default: source NFP)")
    ap.add_argument("--current", choices=("preserve", "zero"),
                    help="retain source total current and profile coefficients, or explicitly remove current")
    args = ap.parse_args(argv)

    if args.preset:
        if args.nfp_out not in (None, 4) or args.current not in (None, "zero"):
            ap.error("the nfp4-zero-current preset requires NFP=4 and zero current")
        args.nfp_out, args.current = 4, "zero"
    limits = None
    if args.limits:
        with open(args.limits) as handle:
            limits = json.load(handle)
        if not isinstance(limits, dict):
            ap.error("--limits must contain a JSON object")

    print(f"\n=== Rescale: {Path(args.wout).stem} → R0={args.target_rmajor}m ===")

    scaled_wout = rescale(args.wout, args.target_rmajor, args.output_dir, ns=args.ns,
                           nfp_out=args.nfp_out, current=args.current or "preserve")

    import netCDF4
    ds = netCDF4.Dataset(scaled_wout, 'r')
    mercier = mercier_summary(ds, s_min=0.1, s_max=0.97)
    print(f"  Rescale VMEC: β={float(ds.variables['betatotal'][:])*100:.2f}%, "
          f"ι=[{float(ds.variables['iotaf'][0]):.4f},{float(ds.variables['iotaf'][-1]):.4f}], "
          f"Phi_edge^2*DMerc_min={mercier['dmerc_flux_normalized_min']:.3e}, "
          f"raw={mercier['dmerc_vmec_raw_min']:.3e}, "
          f"neg={mercier['dmerc_negative_count']}")
    ds.close()

    print(f"\n=== Gate at R0={args.target_rmajor}m ===")
    report = gate(scaled_wout, args.output_dir, limits=limits)
    print(f"  Verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
