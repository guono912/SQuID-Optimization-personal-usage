#!/usr/bin/env python3
"""Audit the SQuID Mercier normalization on an existing VMEC wout."""

import argparse
import json
import os
import sys

import netCDF4
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.diagnostics.mercier_normalization import mercier_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wout")
    parser.add_argument("--s-min", type=float, default=0.1)
    parser.add_argument("--s-max", type=float, default=0.95)
    args = parser.parse_args()

    with netCDF4.Dataset(args.wout) as dataset:
        summary = mercier_summary(dataset, args.s_min, args.s_max)

    raw = summary["dmerc_vmec_raw_min"]
    normalized = summary["dmerc_flux_normalized_min"]
    expected_scale = summary["edge_toroidal_flux_wb"] ** 2
    measured_scale = normalized / raw if abs(raw) > 0.0 else expected_scale
    identity_error = abs(measured_scale - expected_scale)
    tolerance = 1.0e-12 * max(1.0, abs(expected_scale))
    output = {
        "wout": os.path.abspath(args.wout),
        "protocol": summary["convention"],
        "formula": summary["formula"],
        "s_min": summary["s_min"],
        "s_max": summary["s_max"],
        "dmerc_min_s": summary["minimum_s"],
        "dmerc_edge_toroidal_flux_wb": summary["edge_toroidal_flux_wb"],
        "dmerc_vmec_raw_min": raw,
        "dmerc_flux_normalized_min": normalized,
        "dmerc_negative_count": summary["dmerc_negative_count"],
        "normalization_identity_abs_error": identity_error,
        "normalization_identity_pass": bool(
            np.isfinite(identity_error) and identity_error <= tolerance
        ),
    }
    print(json.dumps(output, indent=2))
    if not output["normalization_identity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
