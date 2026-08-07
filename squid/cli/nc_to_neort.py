"""Command-line interface for Boozer-to-NEO-RT conversion."""

from __future__ import annotations

import argparse

from ..utils.nc_to_neort import convert_boozmn_to_neort


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a boozmn netCDF file to the NEO-RT in_file format"
    )
    parser.add_argument("boozmn", help="Input boozmn_*.nc file")
    parser.add_argument("--output", required=True, help="Output NEO-RT text file")
    parser.add_argument(
        "--s-values", type=float, nargs="+", default=None,
        help="Optional normalized-flux surfaces to export",
    )
    parser.add_argument("--ns-vmec", type=int, default=None)
    parser.add_argument("--flux-override", type=float, default=None)
    parser.add_argument("--a-override", type=float, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = convert_boozmn_to_neort(
        args.boozmn,
        output_path=args.output,
        s_values=args.s_values,
        ns_vmec=args.ns_vmec,
        flux_override=args.flux_override,
        a_override=args.a_override,
    )
    return 0 if result is not None else 1
