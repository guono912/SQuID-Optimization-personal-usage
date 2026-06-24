#!/usr/bin/env python3
"""Recompose a VMEC input boundary with low-mode filtering."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COEFF_RE = re.compile(
    r"(RBC|ZBS)\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)"
)


def _parse(path: Path):
    text = path.read_text()
    nfp = _find_int(text, "NFP", 1)
    phiedge = _find_float(text, "PHIEDGE", 0.03141592653589793)
    profiles = {
        "pmass_type": _find_string(text, "PMASS_TYPE", "power_series"),
        "am": _find_array(text, "AM", "0.0"),
        "pres_scale": _find_float(text, "PRES_SCALE", 0.0),
        "ncurr": _find_int(text, "NCURR", 1),
        "curtor": _find_float(text, "CURTOR", 0.0),
        "pcurr_type": _find_string(text, "PCURR_TYPE", "power_series"),
        "ac": _find_array(text, "AC", "0.0"),
    }
    coeffs = {}
    for name, n, m, val in COEFF_RE.findall(text):
        coeffs[(name, int(n), int(m))] = float(val.replace("D", "E").replace("d", "e"))
    return text, nfp, phiedge, profiles, coeffs


def _find_int(text: str, key: str, default: int) -> int:
    m = re.search(rf"\b{key}\s*=\s*([+-]?\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else default


def _find_float(text: str, key: str, default: float) -> float:
    m = re.search(
        rf"\b{key}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)",
        text,
        re.IGNORECASE,
    )
    return float(m.group(1).replace("D", "E").replace("d", "e")) if m else default


def _find_string(text: str, key: str, default: str) -> str:
    m = re.search(rf"\b{key}\s*=\s*['\"]?([^'\"\n,]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def _find_array(text: str, key: str, default: str) -> str:
    m = re.search(rf"\b{key}\s*=\s*([^\n!]+)", text, re.IGNORECASE)
    if not m:
        return default
    return m.group(1).strip().rstrip(",")


def _write(out: Path, source: Path, old_nfp: int, phiedge: float, profiles: dict, coeffs: dict, args):
    r00 = coeffs.get(("RBC", 0, 0), 1.0)
    scale = args.R0 / r00 if abs(r00) > 1e-12 else args.R0
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for (name, n, m), val in coeffs.items():
        if m > args.mmax or abs(n) > args.nmax:
            continue
        if abs(val) < args.threshold and (name, n, m) != ("RBC", 0, 0):
            continue
        damp = 1.0
        if name == "ZBS" and m == 0 and n != 0:
            damp *= args.zm0_damp
        if args.taper:
            damp *= max(0.0, 1.0 - m / (args.mmax + 1.0)) * max(0.0, 1.0 - abs(n) / (args.nmax + 1.0))
            if (name, n, m) == ("RBC", 0, 0):
                damp = 1.0
        new_val = val * scale * damp
        if abs(new_val) >= 1e-16 or (name, n, m) == ("RBC", 0, 0):
            kept.append((name, n, m, new_val))

    with out.open("w") as f:
        f.write("&INDATA\n")
        f.write(f"! Recomposed from {source}\n")
        f.write(f"! old_nfp={old_nfp}, new_nfp={args.nfp}, R0={args.R0}, ")
        f.write(f"mmax={args.mmax}, nmax={args.nmax}, zm0_damp={args.zm0_damp}\n")
        f.write("  LASYM = F\n")
        f.write("  DELT = 0.9\n")
        f.write("  NSTEP = 200\n")
        f.write(f"  NFP = {args.nfp}\n")
        f.write(f"  MPOL = {args.mmax + 1}\n")
        f.write(f"  NTOR = {args.nmax}\n")
        f.write(f"  NS_ARRAY = {args.ns}\n")
        f.write("  NITER_ARRAY = 10000\n")
        f.write("  FTOL_ARRAY = 1.0E-11\n")
        f.write(f"  PHIEDGE = {phiedge * scale * scale:.15e}\n")
        if args.preserve_profiles:
            f.write(f'  PMASS_TYPE = "{profiles["pmass_type"]}"\n')
            f.write(f"  AM = {profiles['am']}\n")
            f.write(f"  PRES_SCALE = {profiles['pres_scale']:.15e}\n")
            f.write(f"  NCURR = {profiles['ncurr']}\n")
            f.write(f"  CURTOR = {profiles['curtor']:.15e}\n")
            f.write(f'  PCURR_TYPE = "{profiles["pcurr_type"]}"\n')
            f.write(f"  AC = {profiles['ac']}\n")
        else:
            f.write('  PMASS_TYPE = "power_series"\n')
            f.write("  PRES_SCALE = 0.0\n")
            f.write("  AM = 0.0\n")
            f.write("  NCURR = 1\n")
            f.write("  CURTOR = 0.0\n")
            f.write('  PCURR_TYPE = "power_series"\n')
            f.write("  AC = 0.0\n")
        for name, n, m, val in sorted(kept, key=lambda x: (x[2], x[1], x[0])):
            f.write(f"  {name}({n:4d},{m:4d}) = {val:.15e}\n")
        f.write("/\n")
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--nfp", type=int, default=4)
    ap.add_argument("--R0", type=float, default=2.5)
    ap.add_argument("--mmax", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--zm0-damp", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--ns", type=int, default=31)
    ap.add_argument("--taper", action="store_true")
    ap.add_argument("--preserve-profiles", action="store_true")
    args = ap.parse_args()

    source = Path(args.input)
    _, old_nfp, phiedge, profiles, coeffs = _parse(source)
    n_kept = _write(Path(args.output), source, old_nfp, phiedge, profiles, coeffs, args)
    print(f"Wrote {args.output}: kept {n_kept} coefficients")


if __name__ == "__main__":
    main()
