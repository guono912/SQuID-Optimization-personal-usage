#!/usr/bin/env python3
"""
SQuID Stellarator Optimiser — unified entry point.

Complete PRX Energy target function:
    f_SQuID = w_QI * f_QI
            + w_maxJ * f_maxJ
            + w_Bmin * f_Bmin
            + w_AR * f_A
            + w_mirror * f_delta
            + w_beta * f_beta
            + w_iota * f_iota
            + w_grad_s * f_nabla_s
            + w_reg * f_reg

Backends:
  - VMEC (via simsopt): fastest, requires VMEC2000 Fortran extension
  - DESC (fallback):     pure Python, automatically used when VMEC absent

Usage:
    python scripts/optimize.py \\
        --nc_file path/to/wout_xxx.nc \\
        --maxiter 10
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Backend detection
HAS_VMEC = False
HAS_DESC = False
try:
    import vmec as _vmec_mod  # noqa: F401
    HAS_VMEC = True
except ImportError:
    pass
try:
    import desc as _desc_mod  # noqa: F401
    HAS_DESC = True
except ImportError:
    pass


MODE_PRESETS = {
    "core": {
        "w_qi": 1.0,
        "w_qi_r2": 0.0,
        "w_maxj": 1.0,
        "w_bmin": 1.0,
        "w_ar": 100.0,
        "w_reg": 10.0,
        "w_mirror": 0.0,
        "w_beta": 0.0,
        "w_iota": 0.0,
        "w_grad_s": 0.0,
        "w_well": 0.0,
        "qi_r2_nphi": 101,
        "qi_r2_nalpha": 8,
        "qi_r2_nbj": 101,
        "qi_r2_mpol": 8,
        "qi_r2_ntor": 8,
    },
    "core_r2_assist": {
        "w_qi": 1.0,
        "w_qi_r2": 0.1,
        "w_maxj": 1.0,
        "w_bmin": 1.0,
        "w_ar": 100.0,
        "w_reg": 10.0,
        "w_mirror": 0.0,
        "w_beta": 0.0,
        "w_iota": 0.0,
        "w_grad_s": 0.0,
        "w_well": 0.0,
        "qi_r2_nphi": 101,
        "qi_r2_nalpha": 8,
        "qi_r2_nbj": 101,
        "qi_r2_mpol": 8,
        "qi_r2_ntor": 8,
    },
    "maxj_repair": {
        "w_qi": 0.5,
        "w_qi_r2": 0.05,
        "w_maxj": 4.0,
        "w_bmin": 3.0,
        "w_ar": 100.0,
        "w_reg": 15.0,
        "w_mirror": 0.0,
        "w_beta": 0.0,
        "w_iota": 0.0,
        "w_grad_s": 0.0,
        "w_well": 0.0,
        "qi_r2_nphi": 101,
        "qi_r2_nalpha": 8,
        "qi_r2_nbj": 101,
        "qi_r2_mpol": 8,
        "qi_r2_ntor": 8,
    },
    "engineering": {
        "w_qi": 0.8,
        "w_qi_r2": 0.05,
        "w_maxj": 2.0,
        "w_bmin": 2.0,
        "w_ar": 150.0,
        "w_reg": 15.0,
        "w_mirror": 10.0,
        "w_beta": 0.0,
        "w_iota": 10.0,
        "w_grad_s": 0.0,
        "w_well": 20.0,
        "qi_r2_nphi": 101,
        "qi_r2_nalpha": 8,
        "qi_r2_nbj": 101,
        "qi_r2_mpol": 8,
        "qi_r2_ntor": 8,
    },
}


CONFIG_GROUPS = {
    "numerics": {
        "maxiter", "max_dofs", "num_alpha", "num_pitch", "num_surfaces",
        "ns_vmec", "abs_step", "rel_step", "perturb",
    },
    "weights": {
        "w_qi", "w_maxj", "w_bmin", "w_qi_r2", "w_ar", "w_reg",
        "w_mirror", "w_beta", "w_iota", "w_grad_s", "w_well",
    },
    "targets": {
        "aspect_target", "mirror_target", "beta_target", "iota_ax",
        "iota_edge", "iota_tolerance", "target_well", "bmin_slope_target",
        "grad_s_smin", "grad_s_smax", "grad_s_ns", "itg_method",
    },
    "r2": {
        "qi_r2_nphi", "qi_r2_nalpha", "qi_r2_nbj", "qi_r2_mpol",
        "qi_r2_ntor", "qi_r2_arr_out",
    },
    "desc": {"desc_L", "desc_M", "desc_N"},
}

CONFIG_ALIASES = {
    "r2.nphi": "qi_r2_nphi",
    "r2.nalpha": "qi_r2_nalpha",
    "r2.nbj": "qi_r2_nbj",
    "r2.mpol": "qi_r2_mpol",
    "r2.ntor": "qi_r2_ntor",
    "r2.arr_out": "qi_r2_arr_out",
}


def _build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="SQuID stellarator optimiser (complete PRX Energy target)"
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--input_parameter", type=str, default=defaults.get("input_parameter"),
                        help="JSON parameter file. CLI arguments override file values.")
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), default=defaults.get("mode", "core"),
                        help="Optimisation preset used before file/CLI overrides")
    parser.add_argument("--run_dir", type=str, default=defaults.get("run_dir"),
                        help="Directory for resolved parameters and run metadata")
    parser.add_argument("--run_name", type=str, default=defaults.get("run_name"),
                        help="Run name used when run_dir is not specified")
    parser.add_argument("--nc_file", type=str, default=defaults.get("nc_file"),
                        help="Initial VMEC wout .nc file")
    parser.add_argument("--maxiter", type=int, default=defaults.get("maxiter", 10))
    parser.add_argument("--aspect_target", type=float, default=defaults.get("aspect_target"))
    parser.add_argument("--max_dofs", type=int, default=defaults.get("max_dofs", 15))
    parser.add_argument("--num_alpha", type=int, default=defaults.get("num_alpha", 4))
    parser.add_argument("--num_pitch", type=int, default=defaults.get("num_pitch", 20))
    parser.add_argument("--num_surfaces", type=int, default=defaults.get("num_surfaces", 3))
    parser.add_argument("--ns_vmec", type=int, default=defaults.get("ns_vmec", 31))

    # --- Core objectives (always active) ---
    parser.add_argument("--w_qi", type=float, default=defaults.get("w_qi", 1.0),
                        help="Simple QI penalty weight [core, always on]")
    parser.add_argument("--w_maxj", type=float, default=defaults.get("w_maxj", 1.0),
                        help="max-J penalty weight [core, always on]")
    parser.add_argument("--w_bmin", type=float, default=defaults.get("w_bmin", 1.0),
                        help="B_min radial-growth penalty weight [core, always on]")
    parser.add_argument("--w_qi_r2", type=float, default=defaults.get("w_qi_r2", 0.0),
                        help="R2 squash-stretch-shuffle QI weight (0 = disabled)")
    parser.add_argument("--w_ar", type=float, default=defaults.get("w_ar", 100.0),
                        help="Aspect ratio penalty weight [core, always on]")
    parser.add_argument("--w_reg", type=float, default=defaults.get("w_reg", 10.0),
                        help="Regularisation weight [core, always on]")

    # --- Optional objectives (off by default, 0 = disabled) ---
    parser.add_argument("--w_mirror", type=float, default=defaults.get("w_mirror", 0.0),
                        help="Mirror ratio penalty weight (0 = disabled)")
    parser.add_argument("--w_beta", type=float, default=defaults.get("w_beta", 0.0),
                        help="Plasma beta upper-bound penalty weight (0 = disabled)")
    parser.add_argument("--w_iota", type=float, default=defaults.get("w_iota", 0.0),
                        help="Iota tolerance-hinge penalty weight (0 = disabled)")
    parser.add_argument("--w_grad_s", type=float, default=defaults.get("w_grad_s", 0.0),
                        help="ITG grad-s penalty weight (0 = disabled)")
    parser.add_argument("--w_well", type=float, default=defaults.get("w_well", 0.0),
                        help="Magnetic well penalty weight (0 = disabled)")

    # --- Targets for optional objectives ---
    parser.add_argument("--mirror_target", type=float, default=defaults.get("mirror_target", 0.20),
                        help="Upper bound for mirror ratio (only used when w_mirror > 0)")
    parser.add_argument("--beta_target", type=float, default=defaults.get("beta_target", 0.02),
                        help="Upper bound for total beta (only used when w_beta > 0)")
    parser.add_argument("--iota_ax", type=float, default=defaults.get("iota_ax"),
                        help="Axis iota soft target, or hard AI target with --prescribe_iota")
    parser.add_argument("--iota_edge", type=float, default=defaults.get("iota_edge"),
                        help="Edge iota soft target, or hard AI target with --prescribe_iota")
    parser.add_argument("--iota_tolerance", type=float, default=defaults.get("iota_tolerance", 0.01),
                        help="Allowed |iota - target| before iota penalty turns on")
    parser.add_argument("--target_well", type=float, default=defaults.get("target_well", 0.01),
                        help="Target magnetic well depth (only used when w_well > 0)")
    parser.add_argument("--bmin_slope_target", type=float, default=defaults.get("bmin_slope_target", 0.01),
                        help="Minimum target for normalised outward B_min growth")
    parser.add_argument("--grad_s_smin", type=float, default=defaults.get("grad_s_smin", 0.1))
    parser.add_argument("--grad_s_smax", type=float, default=defaults.get("grad_s_smax", 0.5))
    parser.add_argument("--grad_s_ns", type=int, default=defaults.get("grad_s_ns", 3))
    parser.add_argument("--itg_method", choices=["drift_curvature", "vacuum_dBds"],
                        default=defaults.get("itg_method", "drift_curvature"),
                        help="Bad-curvature method for f_nabla_s")
    parser.add_argument("--qi_r2_nphi", type=int, default=defaults.get("qi_r2_nphi", 301))
    parser.add_argument("--qi_r2_nalpha", type=int, default=defaults.get("qi_r2_nalpha", 24))
    parser.add_argument("--qi_r2_nbj", type=int, default=defaults.get("qi_r2_nbj", 301))
    parser.add_argument("--qi_r2_mpol", type=int, default=defaults.get("qi_r2_mpol", 12))
    parser.add_argument("--qi_r2_ntor", type=int, default=defaults.get("qi_r2_ntor", 12))
    parser.add_argument("--qi_r2_arr_out", action="store_true",
                        default=defaults.get("qi_r2_arr_out", False),
                        help="Use full (surface, alpha, phi) R2 residuals instead of per-alpha RMS")

    parser.add_argument("--abs_step", type=float, default=defaults.get("abs_step", 1e-4),
                        help="Absolute FD step for Jacobian (simsopt default 1e-7)")
    parser.add_argument("--rel_step", type=float, default=defaults.get("rel_step", 0.0),
                        help="Relative FD step for Jacobian")
    parser.add_argument("--perturb", type=float, default=defaults.get("perturb", 0.0),
                        help="Random perturbation amplitude (fraction of |x|) to escape local minima")
    parser.add_argument("--free_iota", dest="free_iota", action="store_true",
                        default=defaults.get("free_iota", True),
                        help="Use NCURR=1 so iota is computed self-consistently (default)")
    parser.add_argument("--prescribe_iota", dest="free_iota", action="store_false",
                        help="Use NCURR=0 and write iota_ax/iota_edge as hard AI constraints")

    parser.add_argument("--backend", choices=["auto", "vmec", "desc"],
                        default=defaults.get("backend", "auto"))
    parser.add_argument("--desc_L", type=int, default=defaults.get("desc_L", 4))
    parser.add_argument("--desc_M", type=int, default=defaults.get("desc_M", 4))
    parser.add_argument("--desc_N", type=int, default=defaults.get("desc_N", 4))
    return parser


def _flatten_config(config):
    flat = {}
    for key, value in config.items():
        if key in CONFIG_GROUPS and isinstance(value, dict):
            for subkey, subvalue in value.items():
                candidate = f"{key}.{subkey}"
                flat[CONFIG_ALIASES.get(candidate, subkey)] = subvalue
        elif key == "r2" and isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[CONFIG_ALIASES.get(f"r2.{subkey}", subkey)] = subvalue
        else:
            flat[key] = value
    return flat


def _load_config(path):
    if not path:
        return {}
    with open(path) as fh:
        return _flatten_config(json.load(fh))


def _jsonify(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_resolved_parameters(args, config):
    run_name = args.run_name or config.get("name")
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(args.nc_file).stem if args.nc_file else "squid"
        run_name = f"{stamp}_{args.mode}_{stem}"
    run_dir = Path(args.run_dir or Path("runs") / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    args.run_name = run_name

    resolved = vars(args).copy()
    resolved["_input_parameter_file"] = args.input_parameter
    resolved["_source_config"] = config
    path = run_dir / "input_parameter.resolved.json"
    with open(path, "w") as fh:
        json.dump(resolved, fh, indent=2, default=_jsonify)
    return path


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--input_parameter", type=str)
    pre.add_argument("--mode", choices=sorted(MODE_PRESETS), default="core")
    pre_args, _ = pre.parse_known_args()

    config = _load_config(pre_args.input_parameter)
    cli_mode_set = any(
        arg == "--mode" or arg.startswith("--mode=")
        for arg in sys.argv[1:]
    )
    mode = pre_args.mode if cli_mode_set else config.get("mode", pre_args.mode)
    defaults = MODE_PRESETS.get(mode, MODE_PRESETS["core"]).copy()
    defaults.update(config)
    defaults["mode"] = mode
    defaults["input_parameter"] = pre_args.input_parameter

    parser = _build_parser(defaults)
    args = parser.parse_args()
    if not args.nc_file:
        parser.error("--nc_file is required unless provided by --input_parameter")
    resolved_path = _write_resolved_parameters(args, config)

    print(f"\n{'=' * 60}")
    print("SQuID Optimisation  --  unified squid package")
    print(f"{'=' * 60}")
    print(f"  Input : {args.nc_file}")
    print(f"  Mode  : {args.mode}")
    print(f"  Run   : {args.run_dir}")
    print(f"  Params: {resolved_path}")

    use_desc = False
    if args.backend == "vmec":
        if not HAS_VMEC:
            print("  ERROR: --backend=vmec but VMEC extension not installed.")
            sys.exit(1)
    elif args.backend == "desc":
        use_desc = True
    else:
        if HAS_VMEC:
            use_desc = False
        elif HAS_DESC:
            use_desc = True
        else:
            print("  ERROR: neither VMEC nor DESC is installed.")
            sys.exit(1)

    if use_desc:
        from squid.backends.desc_backend import run_desc
        run_desc(args)
    else:
        from squid.backends.vmec_backend import run_vmec
        run_vmec(args)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
