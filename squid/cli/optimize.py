"""SQuID Stellarator Optimiser — unified CLI implementation.

Reusable entry point: scripts/opt/optimize.py and the legacy
scripts/optimize.py wrapper both call into this module. The backend drivers
live in squid/backends/; this module holds the CLI, config resolution, run
directory handling, and dispatch.

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
    python scripts/opt/optimize.py \\
        --nc_file path/to/wout_xxx.nc \\
        --maxiter 10
"""

import sys
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..utils.run_paths import resolve_run_dir
from ..utils.jax_runtime import initialize_requested_jax_backend

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
    "edge_bal_repair": {
        "w_qi": 0.25,
        "w_qi_r2": 0.0,
        "w_maxj": 2.0,
        "w_bmin": 2.0,
        "w_ar": 120.0,
        "w_reg": 25.0,
        "w_mirror": 0.0,
        "w_beta": 0.0,
        "w_iota": 25.0,
        "w_grad_s": 0.0,
        "w_well": 30.0,
        "w_mercier": 0.0,
        "w_mercier_margin": 20.0,
        "qi_r2_nphi": 101,
        "qi_r2_nalpha": 8,
        "qi_r2_nbj": 101,
        "qi_r2_mpol": 8,
        "qi_r2_ntor": 8,
    },
    "edge_bal_direct": {
        "w_qi": 0.10,
        "w_qi_r2": 0.0,
        "w_maxj": 1.5,
        "w_bmin": 1.5,
        "w_ar": 100.0,
        "w_reg": 20.0,
        "w_mirror": 0.0,
        "w_beta": 0.0,
        "w_iota": 15.0,
        "w_grad_s": 0.0,
        "w_well": 15.0,
        "w_mercier": 0.0,
        "w_mercier_margin": 10.0,
        "w_ballooning": 8.0,
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
        "ns_vmec", "abs_step", "rel_step", "perturb", "max_total_evals",
        "checkpoint_every", "html_every", "wout_to_html_script", "dof_bound_frac",
        "free_m_max", "free_n_max", "mboz", "nboz", "vmec_input_file",
        "zero_current",
    },
    "weights": {
        "w_qi", "w_maxj", "w_bmin", "w_qi_r2", "w_ar", "w_reg",
        "w_mirror", "w_beta", "w_iota", "w_grad_s", "w_well", "w_mercier",
        "w_mercier_margin", "w_ballooning", "w_force_balance", "w_highB_topology",
        "w_coil_proxy_bn", "w_coil_proxy_k", "w_coil_proxy_phi", "w_shape_anchor",
        "w_boundary_curvature", "w_rational", "w_pdrot", "w_oops",
        "w_local_maxj", "w_maxj_guard", "w_oops_guard",
        "w_iota_profile", "w_iota_shear", "w_iota_monotonic",
    },
    "targets": {
        "aspect_target", "mirror_target", "beta_target", "iota_ax",
        "iota_edge", "iota_edge_mode", "iota_tolerance", "target_well",
        "iota_profile_s_min", "iota_profile_s_max", "iota_profile_ns",
        "iota_profile_tolerance", "iota_profile_scale",
        "iota_shear_absmin", "iota_shear_scale", "iota_monotonic_scale",
        "hard_iota_profile_max_dev", "hard_iota_shear_absmin",
        "hard_iota_monotonic_violation_max",
        "bmin_slope_target",
        "grad_s_smin", "grad_s_smax", "grad_s_ns", "itg_method",
        "mercier_s_min", "mercier_s_max",
        "mercier_flux_normalized_margin_target",
        "ballooning_lambda_target", "highB_target_phi_coverage", "highB_max_phi_gap",
        "coil_proxy_bn_max_target", "coil_proxy_k_rms_target_MApm",
        "coil_proxy_phi_high_target", "shape_anchor_frac", "shape_anchor_abs",
        "boundary_pdrot_aw_target", "boundary_pdrot_max_target",
        "boundary_pdrot_p99_target", "boundary_pdrot_cvar1_target",
        "boundary_k2_abs_max_target", "boundary_H_abs_max_target",
        "boundary_pdrot_aw_scale", "boundary_pdrot_max_scale",
        "boundary_pdrot_p99_scale", "boundary_pdrot_cvar1_scale",
        "boundary_k2_abs_scale", "boundary_H_abs_scale",
        "boundary_curvature_nphi", "boundary_curvature_ntheta",
        "rational_targets", "rational_hard_targets", "rational_s_min",
        "rational_s_max", "rational_min_distance", "rational_distance_scale",
        "rational_scan_ns",
        "ballooning_force_balance", "ballooning_fb_maxiter",
        "pdrot_q_target", "pdrot_rho_target", "pdrot_top_rho_target",
        "pdrot_top_fraction", "pdrot_weight_top_rho", "pdrot_delta_kappa_a",
        "oops_desc_source_path", "oops_desc_L", "oops_desc_M", "oops_desc_N",
        "oops_rhos", "oops_grid_m", "oops_grid_n",
        "oops_m_harmonics", "oops_n_harmonics", "oops_S_list",
        "oops_D_list", "oops_helicity", "oops_scalar_scale",
        "local_maxj_s_center", "local_maxj_s_width",
        "local_maxj_lambda_min", "local_maxj_lambda_max",
        "local_maxj_exceedance_target", "local_maxj_scale",
        "maxj_guard_rel_tol", "maxj_guard_abs_tol", "maxj_guard_scale",
        "oops_guard_rel_tol", "oops_guard_abs_tol", "oops_guard_scale",
    },
    "gates": {
        "hard_gate_mhd", "hard_dmerc_flux_normalized_min",
        "hard_dmerc_neg_max",
        "hard_ballooning_n_max", "hard_ballooning_lambda_max",
        "hard_beta_max", "hard_fb_rms_max", "hard_rational_crossing",
        "hard_coil_proxy_bn_max", "hard_gate_iota_topology",
    },
    "r2": {
        "qi_r2_nphi", "qi_r2_nalpha", "qi_r2_nbj", "qi_r2_mpol",
        "qi_r2_ntor", "qi_r2_arr_out",
    },
    "topology": {
        "highB_s_min", "highB_s_max", "highB_ns", "highB_threshold_min",
        "highB_threshold_max", "highB_n_thresholds", "highB_ntheta",
        "highB_nphi", "highB_mpol", "highB_ntor",
    },
    "desc": {"desc_L", "desc_M", "desc_N"},
    "coil_proxy": {
        "coil_proxy_offset_fraction", "coil_proxy_lambda_regularization",
        "coil_proxy_desc_L", "coil_proxy_desc_M", "coil_proxy_desc_N",
        "coil_proxy_M_Phi", "coil_proxy_N_Phi", "coil_proxy_source_M",
        "coil_proxy_source_N", "coil_proxy_eval_M", "coil_proxy_eval_N",
        "coil_proxy_current_helicity", "coil_proxy_regularization_type",
        "coil_proxy_vacuum", "coil_proxy_chunk_size", "coil_proxy_verbose",
    },
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
    parser.add_argument(
        "--overwrite_run_dir", action="store_true",
        default=bool(defaults.get("overwrite_run_dir", False)),
        help="Allow writing into a non-empty run directory (unsafe; explicit only)",
    )
    parser.add_argument("--nc_file", type=str, default=defaults.get("nc_file"),
                        help="Initial VMEC wout .nc file")
    parser.add_argument("--vmec_input_file", type=str,
                        default=defaults.get("vmec_input_file"),
                        help="Initial VMEC input file. Used as-is, preserving profiles/current.")
    parser.add_argument("--maxiter", type=int, default=defaults.get("maxiter", 10))
    parser.add_argument("--max_total_evals", type=int, default=defaults.get("max_total_evals", 0),
                        help="Hard cap on actual SQuID objective evaluations (0 = off)")
    parser.add_argument("--aspect_target", type=float, default=defaults.get("aspect_target"))
    parser.add_argument("--max_dofs", type=int, default=defaults.get("max_dofs", 15))
    parser.add_argument("--free_m_max", type=int, default=defaults.get("free_m_max", 4),
                        help="Maximum poloidal mode m allowed in VMEC boundary free-DoF list")
    parser.add_argument("--free_n_max", type=int, default=defaults.get("free_n_max", 3),
                        help="Maximum reduced toroidal mode |n| allowed in VMEC boundary free-DoF list")
    parser.add_argument("--num_alpha", type=int, default=defaults.get("num_alpha", 4))
    parser.add_argument("--num_pitch", type=int, default=defaults.get("num_pitch", 20))
    parser.add_argument("--num_surfaces", type=int, default=defaults.get("num_surfaces", 3))
    parser.add_argument("--mboz", type=int, default=defaults.get("mboz", 8),
                        help="Boozer poloidal resolution for SQuID core evaluation")
    parser.add_argument("--nboz", type=int, default=defaults.get("nboz", 8),
                        help="Boozer toroidal resolution for SQuID core evaluation")
    parser.add_argument("--ns_vmec", type=int, default=defaults.get("ns_vmec", 31))
    parser.add_argument("--checkpoint_every", type=int, default=defaults.get("checkpoint_every", 0),
                        help="Save VMEC input checkpoints every N SQuID evaluations (0 = off)")
    parser.add_argument("--html_every", type=int, default=defaults.get("html_every", 0),
                        help="Save VMEC wout+HTML checkpoints every N SQuID evaluations (0 = off)")
    parser.add_argument("--wout_to_html_script", type=str, default=defaults.get("wout_to_html_script"),
                        help="Path to wout_to_html.py for HTML checkpoint generation (optional)")

    # --- Core objectives (always active) ---
    parser.add_argument("--w_qi", type=float, default=defaults.get("w_qi", 1.0),
                        help="Simple QI penalty weight [core, always on]")
    parser.add_argument("--w_maxj", type=float, default=defaults.get("w_maxj", 1.0),
                        help="max-J penalty weight [core, always on]")
    parser.add_argument("--w_bmin", type=float, default=defaults.get("w_bmin", 1.0),
                        help="B_min radial-growth penalty weight [core, always on]")
    parser.add_argument("--w_qi_r2", type=float, default=defaults.get("w_qi_r2", 0.0),
                        help="R2 squash-stretch-shuffle QI weight (0 = disabled)")
    parser.add_argument("--w_highB_topology", type=float, default=defaults.get("w_highB_topology", 0.0),
                        help="High-|B| Boozer contour topology penalty weight (0 = disabled)")
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
    parser.add_argument("--w_iota_profile", type=float, default=defaults.get("w_iota_profile", 0.0),
                        help="Full iota-profile anchor weight (0 = disabled)")
    parser.add_argument("--w_iota_shear", type=float, default=defaults.get("w_iota_shear", 0.0),
                        help="Minimum |iota'| penalty weight (0 = disabled)")
    parser.add_argument("--w_iota_monotonic", type=float, default=defaults.get("w_iota_monotonic", 0.0),
                        help="Iota monotonicity penalty weight (0 = disabled)")
    parser.add_argument("--w_grad_s", type=float, default=defaults.get("w_grad_s", 0.0),
                        help="ITG grad-s penalty weight (0 = disabled)")
    parser.add_argument("--w_well", type=float, default=defaults.get("w_well", 0.0),
                        help="Magnetic well penalty weight (0 = disabled)")
    parser.add_argument("--w_mercier", type=float, default=defaults.get("w_mercier", 0.0),
                        help="Mercier DMerc positivity penalty weight (0 = disabled)")
    parser.add_argument("--w_mercier_margin", type=float, default=defaults.get("w_mercier_margin", 0.0),
                        help="Mercier positive-margin penalty weight (0 = disabled)")
    parser.add_argument("--w_ballooning", type=float, default=defaults.get("w_ballooning", 0.0),
                        help="DESC ballooning λ positivity penalty weight (0 = disabled)")
    parser.add_argument("--ballooning_rhos", type=float, nargs="+",
                        default=defaults.get("ballooning_rhos", [0.5, 0.65, 0.8, 0.9]),
                        help="ρ surfaces for ballooning penalty")
    parser.add_argument("--ballooning_lambda_target", type=float,
                        default=defaults.get("ballooning_lambda_target", 0.0),
                        help="Only penalize DESC ballooning λ above this target")
    parser.add_argument("--ballooning_force_balance", action="store_true",
                        default=defaults.get("ballooning_force_balance", False),
                        help="Solve fixed-boundary DESC force balance before ballooning, matching mhd_gate.py")
    parser.add_argument("--ballooning_fb_maxiter", type=int,
                        default=defaults.get("ballooning_fb_maxiter", 100),
                        help="DESC force-balance maxiter for in-loop ballooning when enabled")
    parser.add_argument("--w_force_balance", type=float, default=defaults.get("w_force_balance", 0.0),
                        help="DESC force-balance RMS penalty (0 = disabled)")
    parser.add_argument("--w_coil_proxy_bn", type=float, default=defaults.get("w_coil_proxy_bn", 0.0),
                        help="QSS coil proxy Bn_max hinge weight (0 = disabled)")
    parser.add_argument("--w_coil_proxy_k", type=float, default=defaults.get("w_coil_proxy_k", 0.0),
                        help="QSS coil proxy K_rms hinge weight (0 = disabled)")
    parser.add_argument("--w_coil_proxy_phi", type=float, default=defaults.get("w_coil_proxy_phi", 0.0),
                        help="QSS coil proxy current-potential spectrum hinge weight (0 = disabled)")
    parser.add_argument("--w_shape_anchor", type=float, default=defaults.get("w_shape_anchor", 0.0),
                        help="Hinge penalty for per-DoF boundary drift from the initial shape (0 = disabled)")
    parser.add_argument("--w_boundary_curvature", type=float,
                        default=defaults.get("w_boundary_curvature", 0.0),
                        help="Boundary curvature/twist hinge proxy weight (0 = disabled)")
    parser.add_argument("--w_rational", type=float, default=defaults.get("w_rational", 0.0),
                        help="Low-order iota rational-placement penalty weight (0 = disabled)")
    parser.add_argument("--w_pdrot", type=float, default=defaults.get("w_pdrot", 0.0),
                        help="Principal-direction rotation rate (pdrot) penalty weight (0 = disabled)")
    parser.add_argument("--w_oops", type=float, default=defaults.get("w_oops", 0.0),
                        help="OOPS / piecewise-omnigenity harmonics residual weight (0 = disabled)")
    parser.add_argument("--w_local_maxj", type=float, default=defaults.get("w_local_maxj", 0.0),
                        help="Localized max-J exceedance penalty weight (0 = disabled)")
    parser.add_argument("--w_maxj_guard", type=float, default=defaults.get("w_maxj_guard", 0.0),
                        help="Hinge guard against f_maxJ degradation relative to eval1 (0 = disabled)")
    parser.add_argument("--w_oops_guard", type=float, default=defaults.get("w_oops_guard", 0.0),
                        help="Hinge guard against OOPS scalar degradation relative to eval1 (0 = disabled)")

    # --- Targets for optional objectives ---
    parser.add_argument("--mirror_target", type=float, default=defaults.get("mirror_target", 0.20),
                        help="Upper bound for mirror ratio (only used when w_mirror > 0)")
    parser.add_argument("--beta_target", type=float, default=defaults.get("beta_target", 0.02),
                        help="Upper bound for total beta (only used when w_beta > 0)")
    parser.add_argument("--iota_ax", type=float, default=defaults.get("iota_ax"),
                        help="Axis iota soft target, or hard AI target with --prescribe_iota")
    parser.add_argument("--iota_edge", type=float, default=defaults.get("iota_edge"),
                        help="Edge iota soft target, or hard AI target with --prescribe_iota")
    parser.add_argument("--iota_edge_mode", choices=["target", "upper", "lower"],
                        default=defaults.get("iota_edge_mode", "target"),
                        help="How to apply iota_edge in the soft iota penalty")
    parser.add_argument("--iota_tolerance", type=float, default=defaults.get("iota_tolerance", 0.01),
                        help="Allowed |iota - target| before iota penalty turns on")
    parser.add_argument("--iota_profile_s_min", type=float, default=defaults.get("iota_profile_s_min", 0.1))
    parser.add_argument("--iota_profile_s_max", type=float, default=defaults.get("iota_profile_s_max", 0.95))
    parser.add_argument("--iota_profile_ns", type=int, default=defaults.get("iota_profile_ns", 17))
    parser.add_argument("--iota_profile_tolerance", type=float,
                        default=defaults.get("iota_profile_tolerance", 0.004),
                        help="Pointwise iota-profile deviation allowed before the anchor turns on")
    parser.add_argument("--iota_profile_scale", type=float, default=defaults.get("iota_profile_scale", 0.004))
    parser.add_argument("--iota_shear_absmin", type=float, default=defaults.get("iota_shear_absmin", 0.005),
                        help="Soft lower target for |d iota / ds| on the topology grid")
    parser.add_argument("--iota_shear_scale", type=float, default=defaults.get("iota_shear_scale", 0.0025))
    parser.add_argument("--iota_monotonic_scale", type=float, default=defaults.get("iota_monotonic_scale", 0.0025))
    parser.add_argument("--hard_iota_profile_max_dev", type=float,
                        default=defaults.get("hard_iota_profile_max_dev", float("inf")))
    parser.add_argument("--hard_iota_shear_absmin", type=float, default=defaults.get("hard_iota_shear_absmin", 0.0))
    parser.add_argument("--hard_iota_monotonic_violation_max", type=int,
                        default=defaults.get("hard_iota_monotonic_violation_max", 0))
    parser.add_argument("--target_well", type=float, default=defaults.get("target_well", 0.01),
                        help="Target magnetic well depth (only used when w_well > 0)")
    parser.add_argument("--bmin_slope_target", type=float, default=defaults.get("bmin_slope_target", 0.01),
                        help="Minimum target for normalised outward B_min growth")
    parser.add_argument("--mercier_s_min", type=float, default=defaults.get("mercier_s_min", 0.1),
                        help="Minimum s included in Mercier penalty")
    parser.add_argument("--mercier_s_max", type=float, default=defaults.get("mercier_s_max", 0.95),
                        help="Maximum s included in Mercier penalty")
    parser.add_argument(
        "--mercier_flux_normalized_margin_target", type=float,
        default=defaults.get("mercier_flux_normalized_margin_target", 0.0),
        help=("Target Phi_edge^2*DMerc margin used when w_mercier_margin > 0; "
              "calibrate positive margins within a declared protocol"),
    )
    parser.add_argument("--coil_proxy_bn_max_target", type=float,
                        default=defaults.get("coil_proxy_bn_max_target", 5e-3),
                        help="Target Bn_max/|B| for QSS coil proxy hinge")
    parser.add_argument("--coil_proxy_k_rms_target_MApm", type=float,
                        default=defaults.get("coil_proxy_k_rms_target_MApm", 2.2),
                        help="Target K_rms in MA/m for QSS coil proxy hinge")
    parser.add_argument("--coil_proxy_phi_high_target", type=float,
                        default=defaults.get("coil_proxy_phi_high_target", 0.50),
                        help="Target high-mode current-potential fraction")
    parser.add_argument("--shape_anchor_frac", type=float,
                        default=defaults.get("shape_anchor_frac", 0.015),
                        help="Allowed fractional drift per free boundary DoF before shape-anchor penalty")
    parser.add_argument("--shape_anchor_abs", type=float,
                        default=defaults.get("shape_anchor_abs", 1e-3),
                        help="Allowed absolute drift floor per free boundary DoF before shape-anchor penalty")
    parser.add_argument("--boundary_pdrot_aw_target", type=float,
                        default=defaults.get("boundary_pdrot_aw_target", 1.45),
                        help="Target area-weighted pdrot [1/m] for boundary curvature proxy")
    parser.add_argument("--boundary_pdrot_max_target", type=float,
                        default=defaults.get("boundary_pdrot_max_target", 16.0),
                        help="Target max pdrot [1/m] for boundary curvature proxy")
    parser.add_argument("--boundary_pdrot_p99_target", type=float,
                        default=defaults.get("boundary_pdrot_p99_target", 1e99),
                        help="Target p99 pdrot [1/m] for boundary top-tail proxy")
    parser.add_argument("--boundary_pdrot_cvar1_target", type=float,
                        default=defaults.get("boundary_pdrot_cvar1_target", 1e99),
                        help="Target top-1%% mean pdrot [1/m] for boundary top-tail proxy")
    parser.add_argument("--boundary_k2_abs_max_target", type=float,
                        default=defaults.get("boundary_k2_abs_max_target", 100.0),
                        help="Target max -k2 [1/m] for boundary curvature proxy")
    parser.add_argument("--boundary_H_abs_max_target", type=float,
                        default=defaults.get("boundary_H_abs_max_target", 50.0),
                        help="Target max -H [1/m] for boundary curvature proxy")
    parser.add_argument("--boundary_pdrot_aw_scale", type=float,
                        default=defaults.get("boundary_pdrot_aw_scale", 0.05))
    parser.add_argument("--boundary_pdrot_max_scale", type=float,
                        default=defaults.get("boundary_pdrot_max_scale", 1.0))
    parser.add_argument("--boundary_pdrot_p99_scale", type=float,
                        default=defaults.get("boundary_pdrot_p99_scale", 1.0))
    parser.add_argument("--boundary_pdrot_cvar1_scale", type=float,
                        default=defaults.get("boundary_pdrot_cvar1_scale", 1.0))
    parser.add_argument("--boundary_k2_abs_scale", type=float,
                        default=defaults.get("boundary_k2_abs_scale", 5.0))
    parser.add_argument("--boundary_H_abs_scale", type=float,
                        default=defaults.get("boundary_H_abs_scale", 3.0))
    parser.add_argument("--boundary_curvature_nphi", type=int,
                        default=defaults.get("boundary_curvature_nphi", 48))
    parser.add_argument("--boundary_curvature_ntheta", type=int,
                        default=defaults.get("boundary_curvature_ntheta", 48))
    parser.add_argument("--rational_targets", nargs="+",
                        default=defaults.get("rational_targets", ["3/4", "4/5", "5/7", "7/9"]),
                        help="Low-order iota rationals to avoid, e.g. 3/4 4/5")
    parser.add_argument("--rational_hard_targets", nargs="+",
                        default=defaults.get("rational_hard_targets", ["4/5"]),
                        help="Rationals for which crossings get a hard large residual")
    parser.add_argument("--rational_s_min", type=float,
                        default=defaults.get("rational_s_min", 0.1))
    parser.add_argument("--rational_s_max", type=float,
                        default=defaults.get("rational_s_max", 0.95))
    parser.add_argument("--rational_min_distance", type=float,
                        default=defaults.get("rational_min_distance", 0.003))
    parser.add_argument("--rational_distance_scale", type=float,
                        default=defaults.get("rational_distance_scale", 0.001))
    parser.add_argument("--rational_scan_ns", type=int,
                        default=defaults.get("rational_scan_ns", 96))
    parser.add_argument("--pdrot_q_target", type=float, default=defaults.get("pdrot_q_target"),
                        help="Target for dimensionless a_eff*pdrot hinge (None = unused)")
    parser.add_argument("--pdrot_rho_target", type=float, default=defaults.get("pdrot_rho_target"),
                        help="Target for absolute pdrot [1/m] hinge (None = unused)")
    parser.add_argument("--pdrot_top_rho_target", type=float,
                        default=defaults.get("pdrot_top_rho_target"),
                        help="Target for unweighted top-fraction pdrot [1/m] hinge (None = unused)")
    parser.add_argument("--pdrot_top_fraction", type=float,
                        default=defaults.get("pdrot_top_fraction", 0.01),
                        help="Fraction of largest pdrot excess points used by top-rho hinge")
    parser.add_argument("--pdrot_weight_top_rho", type=float,
                        default=defaults.get("pdrot_weight_top_rho", 0.0),
                        help="Internal weight for top-fraction absolute pdrot hinge")
    parser.add_argument("--pdrot_delta_kappa_a", type=float,
                        default=defaults.get("pdrot_delta_kappa_a", 0.001),
                        help="Regularization scale for umbilic-point handling")
    parser.add_argument("--oops_desc_source_path", type=str,
                        default=defaults.get("oops_desc_source_path"),
                        help="Optional DESC-OOPS source root to prepend to sys.path")
    parser.add_argument("--oops_desc_L", type=int, default=defaults.get("oops_desc_L", 4))
    parser.add_argument("--oops_desc_M", type=int, default=defaults.get("oops_desc_M", 4))
    parser.add_argument("--oops_desc_N", type=int, default=defaults.get("oops_desc_N", 4))
    parser.add_argument("--oops_rhos", type=float, nargs="+",
                        default=defaults.get("oops_rhos", [0.5]),
                        help="rho surfaces for OOPS harmonics residual")
    parser.add_argument("--oops_grid_m", type=int, default=defaults.get("oops_grid_m", 6))
    parser.add_argument("--oops_grid_n", type=int, default=defaults.get("oops_grid_n", 6))
    parser.add_argument("--oops_m_harmonics", type=int,
                        default=defaults.get("oops_m_harmonics", 8))
    parser.add_argument("--oops_n_harmonics", type=int,
                        default=defaults.get("oops_n_harmonics", 8))
    parser.add_argument("--oops_S_list", type=float, nargs="+",
                        default=defaults.get("oops_S_list", [0.3, 0.0]))
    parser.add_argument("--oops_D_list", type=float, nargs="+",
                        default=defaults.get("oops_D_list", [0.0, 0.0]))
    parser.add_argument("--oops_helicity", type=int, nargs=2,
                        default=defaults.get("oops_helicity", [0, 1]))
    parser.add_argument("--oops_scalar_scale", type=float,
                        default=defaults.get("oops_scalar_scale", 1.0),
                        help="Scale for compact OOPS scalar residual sqrt(scalar/scale)")
    parser.add_argument("--local_maxj_s_center", type=float,
                        default=defaults.get("local_maxj_s_center", 0.35),
                        help="Center of radial interval for localized max-J penalty")
    parser.add_argument("--local_maxj_s_width", type=float,
                        default=defaults.get("local_maxj_s_width", 0.35),
                        help="Full radial-interval width for localized max-J penalty")
    parser.add_argument("--local_maxj_lambda_min", type=float,
                        default=defaults.get("local_maxj_lambda_min", 0.0),
                        help="Lower normalized trapped-depth lambda_N for localized max-J penalty")
    parser.add_argument("--local_maxj_lambda_max", type=float,
                        default=defaults.get("local_maxj_lambda_max", 1.0),
                        help="Upper normalized trapped-depth lambda_N for localized max-J penalty")
    parser.add_argument("--local_maxj_exceedance_target", type=float,
                        default=defaults.get("local_maxj_exceedance_target", 0.0),
                        help="Allowed local max-J exceedance before localized penalty")
    parser.add_argument("--local_maxj_scale", type=float,
                        default=defaults.get("local_maxj_scale", 1.0),
                        help="Scale for localized max-J residuals")
    parser.add_argument("--maxj_guard_rel_tol", type=float,
                        default=defaults.get("maxj_guard_rel_tol", 0.0),
                        help="Allowed relative f_maxJ degradation before guard penalty")
    parser.add_argument("--maxj_guard_abs_tol", type=float,
                        default=defaults.get("maxj_guard_abs_tol", 0.0),
                        help="Allowed absolute f_maxJ degradation before guard penalty")
    parser.add_argument("--maxj_guard_scale", type=float,
                        default=defaults.get("maxj_guard_scale", 1e-4),
                        help="Scale for f_maxJ degradation hinge residual")
    parser.add_argument("--oops_guard_rel_tol", type=float,
                        default=defaults.get("oops_guard_rel_tol", 0.0),
                        help="Allowed relative OOPS scalar degradation before guard penalty")
    parser.add_argument("--oops_guard_abs_tol", type=float,
                        default=defaults.get("oops_guard_abs_tol", 0.0),
                        help="Allowed absolute OOPS scalar degradation before guard penalty")
    parser.add_argument("--oops_guard_scale", type=float,
                        default=defaults.get("oops_guard_scale", 1e-3),
                        help="Scale for OOPS scalar degradation hinge residual")
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
    parser.add_argument("--highB_s_min", type=float, default=defaults.get("highB_s_min", 0.25))
    parser.add_argument("--highB_s_max", type=float, default=defaults.get("highB_s_max", 0.90))
    parser.add_argument("--highB_ns", type=int, default=defaults.get("highB_ns", 4))
    parser.add_argument("--highB_threshold_min", type=float, default=defaults.get("highB_threshold_min", 0.75))
    parser.add_argument("--highB_threshold_max", type=float, default=defaults.get("highB_threshold_max", 0.98))
    parser.add_argument("--highB_n_thresholds", type=int, default=defaults.get("highB_n_thresholds", 6))
    parser.add_argument("--highB_ntheta", type=int, default=defaults.get("highB_ntheta", 96))
    parser.add_argument("--highB_nphi", type=int, default=defaults.get("highB_nphi", 96))
    parser.add_argument("--highB_mpol", type=int, default=defaults.get("highB_mpol", 20))
    parser.add_argument("--highB_ntor", type=int, default=defaults.get("highB_ntor", 20))
    parser.add_argument("--highB_target_phi_coverage", type=float,
                        default=defaults.get("highB_target_phi_coverage", 0.85))
    parser.add_argument("--highB_max_phi_gap", type=float,
                        default=defaults.get("highB_max_phi_gap", 0.20))
    parser.add_argument("--hard_gate_mhd", action="store_true",
                        default=defaults.get("hard_gate_mhd", False),
                        help="Add a large penalty for candidates outside MHD gate limits")
    parser.add_argument("--hard_gate_iota_topology", action="store_true",
                        default=defaults.get("hard_gate_iota_topology", False),
                        help="Reject profile, shear, or monotonicity violations")
    parser.add_argument(
        "--hard_dmerc_flux_normalized_min", type=float,
        default=defaults.get("hard_dmerc_flux_normalized_min", 0.0),
        help=("Hard minimum for Phi_edge^2*VMEC.DMerc on the Mercier radial "
              "interval; 0 enforces the convention-independent sign gate"),
    )
    parser.add_argument("--hard_dmerc_neg_max", type=int, default=defaults.get("hard_dmerc_neg_max", 0))
    parser.add_argument("--hard_ballooning_n_max", type=int, default=defaults.get("hard_ballooning_n_max", 0))
    parser.add_argument("--hard_ballooning_lambda_max", type=float,
                        default=defaults.get("hard_ballooning_lambda_max", 0.0))
    parser.add_argument("--hard_beta_max", type=float, default=defaults.get("hard_beta_max", float("inf")))
    parser.add_argument("--hard_fb_rms_max", type=float, default=defaults.get("hard_fb_rms_max", float("inf")))
    parser.add_argument("--hard_rational_crossing", action="store_true",
                        default=defaults.get("hard_rational_crossing", False),
                        help="Use a large residual if any rational_hard_targets crossing is found")
    parser.add_argument("--hard_coil_proxy_bn_max", type=float,
                        default=defaults.get("hard_coil_proxy_bn_max", float("inf")),
                        help="Hard reject if coil proxy Bn_max/|B| exceeds this value")

    parser.add_argument("--abs_step", type=float, default=defaults.get("abs_step", 1e-4),
                        help="Absolute FD step for Jacobian (simsopt default 1e-7)")
    parser.add_argument("--rel_step", type=float, default=defaults.get("rel_step", 0.0),
                        help="Relative FD step for Jacobian")
    parser.add_argument("--perturb", type=float, default=defaults.get("perturb", 0.0),
                        help="Random perturbation amplitude (fraction of |x|) to escape local minima")
    parser.add_argument("--dof_bound_frac", type=float, default=defaults.get("dof_bound_frac", 0.0),
                        help="Max fractional change per DoF (soft bound, 0=disabled)")
    parser.add_argument("--free_iota", dest="free_iota", action="store_true",
                        default=defaults.get("free_iota", True),
                        help="Use NCURR=1 so iota is computed self-consistently (default)")
    parser.add_argument("--prescribe_iota", dest="free_iota", action="store_false",
                        help="Use NCURR=0 and write iota_ax/iota_edge as hard AI constraints")
    parser.add_argument("--zero_current", action="store_true",
                        default=defaults.get("zero_current", False),
                        help="With NCURR=1, explicitly discard the wout current profile")

    parser.add_argument("--backend", choices=["auto", "vmec", "desc"],
                        default=defaults.get("backend", "auto"))
    parser.add_argument("--desc_L", type=int, default=defaults.get("desc_L", 4))
    parser.add_argument("--desc_M", type=int, default=defaults.get("desc_M", 4))
    parser.add_argument("--desc_N", type=int, default=defaults.get("desc_N", 4))
    parser.add_argument("--coil_proxy_offset_fraction", type=float,
                        default=defaults.get("coil_proxy_offset_fraction", 0.35))
    parser.add_argument("--coil_proxy_lambda_regularization", type=float,
                        default=defaults.get("coil_proxy_lambda_regularization", 1e-8))
    parser.add_argument("--coil_proxy_desc_L", type=int,
                        default=defaults.get("coil_proxy_desc_L", 4))
    parser.add_argument("--coil_proxy_desc_M", type=int,
                        default=defaults.get("coil_proxy_desc_M", 4))
    parser.add_argument("--coil_proxy_desc_N", type=int,
                        default=defaults.get("coil_proxy_desc_N", 4))
    parser.add_argument("--coil_proxy_M_Phi", type=int,
                        default=defaults.get("coil_proxy_M_Phi", 4))
    parser.add_argument("--coil_proxy_N_Phi", type=int,
                        default=defaults.get("coil_proxy_N_Phi", 4))
    parser.add_argument("--coil_proxy_source_M", type=int,
                        default=defaults.get("coil_proxy_source_M", 16))
    parser.add_argument("--coil_proxy_source_N", type=int,
                        default=defaults.get("coil_proxy_source_N", 16))
    parser.add_argument("--coil_proxy_eval_M", type=int,
                        default=defaults.get("coil_proxy_eval_M", 16))
    parser.add_argument("--coil_proxy_eval_N", type=int,
                        default=defaults.get("coil_proxy_eval_N", 16))
    parser.add_argument("--coil_proxy_current_helicity", type=int, nargs=2,
                        default=defaults.get("coil_proxy_current_helicity", [1, 0]))
    parser.add_argument("--coil_proxy_regularization_type", choices=["regcoil", "simple"],
                        default=defaults.get("coil_proxy_regularization_type", "regcoil"))
    parser.add_argument("--coil_proxy_vacuum", action="store_true",
                        default=defaults.get("coil_proxy_vacuum", False))
    parser.add_argument("--coil_proxy_chunk_size", type=int,
                        default=defaults.get("coil_proxy_chunk_size"))
    parser.add_argument("--coil_proxy_verbose", type=int,
                        default=defaults.get("coil_proxy_verbose", 0))
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


def _reject_legacy_mercier_config(config, parser):
    legacy = {
        "mercier_margin_target": "mercier_flux_normalized_margin_target",
        "hard_dmerc_min": "hard_dmerc_flux_normalized_min",
        "dmerc_b0_ref": None,
        "dmerc_reference_b0_T": None,
        "dmerc_scaled_min": None,
    }
    present = sorted(key for key in legacy if key in config)
    if not present:
        return
    migration = []
    for key in present:
        replacement = legacy[key]
        if replacement:
            migration.append(f"{key} -> {replacement} (recalibrate the value)")
        else:
            migration.append(f"remove {key}; B0-based DMerc scaling is legacy-only")
    parser.error(
        "ambiguous legacy Mercier configuration detected: "
        + "; ".join(migration)
        + ". New optimization and gate decisions use Phi_edge^2*VMEC.DMerc."
    )


def _jsonify(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _validate_optional_backends(args, parser):
    if float(args.w_oops) <= 0.0 and float(args.w_oops_guard) <= 0.0:
        return
    from ..objectives.oops_residual import oops_backend_status

    available, reason = oops_backend_status(args.oops_desc_source_path)
    if not available:
        parser.error(
            "OOPS objective requested, but the required DESC-OOPS API is "
            f"unavailable: {reason}. Install/select the DESC-OOPS source with "
            "--oops_desc_source_path, or set w_oops=w_oops_guard=0."
        )


def _write_resolved_parameters(args, config):
    run_name = args.run_name or config.get("name")
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source = args.nc_file or getattr(args, "vmec_input_file", None)
        stem = Path(source).stem if source else "squid"
        run_name = f"{stamp}_{args.mode}_{stem}"
    run_dir = resolve_run_dir(args.run_dir, run_name)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_run_dir:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}. Use a new run_dir; "
            "pass --overwrite_run_dir only for an intentional destructive rerun."
        )
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--input_parameter", type=str)
    pre.add_argument("--mode", choices=sorted(MODE_PRESETS), default="core")
    pre_args, _ = pre.parse_known_args(argv)

    config = _load_config(pre_args.input_parameter)
    cli_mode_set = any(
        arg == "--mode" or arg.startswith("--mode=")
        for arg in argv
    )
    mode = pre_args.mode if cli_mode_set else config.get("mode", pre_args.mode)
    defaults = MODE_PRESETS.get(mode, MODE_PRESETS["core"]).copy()
    defaults.update(config)
    defaults["mode"] = mode
    defaults["input_parameter"] = pre_args.input_parameter

    parser = _build_parser(defaults)
    args = parser.parse_args(argv)
    jax_device = initialize_requested_jax_backend()
    if jax_device is not None:
        print(f"  JAX CUDA device initialized early: {jax_device}")
    _reject_legacy_mercier_config(config, parser)
    _validate_optional_backends(args, parser)
    if not args.nc_file and not getattr(args, "vmec_input_file", None):
        parser.error("--nc_file or --vmec_input_file is required unless provided by --input_parameter")
    resolved_path = _write_resolved_parameters(args, config)

    print(f"\n{'=' * 60}")
    print("SQuID Optimisation  --  unified squid package")
    print(f"{'=' * 60}")
    print(f"  Input : {args.nc_file or args.vmec_input_file}")
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
        from ..backends.desc_backend import run_desc
        result = run_desc(args)
    else:
        from ..backends.vmec_backend import run_vmec
        result = run_vmec(args)

    if isinstance(result, dict) and not result.get("accepted", True):
        print(f"\n{'=' * 60}")
        print("Run completed without an accepted optimization result.")
        print(f"State: {result.get('state', 'unknown')}")
        print(f"{'=' * 60}\n")
        return 2

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}\n")
    return 0
