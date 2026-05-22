#!/usr/bin/env python3
"""
Integration tests for the squid package.

Fast tests cover core numerical helpers without VMEC. Heavy VMEC-backed
checks are opt-in via SQUID_TEST_WOUT.

Run:
    cd /home/guozx/SQuID
    python -m pytest tests/test_integration.py -v

or directly:
    python tests/test_integration.py
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Heavy VMEC tests are opt-in. Some local wout files are generated artefacts
# whose VMEC compatibility depends on the installed Fortran extension, and a
# failed VMEC run can terminate the process before Python can skip cleanly.
REFERENCE_WOUT = os.environ.get("SQUID_TEST_WOUT")


def _find_wout():
    if REFERENCE_WOUT and os.path.exists(REFERENCE_WOUT):
        return REFERENCE_WOUT
    return None


def _skip_if_no_wout(func):
    """Decorator to skip test if no wout file is available."""
    def wrapper(self):
        if self.wout_path is None:
            self.skipTest("No reference wout file found")
        return func(self)
    return wrapper


class TestCoreModules(unittest.TestCase):
    """Test core subpackage without VMEC."""

    def test_bounce_simple(self):
        """find_bounce_points on a synthetic well."""
        from squid.core.bounce import find_bounce_points
        phi = np.linspace(0, 2 * np.pi, 100)
        B = 1.0 + 0.3 * np.cos(phi)
        B_star = 1.1
        p1, p2, _, _ = find_bounce_points(phi, B, B_star, np.max(B), np.min(B))
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertGreater(p2, p1)

    def test_squash_stretch_simple(self):
        """squash_and_stretch_simple produces monotone arms."""
        from squid.core.squash_stretch import squash_and_stretch_simple
        zeta = np.linspace(0, 2 * np.pi, 200)
        # A well shape: high at endpoints, low in the middle
        B_I = 1.0 + 0.3 * np.cos(zeta) + 0.05 * np.cos(3 * zeta)
        B_C = squash_and_stretch_simple(zeta, B_I, 0.7, 1.3)
        # B_C should be monotone on each arm
        idx_min = np.argmin(B_C)
        left = B_C[:idx_min + 1]
        right = B_C[idx_min:]
        self.assertTrue(np.all(np.diff(left) <= 1e-12), "left arm not decreasing")
        self.assertTrue(np.all(np.diff(right) >= -1e-12), "right arm not increasing")

    def test_squash_stretch_r2(self):
        """squash_and_stretch_r2 on normalised input."""
        from squid.core.squash_stretch import squash_and_stretch_r2
        phi = np.linspace(0, 1, 100)
        B_norm = 0.5 * (1 - np.cos(2 * np.pi * phi))
        B_C = squash_and_stretch_r2(B_norm)
        self.assertEqual(len(B_C), len(B_norm))
        self.assertGreaterEqual(np.min(B_C), -0.1)
        self.assertLessEqual(np.max(B_C), 1.1)

    def test_hinge_loss(self):
        from squid.objectives.penalties import hinge_loss
        self.assertAlmostEqual(hinge_loss(0.5, 1.0), 0.0)
        self.assertAlmostEqual(hinge_loss(1.5, 1.0), 0.25)

    def test_generate_initial(self):
        from squid.utils.generate_initial import generate_boundary
        coeffs = generate_boundary(aspect=10, elongation=6, mirror=0.25, nfp=4)
        self.assertIn((0, 0), coeffs)
        self.assertIn((0, 1), coeffs)
        self.assertIn((2, 0), coeffs)
        rbc00, zbs00 = coeffs[(0, 0)]
        self.assertAlmostEqual(rbc00, 1.0)
        self.assertAlmostEqual(zbs00, 0.0)

    def test_bmin_slope_penalty(self):
        from squid.objectives.penalties import bmin_slope_penalty
        s_vals = np.array([0.2, 0.5, 0.8])
        good_bmins = np.array([1.00, 1.03, 1.06])
        bad_bmins = np.array([1.00, 0.99, 0.98])
        self.assertAlmostEqual(bmin_slope_penalty(s_vals, good_bmins), 0.0)
        self.assertGreater(bmin_slope_penalty(s_vals, bad_bmins), 0.0)

    def test_optimize_config_mode_defaults(self):
        from scripts.optimize import MODE_PRESETS, _build_parser, _flatten_config
        config = _flatten_config({
            "mode": "maxj_repair",
            "nc_file": "artifacts/legacy_root/wout_squid_optimized.nc",
            "numerics": {"maxiter": 2},
            "r2": {"nphi": 77},
        })
        defaults = MODE_PRESETS[config["mode"]].copy()
        defaults.update(config)
        parser = _build_parser(defaults)
        args = parser.parse_args(["--maxiter", "0", "--w_maxj", "9.0"])

        self.assertEqual(args.mode, "maxj_repair")
        self.assertEqual(args.maxiter, 0)
        self.assertEqual(args.qi_r2_nphi, 77)
        self.assertAlmostEqual(args.w_qi, MODE_PRESETS["maxj_repair"]["w_qi"])
        self.assertAlmostEqual(args.w_maxj, 9.0)

    def test_optimize_cli_mode_can_override_config_mode(self):
        from scripts.optimize import MODE_PRESETS, _build_parser, _flatten_config
        config = _flatten_config({
            "mode": "core",
            "nc_file": "artifacts/legacy_root/wout_squid_optimized.nc",
        })
        cli_mode = "maxj_repair"
        defaults = MODE_PRESETS[cli_mode].copy()
        defaults.update(config)
        defaults["mode"] = cli_mode
        parser = _build_parser(defaults)
        args = parser.parse_args(["--mode", cli_mode])

        self.assertEqual(args.mode, "maxj_repair")
        self.assertAlmostEqual(args.w_maxj, MODE_PRESETS["maxj_repair"]["w_maxj"])

    def test_iota_rational_scan_reports_nearest_low_order(self):
        from scripts.diagnose import _scan_iota_rationals

        class Wout:
            iotaf = np.array([-0.42, -0.45, -0.50, -0.56])

        class FakeVmec:
            wout = Wout()

        scan = _scan_iota_rationals(
            FakeVmec(),
            max_denominator=4,
            warn_distance=0.02,
            s_min=0.0,
            s_max=1.0,
        )

        self.assertTrue(scan["available"])
        self.assertEqual(scan["nearest_low_order"]["rational"], "-1/2")
        self.assertLessEqual(scan["nearest_low_order"]["distance"], 1e-12)
        self.assertGreaterEqual(len(scan["near_low_order"]), 1)


class TestWithVMEC(unittest.TestCase):
    """Tests requiring a real VMEC wout file and simsopt."""

    @classmethod
    def setUpClass(cls):
        cls.wout_path = _find_wout()
        if cls.wout_path is not None:
            try:
                from simsopt.mhd import Vmec
                cls.vmec = Vmec(cls.wout_path)
                cls.vmec.run()
            except Exception:
                cls.wout_path = None

    @_skip_if_no_wout
    def test_run_boozer(self):
        from squid.core.boozer_utils import run_boozer
        bx, data = run_boozer(self.vmec, [0.5])
        self.assertEqual(len(data), 1)
        d = data[0]
        self.assertIn("B_min", d)
        self.assertIn("B_max", d)
        self.assertGreater(d["B_max"], d["B_min"])

    @_skip_if_no_wout
    def test_evaluate_squid(self):
        from squid.evaluation.evaluate import evaluate_squid
        info = evaluate_squid(
            self.vmec,
            s_vals=np.array([0.25, 0.5, 0.75]),
            num_alpha=4,
            num_pitch=10,
            verbose=False,
        )
        self.assertIn("f_maxJ", info)
        self.assertIn("f_QI", info)
        self.assertIn("mirror_ratio", info)
        self.assertGreater(info["mirror_ratio"], 0.0)

    @_skip_if_no_wout
    def test_maxj_residual_class(self):
        from squid.objectives.maxj_residual import MaxJResidual
        mj = MaxJResidual(
            self.vmec,
            s_vals=np.array([0.3, 0.6]),
            num_alpha=4,
            num_pitch=10,
        )
        r = mj.residuals()
        self.assertEqual(r.shape, (1,))
        self.assertGreaterEqual(r[0], 0.0)

    @_skip_if_no_wout
    def test_itg_residual_vacuum(self):
        from squid.objectives.itg_residual import ITGResidual
        itg = ITGResidual(
            self.vmec,
            snorms=[0.3],
            method="vacuum_dBds",
        )
        t = itg.total()
        self.assertIsInstance(t, float)

    @_skip_if_no_wout
    def test_mirror_ratio_penalty(self):
        from squid.objectives.penalties import MirrorRatioPenalty
        pen = MirrorRatioPenalty(self.vmec, target=0.10)
        r = pen.residuals()
        self.assertEqual(r.shape, (1,))

    @_skip_if_no_wout
    def test_iota_profile_penalty(self):
        from squid.objectives.penalties import IotaProfilePenalty
        pen = IotaProfilePenalty(self.vmec)
        r = pen.residuals()
        self.assertEqual(r.shape, (2,))
        # at auto-detected targets, residuals should be near zero
        np.testing.assert_allclose(r, [0, 0], atol=0.01)

    @_skip_if_no_wout
    def test_tikhonov(self):
        from squid.objectives.penalties import TikhonovRegularization
        reg = TikhonovRegularization(self.vmec)
        t = reg.total()
        self.assertAlmostEqual(t, 0.0, places=10)

    @_skip_if_no_wout
    def test_field_line_boozer(self):
        from squid.core.boozer_utils import run_boozer
        from squid.core.fieldline import extract_field_line_boozer
        _, data = run_boozer(self.vmec, [0.5])
        zeta, B_I = extract_field_line_boozer(data[0], alpha=0.0, npts=200)
        self.assertEqual(len(zeta), 200)
        self.assertEqual(len(B_I), 200)
        self.assertTrue(np.all(np.isfinite(B_I)))


if __name__ == "__main__":
    unittest.main()
