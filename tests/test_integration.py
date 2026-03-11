#!/usr/bin/env python3
"""
Integration tests for the squid package.

Compares outputs of the new unified squid package against the
original standalone scripts (max_J_evaluation.py, simsopt_optimize_v3.py)
to verify numerical equivalence.

Run:
    cd /home/guozx/Squid
    python -m pytest tests/test_integration.py -v

or directly:
    python tests/test_integration.py
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Reference wout file — change this to your own if needed
REFERENCE_WOUT = "/home/guozx/constellaration/nc_files/wout_DATJ5FCamJ3dNpdKmj8QsfL.nc"
ALT_WOUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "_wout_opt_tmp.nc",
)


def _find_wout():
    for path in [REFERENCE_WOUT, ALT_WOUT]:
        if os.path.exists(path):
            return path
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
        self.assertIn((1, 0), coeffs)
        rbc00, zbs00 = coeffs[(0, 0)]
        self.assertAlmostEqual(rbc00, 1.5)


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


class TestNumericalEquivalence(unittest.TestCase):
    """
    Compare outputs of the new squid package against the original
    max_J_evaluation.py to verify numerical equivalence.
    """

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

        cls.old_module_available = False
        if cls.wout_path is not None:
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
                from max_J_evaluation import evaluate_squid as old_evaluate_squid
                cls.old_evaluate_squid = old_evaluate_squid
                cls.old_module_available = True
            except ImportError:
                pass

    @_skip_if_no_wout
    def test_evaluate_squid_equivalence(self):
        """Verify that squid.evaluation matches max_J_evaluation outputs."""
        if not self.old_module_available:
            self.skipTest("Original max_J_evaluation.py not importable")

        from squid.evaluation.evaluate import evaluate_squid

        s_vals = np.array([0.25, 0.5, 0.75])
        alphas = np.linspace(0, 2 * np.pi, 4, endpoint=False)

        new_info = evaluate_squid(
            self.vmec, s_vals=s_vals,
            num_alpha=4, num_pitch=10,
            verbose=False,
        )

        old_info = self.old_evaluate_squid(
            self.vmec, s_vals, alphas,
            num_pitch=10, T_J=-0.06, mboz=8, nboz=8,
        )

        rtol = 1e-4
        np.testing.assert_allclose(
            new_info["f_maxJ"], old_info["f_maxJ"], rtol=rtol,
            err_msg="f_maxJ mismatch",
        )
        np.testing.assert_allclose(
            new_info["f_QI"], old_info["f_QI"], rtol=rtol,
            err_msg="f_QI mismatch",
        )
        np.testing.assert_allclose(
            new_info["mirror_ratio"], old_info["mirror_ratio"], rtol=rtol,
            err_msg="mirror_ratio mismatch",
        )


if __name__ == "__main__":
    unittest.main()
