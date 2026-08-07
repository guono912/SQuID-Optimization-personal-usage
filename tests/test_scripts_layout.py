#!/usr/bin/env python3
"""
Layout and compatibility smoke tests for the scripts/ reorganization.

Covers:
- squid/utils/run_paths.py unit behaviour (stdlib-only, runs everywhere);
- scripts/ group layout: new paths and old flat-path wrappers import and
  re-export the same names;
- CLI --help smoke tests via subprocess, skipped when the fusion
  dependencies (desc/simsopt/netCDF4/scipy) are not installed in the
  current environment.

Run:
    python3 -m unittest tests.test_scripts_layout -v
    (or) python -m pytest -q tests/test_scripts_layout.py
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# DESC imports JAX during the dependency probes below. Select the portable
# backend before those probes so the test itself does not depend on GPU state.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _have(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAVE_SCIPY = _have("scipy")
HAVE_NETCDF4 = _have("netCDF4")
HAVE_DESC = _have("desc")
HAVE_SIMSOPT = _have("simsopt")


def run_cli(script_path, *args, expect_exit=0):
    """Run a script as `python <script> <args>` from the repo root."""
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if expect_exit is not None:
        assert proc.returncode == expect_exit, (
            f"{script_path} {args} -> exit {proc.returncode}: "
            f"{proc.stderr[-800:]}"
        )
    return proc


class TestRunPaths(unittest.TestCase):
    """Unit behaviour of the unified run-path helper (stdlib only)."""

    def test_resolve_explicit(self):
        from squid.utils.run_paths import resolve_run_dir
        self.assertEqual(resolve_run_dir("/tmp/foo").name, "foo")


    def test_resolve_requires_explicit_dir_or_name(self):
        from squid.utils.run_paths import resolve_run_dir
        with self.assertRaises(ValueError):
            resolve_run_dir(None, None)

    def test_resolve_run_name_fallback(self):
        from squid.utils.run_paths import resolve_run_dir
        path = resolve_run_dir(None, "stage1")
        self.assertEqual(path.parts[-2:], ("runs", "stage1"))
        self.assertTrue(path.is_absolute())

    def test_safe_output_path_rejects_escape(self):
        from squid.utils.run_paths import safe_output_path
        with self.assertRaises(ValueError):
            safe_output_path("/tmp/run", "../escape.nc")
        self.assertEqual(safe_output_path("/tmp/run", "ok.nc").name, "ok.nc")

    def test_manifest_and_sha256(self):
        from squid.utils.run_paths import sha256_file, write_manifest
        with tempfile.TemporaryDirectory() as d:
            manifest = write_manifest(d, command="test-cmd")
            import json
            data = json.loads(manifest.read_text())
            self.assertEqual(data["command"], "test-cmd")
            self.assertEqual(data["run_dir"], str(Path(d).resolve()))
            payload = Path(d) / "a.bin"
            payload.write_bytes(b"hello")
            import hashlib
            self.assertEqual(sha256_file(payload), hashlib.sha256(b"hello").hexdigest())

    def test_jax_runtime_cpu_selection_stays_lazy(self):
        from squid.utils.jax_runtime import initialize_requested_jax_backend

        previous = os.environ.get("JAX_PLATFORMS")
        os.environ["JAX_PLATFORMS"] = "cpu"
        try:
            self.assertIsNone(initialize_requested_jax_backend())
        finally:
            if previous is None:
                os.environ.pop("JAX_PLATFORMS", None)
            else:
                os.environ["JAX_PLATFORMS"] = previous

    def test_vmec_run_directory_context_restores_cwd(self):
        from squid.backends.vmec_backend import _run_in_directory

        original = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            with _run_in_directory(d):
                self.assertEqual(Path.cwd(), Path(d))
            self.assertEqual(Path.cwd(), original)


@unittest.skipUnless(HAVE_DESC and HAVE_SIMSOPT, "viz dependencies unavailable")
class TestReportGeometry(unittest.TestCase):
    def test_surface_area_prefers_quadrature_and_has_correct_fallback(self):
        from squid.cli.viz import _surface_area_for_report

        wout = SimpleNamespace(Rmajor_p=2.0, Aminor_p=0.5)
        resolved, source = _surface_area_for_report(
            wout, {"surface_area_m2_simsopt": 12.5}
        )
        self.assertEqual(resolved, 12.5)
        self.assertEqual(source, "simsopt_lcfs_quadrature")

        fallback, source = _surface_area_for_report(wout, {})
        self.assertAlmostEqual(fallback, 4.0 * math.pi**2)
        self.assertEqual(source, "circular_torus_4pi2_Ra_fallback")

    def test_field_scale_uses_flux_magnitude(self):
        from squid.cli.viz import _field_scale_from_flux

        expected = 0.2 / (math.pi * 0.1**2)
        self.assertAlmostEqual(_field_scale_from_flux(-0.2, 0.1), expected)

    def test_iss04_field_scale_prefers_bvco_over_flux_fallback(self):
        from squid.cli.viz import _iss04_field_scale

        wout = SimpleNamespace(
            bvco=np.array([0.0, -0.24]),
            Rmajor_p=1.2,
            phi=np.array([0.0, 0.01]),
            Aminor_p=0.1,
        )
        value, method = _iss04_field_scale(wout)
        self.assertAlmostEqual(value, 0.2)
        self.assertEqual(method, "abs_bvco_first_nonzero_over_Rmajor")

    def test_nfp_rationals_support_negative_iota(self):
        from squid.cli.viz import _nfp_dangerous_rationals

        rationals = _nfp_dangerous_rationals(2, -0.37, -0.32, m_max=16)
        labels = {(k * 2, m) for _, m, k in rationals}
        self.assertIn((-1 * 2, 6), labels)

    def test_axis_field_summary_uses_vmec_fourier_modes(self):
        from squid.cli.viz import _axis_field_summary

        wout = SimpleNamespace(
            bmnc=np.array([[2.0, 0.1]]),
            xm_nyq=np.array([0, 0]),
            xn_nyq=np.array([0, 2]),
            b0=9.0,
        )
        mean, minimum, maximum = _axis_field_summary(wout, fallback=8.0)
        self.assertAlmostEqual(mean, 2.0, places=12)
        self.assertAlmostEqual(minimum, 1.9, places=12)
        self.assertAlmostEqual(maximum, 2.1, places=12)

        wout.bmnc = wout.bmnc.T
        transposed = _axis_field_summary(wout, fallback=8.0)
        np.testing.assert_allclose(transposed, (mean, minimum, maximum))


class TestScriptsLayout(unittest.TestCase):
    """Grouped implementations exist and old flat paths re-export them."""

    def test_rescale_and_gate_wrapper_reexports(self):
        from scripts.rescale_and_gate import gate, rescale
        from scripts.transform.rescale_and_gate import gate as new_gate
        self.assertIs(gate, new_gate)
        self.assertTrue(callable(rescale))

    def test_nc_to_neort_wrapper_reexports_library(self):
        if not HAVE_SCIPY:
            self.skipTest("scipy not installed")
        import scripts.nc_to_neort
        from squid.utils import nc_to_neort
        self.assertIs(
            scripts.nc_to_neort.convert_boozmn_to_neort,
            nc_to_neort.convert_boozmn_to_neort,
        )

    def test_coil_contour_metrics_uses_squid_coil_metrics(self):
        if not HAVE_SCIPY:
            self.skipTest("scipy not installed")
        import scripts.util.coil_contour_metrics
        from squid.evaluation import coil_metrics
        self.assertIs(
            scripts.util.coil_contour_metrics.make_winding_surface,
            coil_metrics.make_winding_surface,
        )

    def test_diag_group_importable(self):
        if not HAVE_NETCDF4:
            self.skipTest("netCDF4 not installed")
        from scripts.diag import check_mercier_normalization
        self.assertTrue(callable(check_mercier_normalization.main))

    def test_util_group_importable(self):
        if not HAVE_DESC:
            self.skipTest("desc not installed")
        from scripts.util import compare_desc_profiles, coil_contour_metrics
        self.assertTrue(callable(compare_desc_profiles.main))
        self.assertTrue(callable(coil_contour_metrics.main))


class TestIotaRationalDiagnostics(unittest.TestCase):
    def test_signed_scan_reports_zero_crossing_and_both_signs(self):
        from squid.diagnostics.iota_rationals import _scan_iota_rationals

        class Wout:
            iotaf = [-0.40, -0.20, 0.20, 0.40]

        class Vmec:
            wout = Wout()

        result = _scan_iota_rationals(
            Vmec(), max_denominator=4, warn_distance=0.01,
            s_min=0.0, s_max=1.0,
        )
        self.assertTrue(result["sign_flip"])
        self.assertTrue(result["zero_crossings"])
        labels = {item["rational"] for item in result["crossings"]}
        self.assertIn("-1/4", labels)
        self.assertIn("1/4", labels)


class TestCliHelpSmoke(unittest.TestCase):
    """--help must start from both old and new paths without artifacts."""

    SCRIPTS = [
        "rescale_and_gate.py",
    ]
    PAIRS = [
        ("scripts/rescale_and_gate.py", "scripts/transform/rescale_and_gate.py"),
        ("scripts/diagnose.py", "scripts/diag/diagnose.py"),
        ("scripts/nc_to_neort.py", "scripts/util/nc_to_neort.py"),
    ]

    def _assert_no_root_artifacts(self):
        patterns = (
            "wout_*.nc",
            "input.*",
            "threed1.*",
            "in_file",
            "boozmn_squid_diag.nc",
        )
        for pattern in patterns:
            for hit in REPO_ROOT.glob(pattern):
                self.fail(f"smoke test left artifact at repo root: {hit}")

    def test_help_old_and_new_paths(self):
        for old, new in self.PAIRS:
            run_cli(REPO_ROOT / old, "--help")
            run_cli(REPO_ROOT / new, "--help")
        self._assert_no_root_artifacts()


if __name__ == "__main__":
    unittest.main()
