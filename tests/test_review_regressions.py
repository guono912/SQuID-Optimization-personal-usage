from types import SimpleNamespace
import sys

import numpy as np
import pytest

from squid.cli.optimize import _build_parser


@pytest.mark.parametrize("flags,hard_count", [
    ([], 1),
    (["--rational_targets", "4/5", "--rational_hard_targets", "4/5"], 0),
    (["--rational_targets", "3/4", "--rational_hard_targets", "-4/5"], 1),
])
def test_negative_iota_guard(flags, hard_count):
    pytest.importorskip("simsopt")
    from squid.backends.vmec_backend import _rational_placement_residuals

    args = _build_parser().parse_args(["--hard_rational_crossing", *flags])
    vmec = SimpleNamespace(wout=SimpleNamespace(iotaf=np.array([-.85, -.75])))
    residuals, info = _rational_placement_residuals(vmec, args)
    assert len(residuals) == len(args.rational_targets) + 1
    assert info["hard_crossing_count"] == hard_count
    assert bool(residuals[-1]) == bool(hard_count)


def test_ripple_sources_keep_separate_peaks():
    from squid.cli.diagnose import _main_issue_summary

    info = {"s_vals": [], "interval_centers": []}
    rows = [{"s": .1, "value": 2, "source": "D11"},
            {"s": .2, "value": .1, "source": "DESC"},
            {"s": .3, "value": .01, "source": "DESC"}]
    notes = _main_issue_summary(info, rows)
    assert notes == ["D11: largest sampled value at s=0.10",
                     "DESC: largest sampled value at s=0.20"]


def test_neort_result_reaches_transport_report(tmp_path, monkeypatch):
    pytest.importorskip("simsopt")
    from pathlib import Path
    import subprocess
    from simsopt.mhd import boozer
    from squid.cli.diagnose import _evaluate_effective_ripple
    from squid.utils import nc_to_neort

    monkeypatch.setitem(sys.modules, "desc.vmec", None)
    fake = SimpleNamespace(register=lambda s: None, run=lambda: None,
                           write_boozmn=lambda path: Path(path).touch())
    monkeypatch.setattr(boozer, "Boozer", lambda *args, **kw: fake)
    monkeypatch.setattr(nc_to_neort, "convert_boozmn_to_neort",
                        lambda *args, **kw: ("in_file", .1, 1, 4))
    monkeypatch.setenv("NEORT_EXECUTABLE", sys.executable)

    def solver(cmd, **kwargs):
        (Path(kwargs["cwd"]) / "squid_diag.out").write_text("0 0 0 0 0.02")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", solver)
    vmec = SimpleNamespace(wout=SimpleNamespace(phi=[1], Aminor_p=1))
    value, source = _evaluate_effective_ripple(vmec, .5, "seed.nc", tmp_path)
    assert value == .02 and source.startswith("NEO-RT D11")
