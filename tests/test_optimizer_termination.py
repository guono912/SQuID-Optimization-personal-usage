import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from squid.cli.optimize import _write_resolved_parameters, main


def test_overwrite_preserves_old_result_and_seed(tmp_path):
    candidate = tmp_path / "input.squid_optimized"
    candidate.write_text("previous equilibrium")
    (tmp_path / "run_status.json").write_text('{"accepted": true}')
    args = Namespace(run_name="reuse", run_dir=str(tmp_path), overwrite_run_dir=True,
                     vmec_input_file=str(candidate), nc_file=None, input_parameter=None)
    _write_resolved_parameters(args, {})
    assert not candidate.exists()
    assert not (tmp_path / "run_status.json").exists()
    assert Path(args.vmec_input_file).read_text() == "previous equilibrium"


@pytest.mark.parametrize("backend", ["vmec", "desc"])
def test_one_evaluation_cannot_publish_an_optimized_candidate(tmp_path, backend):
    source = os.environ.get("SQUID_TEST_WOUT")
    if not source:
        pytest.skip("set SQUID_TEST_WOUT to run solver-backed termination checks")
    run = tmp_path / backend
    argv = ["--backend", backend, "--nc_file", str(Path(source).resolve()),
            "--run_dir", str(run), "--maxiter", "3", "--max_total_evals", "1",
            "--max_dofs", "1", "--num_surfaces", "3", "--num_alpha", "3",
            "--num_pitch", "8", "--desc_L", "4", "--desc_M", "4", "--desc_N", "2"]
    if backend == "vmec":
        # Select the zero-current reconstruction explicitly for this budget test.
        argv.append("--zero_current")
    assert main(argv) == 2
    status = json.loads((run / "run_status.json").read_text())
    assert status["evaluations"] == 1
    assert status["state"] in ("evaluation_cap", "evaluation_cap_recovered_start")
    assert not status["accepted"]
    assert not (run / "input.squid_optimized").exists()
    assert not (run / "wout_desc_optimized.nc").exists()
