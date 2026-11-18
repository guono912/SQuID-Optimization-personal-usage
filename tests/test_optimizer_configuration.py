import json

import pytest

from squid.cli import optimize


@pytest.mark.parametrize("config", [
    {"w_qii": 9}, {"weights": {"w_qii": 9}}, {"weigths": {"w_qi": 9}},
    {"backend": "vmecc"},
])
def test_invalid_config_fails_before_creating_run(tmp_path, capsys, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        optimize.main(["--input_parameter", str(path), "--nc_file", "seed.nc",
                       "--run_dir", str(out)])
    assert exc.value.code == 2
    assert ("unknown configuration keys" if "backend" not in config else
            "invalid backend") in capsys.readouterr().err
    assert not out.exists()


def test_alias_metadata_and_existing_pdrot_config(tmp_path, monkeypatch):
    config = {"name": "test", "notes": {"purpose": "regression"},
              "r2": {"nphi": 51}, "weights": {"w_qi": 9},
              "targets": {"pdrot_mean_rho_target": 1.8, "pdrot_weight_mean_rho": 4}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "run"
    captured = []
    from squid.backends import vmec_backend
    monkeypatch.setattr(vmec_backend, "run_vmec", captured.append)
    monkeypatch.setattr(optimize, "HAS_VMEC", True)
    assert optimize.main(["--input_parameter", str(path), "--backend", "vmec",
                          "--nc_file", "seed.nc", "--run_dir", str(out),
                          "--w_qi", "7"]) == 0
    args, = captured
    assert (args.qi_r2_nphi, args.w_qi, args.pdrot_weight_mean_rho) == (51, 7, 4)
    resolved = json.loads((out / "input_parameter.resolved.json").read_text())
    assert resolved["_source_config"]["notes"] == config["notes"]


@pytest.mark.parametrize(("backend", "args"), [
    ("vmec", ["--w_highB_topology", "1"]), ("desc", ["--w_pdrot", "1"]),
    ("desc", ["--hard_rational_crossing"]), ("desc", ["--hard_coil_proxy_bn_max", "0"]),
])
def test_active_backend_specific_controls(backend, args, capsys):
    parser = optimize._build_parser()
    with pytest.raises(SystemExit):
        optimize._check_backend_controls(backend, parser.parse_args(args), parser)
    assert args[0][2:] in capsys.readouterr().err
