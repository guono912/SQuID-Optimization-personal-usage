import numpy as np
from netCDF4 import Dataset

from squid.utils.vmec_input import wout_to_input


def test_rescale_preserves_reduced_modes_pressure_and_signed_flux(tmp_path):
    wout = tmp_path / "wout_seed.nc"
    with Dataset(wout, "w") as ds:
        ds.createDimension("radius", 3)
        ds.createDimension("mode", 2)
        for name, value in {"nfp": 2, "mpol": 2, "ntor": 1, "ns": 3,
                            "Rmajor_p": 2, "ctor": 700}.items():
            ds.createVariable(name, "f8")[...] = value
        for key, dim, values in [
            ("phi", "radius", [0, -0.1, -0.2]), ("iotaf", "radius", [.3, .4, .5]),
            ("presf", "radius", [100, 50, 0]), ("xm", "mode", [0, 1]),
            ("xn", "mode", [0, -2]), ("ac", "mode", [1, 0]),
        ]:
            ds.createVariable(key, "f8", (dim,))[:] = values
        for key, edge in [("rmnc", [2, .1]), ("zmns", [0, .2])]:
            ds.createVariable(key, "f8", ("radius", "mode"))[:] = np.tile(edge, (3, 1))

    target = tmp_path / "input.scaled"
    wout_to_input(wout, target, nfp_out=4, target_rmajor=4, zero_current=True)
    fields = dict(line.strip().split(" = ") for line in target.read_text().splitlines()
                  if " = " in line)
    assert int(fields["NFP"]) == 4
    assert float(fields["PHIEDGE"]) == -0.8
    assert float(fields["RBC(-1,1)"]) == .2
    assert float(fields["ZBS(-1,1)"]) == .4
    assert int(fields["NCURR"]) == 1 and float(fields["AC(0)"]) == 0
    assert "CURTOR" not in fields
    am = [float(fields[f"AM({i})"]) for i in range(3)]
    np.testing.assert_allclose(float(fields["PRES_SCALE"]) *
                               np.polynomial.polynomial.polyval([0, .5, 1], am),
                               [100, 50, 0], atol=1e-10)
