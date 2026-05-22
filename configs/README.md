# SQuID optimisation parameter files

Use these JSON files as the normal entry point for optimisation runs:

```bash
python scripts/optimize.py --input_parameter configs/core.json
```

Command-line arguments override JSON values. The optimiser writes the final
effective arguments to `runs/<run-name>/input_parameter.resolved.json`.

## Presets

- `core.json`: baseline fixed-boundary VMEC optimisation with simple QI, max-J,
  B_min radial growth, aspect ratio, and regularisation.
- `core_r2_assist.json`: same baseline with a light R2 QI assist. Use this
  after basic QI/max-J behaviour is sane.
- `maxj_repair.json`: raises max-J and B_min weights for cases where QI is
  acceptable but trapped-particle/max-J diagnostics are poor.

If a JSON file omits a weight, `--mode` supplies the preset default first, then
the JSON file and CLI overrides are applied.
