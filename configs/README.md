# Optimization config templates

Use JSON configs for reproducible optimization:

```bash
PY=/home/guozx/fusion_env/bin/python
$PY scripts/opt/optimize.py --input_parameter <reviewed_config.json>
```

`core.json`, `core_r2_assist.json`, `maxj_repair.json`,
`edge_bal_repair.json`, and `edge_bal_direct.json` illustrate current presets.
Copy a template into the active campaign and review every path, target, weight,
resolution, and guard before running it.

Files named `stage*` are historical campaign configs. They are reproducibility
evidence, not defaults for YF_0 or YF_1. Their weights and thresholds may use
old grids or objectives.

Resolution order is:

1. mode preset;
2. JSON values;
3. command-line overrides.

The optimizer writes the effective values to
`<run_dir>/input_parameter.resolved.json`. Preserve that file with the run.
