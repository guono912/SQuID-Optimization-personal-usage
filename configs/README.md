# Optimization config templates

Use JSON configs for reproducible optimization:

```bash
PY=python
$PY scripts/opt/optimize.py --input_parameter <reviewed_config.json>
```

`core.json`, `core_r2_assist.json`, and `maxj_repair.json` illustrate current presets.
The archived `edge_bal_*.json` examples use a legacy Mercier margin.
Recalibrate that margin before using them with flux-normalized Mercier values.
Copy a template into the active campaign and review every path, target, weight,
resolution, and guard before running it.

Files named `stage*` are historical campaign configs. They are reproducibility
evidence, not defaults for YF_0 or YF_1. Their weights and thresholds may use
old grids or objectives.

Resolution order is:

1. mode preset;
2. JSON values;
3. command-line overrides.

Unknown JSON option names stop the optimizer before it creates a run directory.
The top-level fields `name`, `note`, `notes`, and `outer_gate_note` remain
available for campaign metadata. Both flat options and grouped options such
as `weights.w_qi` and `r2.nphi` are supported.

The optimizer writes the effective values to
`<run_dir>/input_parameter.resolved.json`. Preserve that file with the run.

The [historical QI refinement settings](refinement/legacy_negative_iota.json)
are for the standalone refinement command. Copy that file to select the
weights and targets for a different operating point.
