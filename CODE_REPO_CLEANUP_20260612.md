# SQuID Code/Run Cleanup Notes

Updated: 2026-06-12

## What Was Cleaned

The R2.35 campaign run tree was cleaned without deleting data. Superseded and deprecated outputs were moved to:

`runs/w7x_r2p35_goodman_chain/archive_20260612_cleanup/`

The archive contains a manifest in README, CSV, and JSON form. Active evidence now lives in the Phase A/B/D summaries and `final_comparison/`.

Root-level one-off bundles `scripts.zip` and `tests.zip` were moved to `/home/guozx/SQuID/archive_20260612_repo_misc/`.

## Reusable Code Kept

These are no longer one-off snippets; keep them as reusable campaign tools unless replaced deliberately:

- `scripts/mhd_gate.py`
- `scripts/gate_checkpoints.py`
- `scripts/coil_feasibility_gate.py`
- `scripts/coil_contour_metrics.py`
- `scripts/coil_gate_checkpoints.py`
- `scripts/boundary_geometry_gate_checkpoints.py`
- `scripts/physical_diagnostics_report.py`
- `scripts/rescale_and_gate.py`
- `scripts/recompose_vmec_input.py`
- `squid/objectives/coil_proxy.py`
- `squid/diagnostics/boundary_geometry.py`

## Still Messy But Not Removed

The repo has many historical `configs/stage*.json` files and modified core modules. I did not move these automatically because earlier runs may still depend on them and they are part of the broader SQuID development state, not just the R2.35 campaign.

Recommended next cleanup pass:

1. Move obsolete experiment configs to `configs/archive_YYYYMMDD/` only after checking which runs reference them through `input_parameter.resolved.json`.
2. Keep reusable gate/diagnostic scripts under `scripts/`.
3. Move run-specific helper scripts into the run directory or `runs/<campaign>/archive_*/oneoff_scripts/`.
4. Do not put future campaign-only scripts in repo root; use `runs/<campaign>/tools/` or the local Codex `work/` directory.

## Current Policy

Source code should contain reusable mechanisms. Campaign directories should contain experiment-specific choices, logs, and results. Documentation should distinguish durable skills from chronological logs.
