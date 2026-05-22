Run output directory.

Use one subdirectory per optimisation or diagnostic comparison, for example:

```bash
python scripts/diagnose.py \
  --nc_file artifacts/legacy_root/wout_squid_optimized.nc \
  --output_dir runs/diagnose_wout_squid_optimized \
  --plot
```

Files under run subdirectories are ignored by git. Keep only small summary
notes in tracked documentation.
