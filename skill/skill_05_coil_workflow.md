# Skill 05 — Coil Feasibility Workflow

Three levels, increasing fidelity. Never promote on level 1 alone.

## Level 1 — inner-loop coil proxy (inside optimize.py)

`w_coil_proxy_bn / w_coil_proxy_k` with low resolution
(M_Phi=N_Phi=4, source/eval 16). Biases the search away from
coil-impossible shapes. Its Bn numbers are NOT comparable to level 2
absolute values; treat as steering only.

## Level 2 — current-potential scan (REGCOIL-like)

```bash
python scripts/coil_feasibility_gate.py \
  --wout <wout.nc> --output_dir <dir> \
  --offsets 0.35 --lambdas 1e-8 \
  --M_Phi 6 --N_Phi 6 --desc_L 6 --desc_M 6 --desc_N 6 \
  --source_M 24 --source_N 24 --eval_M 24 --eval_N 24
```

Standard comparison protocol (do not change when comparing branches):
constant-normal winding surface at offset `0.35a`, `lambda = 1e-8`,
regcoil regularization, helicity (1,0). Record `Bn_max_abs_unitless`
(= Bn_max/|B|), `Bn_rms_unitless`, `K_rms`, `phi_high_mode_fraction`.

Guardrails (same-protocol comparisons, not universal physics):

- `Bn_max/|B| <= 3.4e-3` (route1-derived guardrail; desc_seed currently
  ~3.6e-3, route2 blocked at ~4.16e-3)
- pseudoinverse fallback after singular-system detection is logged — note
  it in results; tiny lambdas (< 1e-10) are numerically fragile.

## Level 3 — TRUE coil contours (the deciding engineering metric)

```bash
python scripts/coil_contour_metrics.py \
  --wout <wout.nc> --output_dir <dir> \
  --offset 0.35 --lambda_regularization 1e-14 --num_coils 4
```

Reports per-coil length, curvature (mean/rms/max/p95), min coil-plasma
distance, min coil-coil distance, plus a 3D plot. Decision rules:

- `curvature_max` is the primary scalar guardrail (campaign reference
  values: route1 ~8.96, route2 ~8.86, desc_seed ~8.27-8.30 — lower better).
- `min_coil_plasma_distance`: watch closely at R=2.35/a~0.22; values
  ~0.064 m (~0.3a) leave little room for conductor + casing + first wall.
  Treat < 0.06 m as a blocker, 0.06-0.08 m as a warning.
- Coil length < 5 m, `Bmax <= 3.1 T` (final-size checks).
- A lower-Bn point that worsens true contour curvature is NOT an
  improvement (route2 offset-0.30 lesson: Bn 4.07e-3 but curvature 10.7
  and d_plasma 0.047 m — rejected).

## Known-exhausted moves (do not repeat)

- Offset/lambda/M_Phi-N_Phi grid scans to fix route2 `Bn_max` — done
  (M4/N4 and M5/N5); best 4.07e-3 at unacceptable geometry. Next attempts
  must change the method: optimized (non-offset) winding surface,
  localized boundary repair driven by the Bn hotspot map, or true
  filamentary coil optimization (e.g. simsopt) used as the gate.
- Improving `pdrot_max` while true contour curvature worsens
  (stage05/eval012 lesson).

## Proxy-calibration task (open)

The 3.4e-3 Bn guardrail is inherited from route1, not validated against a
real coil design. When a real filament/REGCOIL workflow result is
available for any branch, recalibrate the guardrail and update this file
and GUIDELINES.md.
