# Skill 06 — Known Pitfalls and Failure Modes (check before acting)

Each entry: symptom → root cause → rule.

1. **VMEC wout `xn` vs input `RBC(n,m)` index mismatch.**
   wout `xn` is PHYSICAL toroidal mode number; input `n` is PER-PERIOD.
   Caused the invalid FFT table (SEED_INVESTIGATION Section 3) and the
   mis-indexed `input.desc_beta1p5`. Rule: when moving between wout and
   input, divide/multiply by NFP explicitly and sanity-check that the
   dominant helical sits at per-period n=1.

2. **Fourier normalization bugs.** Any spectral table where the (0,0)
   coefficient does not equal the known major radius is wrong (the 2026-06-11
   table had RBC(0,0)=2.35/4). Rule: always print the (0,0) check and a
   physical-space reconstruction RMS before using FFT-extracted coefficients.

3. **Periodicity filter vs reindexing confusion.** A strictly NFP=5 boundary
   has ZERO n=4 content; only reindexing transfers helical shaping to NFP=4.
   Rule: never quantitatively compare outputs of the two conversions.

4. **`mhd_gate.py` PASS != promotable.** The gate does not enforce
   `DMerc_min >= 0.08` and stops at rho=0.95 for ballooning. Rule: apply the
   Skill 04 promotion checklist; rerun marginal candidates with extended
   edge rho (0.975, 0.99).

5. **Marginal ballooning ridge-riding.** lambda_max within (-2e-5, 0] flips
   with 0.001% beta changes (route1 1.411 vs 1.412). Rule: require
   lambda_max <= -2e-5 margin before stepping beta up.

6. **Internal vs external ballooning scale mismatch.** Internal optimizer
   ballooning residuals (~0.078) and external gate lambda (~1e-6) are not
   the same quantity (stage07). Rule: do not run gradient pushes on the
   internal ballooning objective expecting external-gate movement.

7. **Pressure-only beta pushes on a frozen boundary plateau quickly.**
   Route1 ceiling 1.411%; tiny low-order probes did not move it. Rule:
   ramp beta WITH boundary re-optimization at each step (continuation),
   and change parameterization if a blocker does not move within one short run.

8. **Optimizer penalty values are not physics.** f_QI etc. depend on grids
   and normalization. Rule: compare only within the same code/grid/config;
   acceptance uses external diagnostics.

9. **Coil proxy improves while real coils worsen.** Current-potential Bn can
   drop while true contours twist (phase01/02 history; route2 offset scan).
   Rule: true contour curvature + clearances decide; fixed protocol always.

10. **Iota target no-op.** Penalty inactive inside `iota_tolerance`; strong
    shape anchor + small dof bounds freeze the boundary (desc stage02:
    iota_edge moved <0.001 toward a 0.04 target). Rule: size tolerance and
    anchors against the intended move before launching.

11. **Uncheckpointed best points.** Stage01's internal best (eval27) was lost
    to cadence-5 checkpoints. Rule: `checkpoint_every: 2-3` in new basins.

12. **Silent beta drift during optimization.** A "better" checkpoint at lower
    beta is not better. Rule: gate JSONs must show beta within 0.02%
    absolute of branch target; otherwise rescale pressure and re-gate.

13. **Scaling shortcuts.** Curvature/pdrot scale as 1/s, Gaussian curvature
    1/s^2 — shrinking R worsens coils; optimize-at-other-scale tricks
    require full re-gating at final size (AGENT_OPTIMIZATION_GUIDE
    "Radius Scaling Warning"). Rule: no MHD/coil claims from non-final scale.

14. **Single-source seed tunnel vision.** The campaign treated one W7-X VMEC
    truncation as "the" seed for weeks; the DESC h5 line (desc_seed) opened
    a better coil basin. Rule: when a branch plateaus on a blocker for >2
    stages, spend one short run on a distinct seed line before more repairs.

15. **Beta scans without profile bookkeeping.** Changing PRES_SCALE changes
    the Shafranov shift, well and DMerc together. Rule: record PRES_SCALE,
    AM[], and achieved betatotal in every seed folder (the
    `seeds/beta*_from_*` convention).

16. **Mojibake / encoding.** Write campaign docs in ASCII (avoid box-drawing
    and arrows) — files in this repo are read from both Linux and Windows.
