# Skill 06 - Failure Modes to Check Before Trusting Results

Use this as a preflight checklist when output is surprising.

## Identity and reconstruction

1. **Wrong file or operating point.** Verify hash, NFP, R/a, beta, B0,
   current, profiles, and resolution before comparing.
2. **wout/input Fourier mismatch.** wout `xn` is physical toroidal mode;
   input `RBC(n,m)` uses per-period n. Check `RBC(0,0)` and a physical-space
   reconstruction.
3. **Current/profile loss in NCURR=1 reconstruction.** Preserve
   `AC/AC_AUX/CURTOR` and pressure auxiliaries; compare parent and rebuilt iota,
   current, and beta before optimization.

## Numerical interpretation

4. **Mercier normalization or grid drift.** Use `Phi_edge^2 * VMEC.DMerc`,
   remove the leading placeholder, and map to the VMEC half-grid. Always state
   radial mask and `ns`.
5. **Near-axis or edge samples dominate a scalar.** Report the full profile,
   minimum location, negative count, and body/axis regions separately.
6. **Resolution-dependent sign or raw maximum.** Refine radial, spectral, and
   surface sampling. A non-convergent geometry tail is a defect, not a metric
   to average away.
7. **Different protocols share a metric name.** Ripple, ballooning, pdrot,
   REGCOIL-like Bn, and curvature values are comparable only with matching
   implementation and grids.
8. **GPU smoke test mistaken for workflow support.** Current DESC ripple uses
   a CPU-only `nufft2` path in this environment.

## Optimization behavior

9. **Penalty value treated as physics.** Optimizer residuals are steering
   quantities; promotion uses independent diagnostics.
10. **Evaluation-cap state treated as optimum.** Rebuild and postcheck it as a
    trial only; verify the solver actually accepted the step.
11. **DoF count mistaken for mode coverage.** Inspect the free-DoF list.
12. **Iota objective is inactive.** Check tolerance, anchors, current policy,
    and whether the desired profile is self-consistent.
13. **Silent beta/current/profile drift pays for improvement.** Compare the
    complete operating state at every checkpoint.
14. **One weighted run tries to fix unrelated blockers.** Use short isolated
    response probes; switch basin after repeated flat/harmful response.

## Physical reasoning

15. **Vacuum stability used as the finite-beta verdict.** Run pressure level
    and profile ladders; finite beta can stabilize or destabilize.
16. **Field reduction conflates scaling policies.** Fixed beta, fixed pressure,
    pressure-only, and geometry scaling are different experiments. Use Skill
    08.
17. **Rational crossing interpreted as an island or ranked by denominator
    alone.** Separate NFP-compatible natural harmonics from symmetry-breaking
    exposure modes, then compute resonant response in the actual coil field.
    Follow Skill 10.
18. **Global normal-field error treated as sufficient.** Inspect resonant and
    localized components plus coil-return physics.
19. **Boundary pdrot treated as coil buildability.** It is a local geometry
    diagnostic, not the final coil metric.
20. **Good nominal point mistaken for a steady operating window.** Perturb
    beta, profiles, current, resolution, and coil errors before promotion.

## Repository hygiene

21. **Calculation runs from source root.** Require explicit run/output paths
    and verify no generated artifacts remain at the root.
22. **Historical handoff treated as current policy.** Use the precedence order
    in `SKILLS_INDEX.md`; old campaign thresholds remain historical evidence.
23. **Mojibake in cross-platform docs.** New operational documentation is
    ASCII unless a scientific symbol is essential and rendering is verified.
