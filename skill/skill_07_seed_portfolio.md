# Skill 07 - Seed Portfolio and Failure-Aware Triage

Use this skill before spending more than one short repair stage on a new basin.
The goal is not to find a seed with one excellent proxy; it is to find several
distinct, repairable basins and eliminate structurally bad lines cheaply.

## Portfolio, not a single ancestor

For each campaign, keep seeds distinct along at least these axes:

- source family: W7-X VMEC, DESC example, Goodman/near-axis, ConStellaration,
  OOPS/PW, or an independently optimized archive;
- transform family: low, medium, and high iota, including both shear signs;
- geometry family: aspect ratio, mirror ratio, elongation, and boundary
  curvature/pdrot;
- equilibrium heritage: vacuum, prescribed-iota, free-iota/current, and
  finite-beta profiles.

Do not count pressure copies or small Fourier perturbations as independent
seed families.

## Three-stage triage

1. **Manifest prefilter.** Reject impossible size, field, aspect, mirror, or
   rational-crossing combinations before VMEC. Preserve diversity; do not sort
   only by QI.
2. **Vacuum smoke.** Run fixed-boundary VMEC and cheap diagnostics. Record
   convergence, resets, aspect, B0/Bmax, iota axis/edge, rational crossings,
   QI, max-J pass ratio, well, DMerc, and boundary geometry.
3. **Finite-beta response.** For survivors, run a fixed-boundary pressure
   ladder before shape optimization. Vacuum DMerc is not a finite-beta verdict.

The ConStellaration 2026-07-03 smoke campaign demonstrated why all three are
needed. Many seeds converged and had attractive QI, but most had negative well
and/or negative DMerc. The diverse 48-seed batch found candidates near
`DMerc ~ 0`, while narrow ranking families missed other response types.

## Failure-aware ranking

Rank by repairability, not by a weighted sum that hides blockers. Keep the
following columns separate:

- hard failures: VMEC failure, nonphysical boundary, field/size cap;
- MHD distance: DMerc minimum/negative count, well depth, ballooning margin;
- topology distance: low-order rational crossings and iota shear;
- optimization quality: QI, max-J pass ratio, Bmin trend;
- engineering distance: pdrot, k2/H, elongation, and same-protocol coil proxy.

A candidate with moderate QI and one movable MHD blocker is often more useful
than a spectacular-QI candidate with a magnetic hill, many rational crossings,
and a hard curvature hotspot.

## When to repair and when to switch seeds

Use one short guarded repair to measure local response. Switch basin when:

- 8-30 evaluations move the active blocker by only numerical noise;
- the best blocker improvement is immediately lost to iota, well, pdrot, or
  beta drift;
- the same blocker survives two parameterizations;
- a pressure/profile ladder shows the entire fixed boundary has the wrong
  finite-beta response.

Observed examples:

- C003/C040/FA2 guarded Mercier repairs changed DMerc by only about
  `0.002-0.05`; they are response probes, not continuation successes.
- MPW Dnjn needed a wide 30-DoF probe to move DMerc from about `-0.328` to
  `-0.232`, while pdrot and iota worsened. This is still a hard basin.
- C007 fixed-boundary pressure at beta about `1.23%` had 35 unstable
  ballooning points and negative DMerc/well. More local QI repair is not the
  first move; change seed, profile, or boundary family.

## Evidence locations

- `/home/guozx/runs/constellaration_nfp4_seed_smoke_20260703/`
- `/home/guozx/runs/constellaration_c003_c040_repair_20260703/`
- `/home/guozx/runs/constellaration_failure_aware_repair_probes_20260703/`
- `/home/guozx/runs/constellaration_c007_beta_ladder_20260704/`
- `/home/guozx/runs/oops_nfp4_reconstruction_20260703/`

Keep detailed rankings in those run folders. This skill stores only the
portable decision rules.
