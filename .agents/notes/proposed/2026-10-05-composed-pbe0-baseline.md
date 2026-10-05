# Decision: retain the optimized composition as the PBE0 comparison baseline

Status: proposed (further algorithm work and default decisions remain open)
Date: 2026-10-05

## Problem

Comparing new work with an unoptimized dense master would overstate progress
against #1895. Existing #1847, #1830 and #1833 implementations compose with
the explicit indexed force policy, although they are not production defaults.

## Decision and evidence

Use the frozen master `1de139fa5` composition `cd121e3af` as the measured
optimized baseline. The retained [publication](../../../benchmarks/results/pbe0-composed-baseline-20261005/README.md)
contains all cold/five-warm/moved/five-moved-warm E/F samples for 48 and 96
atoms and matched full-density-rebuild GPU4PySCF references. Its canonical
source identity is reconstructible from a permanent master commit and a
compact source patch. It does not qualify subsequently merged source changes.

All 144 same-geometry E/F pairings pass the original 1e-8/1e-7 absolute gates.
Warm complete endpoint ratios remain 1.77x and 3.27x; moved-warm ratios are
2.73x and 3.26x. Keep every reference iteration variation and timing sample.
These data do not satisfy the 1.25x target or authorize default promotion.

For 96 atoms, force execution is about 70% of the remaining warm endpoint.
The stationary derivative and semilocal geometry-response medians are 14.35 s
and 8.81 s. Independent J/K dispatch is useful infrastructure, but a J/K-only
optimization cannot close this measured gap. Continue #1892, #1893 and #1894
with the force owners rather than collecting isolated kernel improvements.

Separate fixed-density CUDA-event profiling reports J/K/XC independently. Its
fresh energy-only trajectories differ from the E/F protocol and must not be
substituted for endpoint measurements or summed as a clean wall-time partition.
Actual indexed page counts and per-build J/K admissions were not observed;
native public Fock-build counts remain null. Do not infer them from iterations.

## Invariants and revisit conditions

Keep exact-direct semantics, full moving-grid response, strict final-state
audits, independent oracles, and bounded dense fallbacks. Treat DF/COSX as
separate scientific comparisons. Re-run all four endpoint regimes after a
meaningful algorithm change, retain cold/moved regressions, and qualify any
default separately against the repository's promotion requirements.

Refs #1895, #1892, #1893, #1894, #1834, #1847, #1830, #1833.
