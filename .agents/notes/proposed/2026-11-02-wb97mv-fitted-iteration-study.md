# Investigation: WB97M-V fitted cold iteration work

Status: proposed (further inner-schedule tuning; current policy unchanged)
Date: 2026-11-02

## Finding

The source `d243aaf78c82f870d8110b40eeddc2950489cfbe866f586d070487f1e695562f`
A100 water-96 fitted default takes 28 target-side Focks: nine semilocal/RSH,
fourteen frozen-VV10, and five complete-target evaluations. Its four pair sweeps
and one exact cache hit are separate from seven preliminary atomic Hückel Focks.
GPU4PySCF's ten cycles/eleven potential builds use different work and stopping
norms, so the counts alone do not demonstrate a convergence defect.

The strict inner policy in `src/dft/application_precision.hpp` was previously
qualified primarily for Direct WB97M-V. DF changes the relative cost of J/K,
XC and nonlocal work, making renewed endpoint experiments reasonable. However,
turning off the provisional schedule is already a rejected replacement for
this measured fitted workload: with the same explicit bounded LR cache policy,
iterations fall 28 to 16 while VV10 sweeps rise four to sixteen and cold E+F
increases 115.948489 to 167.804866 seconds (44.72%). Both independent numerical
gates pass, with paired maximum force difference below 3.59e-10 Ha/Bohr.
This is one controlled observation per arm, not a statistical speedup study.

## Boundaries to preserve

- Keep full-target confirmation, physical residuals, energy/density tolerances,
  canonical export, finite total/proposal budgets and strict FP64 unchanged.
- The stored residual diagnostic is max-over-spin RMS, not AO maximum. Never
  use it to assert that the AO-maximum stopping gate already passed.
- Disabling the outer solver also disables automatic LR residency. Hold cache
  policy fixed before attributing time changes to the VV10 iteration schedule.
- Explicit public MINAO is not admitted for DF targets. A rejected diagnostic
  request does not authorize silently changing that production boundary.
- No GPU-name selector or reference-engine work belongs in native execution.

## Next experiment

Measure earlier full-target refreshes or residual-aware inner termination while
retaining the target acceptance gates. Compare complete E+F work, including pair
sweeps and source preparation, on A100 and RTX 5090 before changing defaults.
The prior Direct policy's rejected alternatives remain relevant; fewer Focks
can still cost more. This investigation makes no universal non-regression claim.

## Evidence

[Exact diagnostic campaign](../../../benchmarks/results/wb97mv-iteration-study-20261102/README.md)
retains the successful cache-controlled allocation 224502, incomplete 224501,
and transfer failure 224499. Normal and optimized replay must agree. The
[Direct policy rationale](../implemented/performance/2026-10-18-vv10-inner-default.md)
retains the earlier threshold experiment and its scope.
