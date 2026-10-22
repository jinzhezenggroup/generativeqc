# Decision: bounded streamed DF Hermite contractions and separate Fast XC proposals

Status: implemented
Date: 2026-10-22

## Problem and decision

Repeated raw three-center materialization dominated streamed pure-J DF, including
the r2SCAN-3c README case. The independently qualified Hermite charge, projection
and derivative owner now becomes automatic when its complete counted storage
fits. Full-rank metrics, orbital shells through f and auxiliary shells through g
are required. Resident fitted factors keep their existing policy; unsupported
angular, rank and capacity cases retain the bounded raw path. Explicit
`GENERATIVEQC_DF_MD_COULOMB=0` restores raw selection. No screening, metric cutoff,
quadrature, Standard convergence or integral precision changes.

Fast previously fell back to strict DF SCF even when its local XC products could
use the qualified shared proposal controller. A separate private
`generativeqc_application_df_xc_precision_version_v1` capability admits default
Treutler PBE/r2SCAN RKS, including canonical r2SCAN-3c, at 24–1200 AOs, 2–96 atoms
and orbital shells through d. It narrows only XC matrix products. DF J, metric,
XC point algebra and final physical Focks remain FP64. The existing bounded
proposal/refinement controller and explicit tighter caller ceilings remain in
force. This capability does not claim a mixed DF integral provider.

At 96–1200 AOs, a separate complete-model gradient identity allows combined
budget-converged SCF and mixed semilocal geometry. Admission requires the live
native snapshot's authenticated stationarity allowance, at most 1e-4; a caller
tolerance or functional label cannot substitute for it. Integral and metric
response, moving-grid response, corrections and global reductions remain strict.
The policy rejects unqualified mixed gradient evidence even when the SCF census
is complete. Explicit observable targets retain their independent audit.

## Evidence

The candidate combines master `7522994e2`, the native DF owner, qualified
default-grid gradients, final-state-aware stopping and angular mapping at
`c5bc8690f`. Ignored artifacts under `df-md-default-v1` retain exact source
manifests and patches; core SHA-256 is
`1f9269ace2024ae4ee13f536394d9f96a3c4e221403bac157e13705d4ba811e7`.
Matching copies are retained under
`n1:/data/jzzeng/scf-performance-20261021/`.

All 24 PBE/r2SCAN/r2SCAN-3c × 3/12/48/96-atom × Standard/Fast populations pass
independent README gates for cold and each of three frozen warm calls: energy
1e-5 Eh/atom, maximum force 1e-4 Eh/Bohr and RMS force 3e-5 Eh/Bohr, including
the existing independent reference-stability checks. A separate instrumented
r2SCAN-3c/96 population also passes. These diagnostic receipts are not official
README timings; publication requires a fresh unified campaign.

| DF96 | Standard cold / median warm (s) | Fast cold / median warm (s) | Cold iterations |
| --- | ---: | ---: | ---: |
| PBE | 43.309 / 6.830 | 38.498 / 6.506 | 14 / 9 |
| r2SCAN | 48.407 / 8.060 | 39.442 / 6.831 | 14 / 9 |
| r2SCAN-3c | 70.221 / 15.949 | 47.873 / 14.686 | 15 / 10 |

Every warm call takes one iteration. Fast reports one qualified solve, without
an extra observable audit. These measurements improve the native baseline but
do not establish universal leadership over GPU4PySCF.

The companion default/rollback endpoint suite passes 24 cases / 96 native E/F
calls across PBE spherical/Cartesian orbital and r2SCAN spherical models, each
with spherical auxiliary functions, native/tighter controls, changed geometry
and frozen replay. Standard gates are energy 2e-7 Eh, maximum force 2e-6 Eh/Bohr
and RMS 7e-7 Eh/Bohr; Fast uses the common gates. Production runs exclude CPU
derivative work. Trace checks verify the selected response path. Spherical cases
passed in `df-md-default-geometry-v3`; Cartesian-orbital cases passed in v5.
Combined Fast/default PBE geometry/replay also passes Compute Sanitizer memcheck
with zero errors (`df-xc-sanitizers-v1`). The underlying owner retains its 178
independent angular tests, 16 integrated metric/force tests and earlier clean
memcheck/racecheck evidence from the angular-mapping qualification.

## Retained failures and rejected alternatives

- The first XC proposal prototype left the native blanket DF/AUTO rejection in
  place. Six Fast jobs failed before SCF. V2's separate native gate passed all
  six reruns. Those failures remain under `df-xc-budget-v1`.
- Early geometry fixtures compared unlike auxiliary representations. Both the
  automatic and rollback paths differed by about 2.8e-5 Eh. Trying Cartesian
  auxiliary functions also exercised resource/rank differences. Final tests
  explicitly match the original spherical auxiliary model in the independent
  oracle; no numerical gate was weakened. Private native/NumPy/libcint tests
  retain truncated-rank and Cartesian algebra coverage. A separate attempt with
  an incorrect library path failed before native execution. All attempts remain.
- Forcing the 96-atom resident PBE/r2SCAN owners into streamed execution with an
  explicit 4 GiB DF cap improves cold but regresses warm. PBE Standard becomes
  32.497 / 8.398 s and Fast 23.722 / 8.051 s; r2SCAN becomes 36.316 / 9.410 s and
  24.950 / 8.327 s. Keep the resident policy. These explicit-cap diagnostics are
  not default measurements; 2 GiB attempts refused mandatory metric capacity.

## Remaining work and revisit conditions

The r2SCAN-3c/96 trace still spends about 1.458 s per warm J and 4.631 s on DF
force response. The charge/projection submission describes about 1.9075 billion
logical primitive interactions per complete J, not a measured recurrence census.
This remaining work and strict Standard convergence explain a residual gap;
neither is permission to relax gates or report a universal win. Any future source
compaction, screening, residency or precision change needs independent angular,
geometry, lifetime, capacity, sanitizer and complete endpoint qualification.

See the [angular mapping decision](2026-10-22-df-hermite-angular-mapping.md),
[strict-density DF Fast decision](2026-10-22-default-grid-df-fast-gradient.md), and
[final-state stopping decision](2026-10-22-final-state-aware-ks-stopping.md).

## Clean-source publication follow-up

Clean commit `1178a03c7f45074cc705a7accd676f0ffc1b6d2a` rebuilt matching core and
all ten stationary AOT library/manifest pairs. Core SHA-256 is
`05bbb70e37fc8836216fca394b2d84e5c6676a6f08820e7937073cc449f63f14`.
All 24 geometry cases now pass together against this binary, including the
corrected Cartesian-orbital/spherical-auxiliary fixture and all 96 E/F calls.

The [complete README campaign](../../../../benchmarks/results/readme-final-20261022/README.md)
retains 192 fresh populations / 768 complete calls from this source. It reports
174 qualified populations and the same 18 default-grid/fitting failures across
all three profiles. All 256 Fast calls execute one solve. Standard/Fast win
44/58 and 45/58 qualified warm comparisons, but slower valid cold/warm points
remain visible. Small strict-fallback Fast calls retain host-policy overhead;
same-GPU instrumented B3LYP/3 native batch/force work is essentially equal.
No diagnostic timing replaces an official point and no universal speedup follows.
