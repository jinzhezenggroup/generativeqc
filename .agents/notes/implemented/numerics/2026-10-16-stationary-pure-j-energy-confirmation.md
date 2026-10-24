# Decision: confirm stationary pure-J energy at a held density

Status: implemented
Date: 2026-10-16

## Problem

PR #2532's historical exact-direct r²SCAN-3c/96 moved endpoint takes
327.888863 to 380.302499 seconds: +15.985%, with 17 versus 21 SCF
iterations/builds. Different work limits attribution to final validation but
cannot waive complete-endpoint nonregression. The
[temporary 1088-AO hold](../performance/2026-10-16-bound-pure-j-direct-final-validation.md)
contained exposure; disabling the candidate was not a repair.

The failed trajectory keeps applying tiny density proposals after stationarity,
while compensated energy changes oscillate above the strict 1e-12-Eh gate.
Candidate iterations 15–20 report energy changes of approximately
3.61e-12, 2.90e-12, 1.52e-12, 3.52e-12, 7.26e-12 and 2.78e-12 Eh.
The diagnostic called `physical_residual_max` in the serialized history contains
the ordinary physical residual, not the separate true AO maximum. Historical
history alone therefore cannot authorize a relaxed maximum-norm test.

## Decision

For the ordinary single-submission strict-FP64 all-electron exact-direct
restricted pure-J solver, hold the current density when all of these are true:

- A valid energy history exists and a finite energy change still fails its gate.
- Density change already passes the existing density tolerance.
- Both actual physical residual norms pass the existing stationarity tolerance.
- Diagnostics are nonnegative; nonfinite diagnostics cannot authorize a hold.

Submit another complete physical J/XC/F[D] and evaluate its energy, proposal and
residuals at that same density. Do not copy the unaccepted proposal into D.
Invalidate the proposal's intermediate orbital/fitted-factor authorization,
but retain the last-good warm density and compensated energy until convergence.
The subsequent solve must still satisfy every original convergence gate.

This follows the solver's existing held-density confirmation pattern for the
initialized nonlocal target, without changing that path. It does not reuse a
cached energy, publish a provisional state, increase iteration limits, loosen
tolerances or replace the independent final-state proof. The confirmation is
independent of the host/resident final-validation selector.

The new confirmation excludes fitted J/K, exchange, range corrections, nonlocal
correlation, incremental J/K, budget Fock, lower precision, ECP and UKS. Explicit
device chunk mode is unchanged. Those paths require their own algorithmic
qualification rather than inheriting this stopping strategy accidentally.

Restore the original exact-direct pure-J resident default at 544 or more AOs.
DF pure-J retains its independently qualified 544-AO lower bound; direct global
hybrids retain 384 AOs. Matrix-handle, bounded-packet, accepted-generation and
published W/total-D lease guards, mandatory host exports and explicit selector
`0` remain intact. Explicit `1` still cannot bypass those guards.

## Rejected alternatives

- Treating the upper-bound rollback as a performance fix leaves the SCF tail
  unresolved and merely hides the failing default.
- Waiving changed-work regressions, normalizing by iterations, or checking only
  warm timings discards complete-endpoint behavior.
- Relaxing the 1e-12-Eh energy gate, bypassing the true maximum residual, or
  replaying the previous energy changes scientific acceptance.
- Continuing to perturb an already acceptable density provides no stopping-test
  benefit when only the energy confirmation is missing.
- Disabling Direct/48 for a benefit below 2% mistakes the descriptive label for
  an enablement threshold. Its new warm benefit is positive at 1.683%.

## Evidence and provenance

The separate campaign is
`benchmarks/results/r2scan3c-stationary-confirmation-20261016/`. It retains both
arms of complete cold/5-warm/moved/5-moved-warm Direct/48 and Direct/96 endpoints,
with selector `0` versus genuinely unset on one identical image per comparison.
There are 48 comparison calls and 24 additional 3/12-atom accuracy calls.
Independent GPU4PySCF energies/forces, canonical spherical def2-mTZVPP, D4/gCP,
grids, screening and acceptance gates are unchanged. Intrusive route diagnostics
are separate from timing populations. Every complete phase must pass the
nonregression gate, including changed-work phases; warm work must also match.

The repaired Direct/96 moved pair takes 267.157748 versus 263.365821 seconds,
with 14 iterations/builds in both arms. In both new moved histories, held-density
confirmation keeps the last one-electron trace fixed while a freshly computed
compensated energy change passes the original gate. This is not an inferred true
maximum-residual history.

Native source identity is
`c300965dbc535596b2310a23eecb741eabb51dc1dd158f468292c9147f554867`;
core library SHA256 is
`21526c6c1a1ae5258282a92f942c37ddf9536e1094b81a85e4e7697942460e2f`.
The frozen source is base `fb37f6d1c66cdb53ed83b3b68a284024b6060bab` plus the
retained patch, with 1744 native inputs and 610 unchanged upstream inputs.
Its native inventory differs in 46 entries from the older campaign, including
other changes already in the base. Old-to-new absolute timings are therefore
not a controlled patch-only speedup claim. Qualification uses new same-image
control/default comparisons, not relabeled historical timings.

Initial attempts exposed a missing upstream-input snapshot and stale force AOT
contracts. They are retained as failed attempts, not endpoint qualification.
Regenerate the force sources, binaries and manifests through the compiler, then
check the real AOT loader's contracts/binary integrity before expensive SCF.
A Ninja "no work" result alone is not a source-identity check. Do not repair a
contract mismatch by editing its manifest identity. The final launch preflights
the full input inventory, core identity and both force AOT artifacts.

Compiled host/device predicate tests preserve finite-value and strict-boundary
behavior. The publication/failure harness checks no proposal D copy, no premature
publication and preservation of the last-good warm cache. The original evidence
regression test and verifier still reject historical `none/96/moved`.

## Consequences and revisit conditions

One extra full physical iteration can confirm an otherwise stationary density;
it replaces a potentially lengthy noisy tail within the existing iteration budget
rather than avoiding physical work. Ordinary SCF limits and failure behavior
remain authoritative. Keep both true residual
gates, independent endpoint oracles and all negative observations if modifying
this strategy. Separately qualify extensions to other solver modes or producer
classes. A source change, a different hardware cohort, or a warm-only gain is
not permission to waive a slower complete endpoint.

## Integrated-head requalification

After synchronizing the shared solver with `master` through `4a36f6f3e`, rerun
qualification on clean committed head `65cb37d58bcd384f46bf94f9056eb24cd9d294de`.
Its native source identity is
`e84f242a2260dc9cf6e75782b265f32a9fa6b830404f6b8843156edeb93423f9`;
core SHA256 is
`c521f58d53e99d2c164c399b39c1ee90e927a906c867579328acf4987ef79f14`.
All 1753 native inputs are retained against the pinned historical inventory and
checked against this committed head; all 610 upstream inputs are identified.
The preceding `c300...` campaign is not acceptance of this integrated image.

The [integrated evidence packet](../../../../benchmarks/results/r2scan3c-integrated-acceptance-20261016/README.md)
retains all eight passing complete phase comparisons: 48 comparison calls plus
24 candidate-only 3/12-atom calls. Actual energies and full force arrays for all
72 calls permit independent error recomputation, including the small systems.
Maximum energy/force errors are 3.819878e-11 Eh and 6.043033e-11 Eh/Bohr, below
the unchanged 1e-8/1e-7 gates. Both arms have identical semantic work throughout;
the 96-atom moved pair is 261.452944 to 257.768177 seconds with 13 iterations
and builds, not the earlier image's 14. Warm benefits are +2.123% at 48 atoms and
+4.213% at 96 atoms. Different-image absolute timings are not patch-only gains.

Finite RTX 5090 Slurm jobs 1018/node4 and 1639/node5 execute the comparisons and
separate cold/first-warm resident proofs at both sizes; intrusive calls are not
pooled into the timing population. The native proof and retained lease fixture
also pass; the latter adapts only the removed transport getter to
`BatchKsTransportDiagnostics`, without weakening numerical or lifetime gates.
Host qualification passes 290 cases with 36 explicit device skips. Job
1020/node4 passes all 53 targeted cases without skips, including 45 device cases.
The first job's 30-pass/23-skip result remains visible: resource CUDA and DFT CUDA
tests require their separate `GENERATIVEQC_RESOURCE_CUDA_TEST` and
`GENERATIVEQC_DFT_CUDA_TEST` opt-ins, both set to `1` for the successful rerun.

Building force library targets alone leaves separately generated AOT manifests
stale. The real loader rejects both old contracts; rebuilding the normal
`generativeqc_stationary_r2scan_rks_manifest` and
`generativeqc_stationary_r2scan_rks_spd_manifest` targets regenerates valid
metadata, without editing identity fields. Both rejection and recovery receipts,
all build logs and shared cumulative ccache statistics are retained.

The fixed evidence cap is not relaxed. Losslessly compressing #409's archived
source patch saves 191789 bytes while preserving its original decoded SHA256,
all source bytes, scientific data and incomplete status. Both manifests and
recovery instructions are updated; a one-off comparison against the original
Git bytes passes. The original failed Direct/96 cohort, prior repaired-image
cohort, historical figures and earlier negative attempts remain unchanged.
