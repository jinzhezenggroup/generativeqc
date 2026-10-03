# Proposal: adopt the shared HF density-increment kernels at the KS linear boundary

Status: proposed; experimental implementation, no default promotion
Date: 2026-10-04

## Problem

The prepared CUDA KS facade still received full density on every iteration.
HF already owned safeguarded delta preparation, periodic refresh and anchor
reconstruction. Its anchor was hcore plus the spin Fock, which cannot be reused
as a nonlinear KS Fock: XC must be recomputed from the full current density,
and KS energy accounting needs separate raw J and same-spin K channels.

## Decision under qualification

Extend the existing HF prepare/finalize kernels to accept a separate linear
channel count and an optional hcore. The default preserves the HF layout.
KS supplies null hcore and contiguous J/K channels from its checked arena;
no second J/K provider, scientific contraction or screening implementation is
introduced. The ordinary provider sees delta density and applies its existing
density-dependent gates. XC, physical residuals, DIIS and energy assembly see
the reconstructed full channels and full current density.

The existing `GENERATIVEQC_INCREMENTAL_DIRECT_JK` benchmark selector and interval
parser are shared between HF and KS, remain off by default, and reject malformed
values. Public KS admission requires CUDA, strict FP64, full-range exact J/K and
no nonlocal correlation. DF, range corrections, mixed-precision stages and
host-unfused XC are refused rather than silently reverting. Device chunks are
not admitted for this experiment.

## Invariants

- Each solve, including warm replay, starts with a new full-density anchor.
  The prepared provider owns immutable geometry/basis/operator identity.
- Nonzero screening permits only one delta application between full refreshes,
  matching HF. Unscreened execution may honor the wider requested interval.
- Only linear channels enter the anchor. A failed attempt cannot publish warm
  or final state; the next solve resets the incremental controller.
- Every candidate convergence enters the existing bounded full-density physical
  closure, including RKS. Final E/D/F and exported forces still pass the strict
  residual, density-change and unshifted final-state checks.
- The arena charges anchor density, J/K channels and device control scalars.
  Dead `proposal` storage holds delta density until the eigensolver overwrites it.
  The v1 shape bridge has no hybrid flag, so it conservatively reserves the
  larger J plus spin-K anchor; ordinary hybrid storage remains separately
  charged. The nine host bytes used by the observation are also charged.
- Build/refresh/finalization counters describe executed provider applications.
  The KS prepared provider does not yet expose a complete integral-work census;
  `quartet_work_counters_valid` remains false. Smaller delta density or fewer
  full builds alone does not prove less recurrence work.

## Evidence so far

Base master: `837c2a51c06c6d38a6edc4d41da3060573ca40ab`.
The internal-only first source was
`0639a697567373c714a78db255e3902949dadce74fa39c9fd3b821e6f22168b2`.
Slurm n1 job 5659 passed both native incremental-HF and complete CUDA-KS suites.
New controls cover unequal density/output channel counts, three independent
items including an inactive item, periodic full/delta transitions, nullable
hcore, screened and unscreened RKS/UKS, independent CPU energy/components,
warm anchor reset and strict final-state export. They do not establish large
system performance or resolve the retained def2-SVP OH convergence failure.

The subsequent public-selector/resource-bridge source is
`50d198c3ebec1d5c770a14d0b7ce0a11afd4e2523b83d2ab81853bb9cce5a5c2`.
Its native HF/KS suites pass separately. In n5 job 1415, the independent
spherical def2-SVP water and water-cation OFF controls pass complete E/F and
two-step reconverged finite differences. Both ON tests fail in the diagnostic
assertion: `KsDiagnostic` is a record, not a subscriptable dictionary. The ON
SCF/force calls returned, but their independent reference gates had not yet
executed; these remain failed tests, not numerical passes. The corrected test
uses the record's `fock_builds` attribute. Both failures and the two passes are
retained separately from subsequent reruns.

Further audit found the version-1 precision inventory has no operator kind for
delta preparation/channel reconstruction. The experiment now preserves actual
Fock/final-audit events but leaves the operator-completeness flags false rather
than inheriting a false full-inventory certification. This and formatting change
the source identity; the older native/FD results are not exact-head evidence.

All native builds run on n5 with explicit CXX/CUDA ccache launchers and a
checkout-root `CCACHE_BASEDIR`; libraries transfer directly n5 to n1. The first
remote GitHub fetch failed with TLS error. A local incremental Git bundle
provided the verified base; the initial build receipt retains its older base
metadata and was superseded by a fresh verified-base build before job 5659.
No numerical run is relabelled as the failed fetch or the first build.

## Remaining gates and rejected shortcuts

Do not reuse a complete XC-containing Fock, treat independent channels as spin
matrices, accumulate arbitrarily many screened deltas, or publish a delta-built
final Fock without a full physical audit. Do not infer work counts from build
counts. The additional conservative final closure can increase warm cost;
retain that cost rather than timing only SCF delta applications.

Still required: actual provider work observations, all complete cold/warm/moved
48/96 endpoints with independent references, larger numerical/force controls,
failure and tight-budget qualification, and sanitizer evidence for the changed
shared kernels. No speedup, convergence fix, merge readiness or default
promotion is claimed by the current native correctness controls.

## References

### Current qualification checkpoint

After the diagnostic-test correction, precision-inventory boundary and formatting,
the frozen source is
`8269ab4411352e9ee653efafa85522a1ce4a0c15ecfe76f8e944efce54a22630`, library
`0455a330b42871e9e4b4e8255b0d8a9351623128446ec68ef904e6453e94fd70`.
All 196 focused host checks and the changed-file pre-commit hooks pass.
The broader host selection previously had four missing registered D3-source
cache failures; no unrelated source-registry or dispersion code was changed.

n1 job 5663 passes both native suites and whole-test memcheck/initcheck. n5 job
1416 passes all seven public controls: three metadata-only anchor/overflow
admissions and four OFF/ON RKS/UKS independent E/F/warm/two-step-FD cases.
Maximum independent E/F errors across those cases are 2.70e-13 Eh / 6.06e-11
Eh/Bohr. Cold RKS executes 9 full + 9 delta + 1 final full builds; UKS executes
11 + 11 + 1. Warm performs 1 full + 0 delta + 1 final full, confirming the
conservative audit cost rather than demonstrating a warm speedup.

The whole HF test fails synccheck with 96 divergent-warp barrier reports in the
unchanged `reduce_shell_pair_density_bounds_kernel`, reached by `verify_case`.
An independent original HF fixture linked to unmodified-master 9c54107c library
`73c54dc18529365e25b9d4bfb70717002170d37932becffe5193e5548f530e78` reproduces
the same 96 reports. This does not establish the root cause or close the full
sanitizer gate. Do not modify an unrelated reduction or suppress the failure to
qualify this experiment. An initial scoped invocation also had an invalid CLI
filter and exited before checking; its retry uses the documented `kns=` syntax.
That scoped synccheck still reports the graph's density-bounds kernel, so it
does not isolate a clean changed-kernel gate. Scoped racecheck reports zero
hazards, not a blanket all-kernel certification.

Frozen-source n1 jobs 5664/5665 measure fresh reference/OFF/ON complete endpoints
at 48/96 atoms. Their wrapper records actual `KsDiagnostic.fock_builds` and
incremental full/delta/final counts, preserving unavailable integral counters.
No endpoint result is assumed while these jobs are incomplete. This is an
ordered first campaign, not matched/interleaved speedup evidence.

- Progress: issue #1423, comment 5972796394.

- Shared sequencing and acceptance: issue #1423, comment 5968957958.
- Existing HF foundation: #990 and #1579.
- Separate derivative AO-screen experiment: draft #1798. Its performance
  observations are not evidence for this incremental-SCF source.

The unconditional RKS closure policy is superseded experimentally by
`2026-10-04-reuse-full-rks-audit.md`. It preserves these frozen-source observations
and separately documents the negative large-system warm results; neither is
relabeled as follow-up-source evidence.
