# Decision: certify the current ordinary CPU UKS state before leaving DIIS

Status: implemented
Date: 2026-11-18

The final-closure strategy below is superseded by
[monotone physical polishing](2026-11-18-uks-monotone-final-polishing.md).
The primary DIIS certificate remains required. Hosted run `44218958302` passes
core-b and compiler-heavy, but exposes two VV10 force-residual regressions in
core-c; the local results below do not establish an all-green hosted result.

## Problem

PR #3083's CI run `44134150903` finished its Python shards within their existing
deadlines, but `core-c` exposed real nonconvergence in the polarized automatic
`MGGA_X_LTA` lifecycle. The same failure reproduces locally: the cold four-H
endpoint exhausts 200 iterations at energy `-1.8830810105810145`. Its history
repeatedly reaches a small physical residual, then restarts after the bounded
closure's unaccelerated density projections amplify the remaining error.

The unshifted physical projector is a certificate, not necessarily a contractive
fixed-point iteration. Replacing a DIIS-converged density with that projector
can destroy convergence even when the exported state can meet every existing
scientific gate. Increasing the cap or exempting stabilized occupations would
not repair this distinction.

## Decision

For ordinary CPU UKS owners requiring an automatic-program or explicit export
certificate, check the current density against the unshifted physical Fock
projector before the shared iteration driver accepts convergence. If this gate
fails, keep iterating with DIIS instead of leaving it and discarding its history.
Only compute the additional projector near an otherwise eligible convergence;
include its live arrays in the CPU capacity observation.

The bounded final closure independently rebuilds `F[D]` for that same certified
density rather than taking another unaccelerated projection step. It still
checks energy change, density drift, physical RMS/maximum-entry residuals and
the unshifted canonical RMS/maximum-entry density reconstruction. Preserve the
original SCF budget, all numerical thresholds and independent export validator.
Generic curated owners without an export request retain their occupation policy;
non-ordinary providers retain their existing bounded correction path.

This refines the iteration strategy in
[explicit export certification](2026-11-17-explicit-ks-export-certification.md),
not its consumer-selection rule or historical source-specific measurements.

## Evidence

- Before the repair, the exact CI failure reproduces with the cached Release
  CPU library, including repeated rejected physical closures within 200 steps.
- After the repair, all 26 automatic CPU KS lifecycle tests pass in 33.73 s,
  including polarized LTA cold/warm/changed-geometry execution. They compare
  energies, physical Focks and densities against pinned PySCF/LibXC, retain
  independent fixed-density directional derivatives, and check stale snapshots,
  JSON-finite diagnostics and semantic Coulomb/exchange work accounting.
- Local receipts and subsequent broader qualification are retained under
  `/data/jzzeng/pr3083-ci-20261118/remaining/`. They are not a hosted CI pass or
  a practical-scale Hessian, Ni saddle, CUDA, or promotion qualification.
- The final source-matched CPU build passes all 149 native CTests in 88.43 s.
  The focused executable eligibility/admission/history/routing and cache-policy
  probes pass 89 tests (four explicitly GPU-only skips). Eligibility controls
  now independently reject an uncertified density even with a small commutator
  and exercise delayed certification and exhaustion within the original budget.
  Native source identity:
  `d73e1fb390fa66768a6faa5be6ef8e253f6270eaa614f399b1a350bab6750d25`.

## Related bounded CI repairs

The same hosted run exposes outdated native probe stand-ins after the explicit
energy-only interface and DF progress telemetry changes, an uninitialized CUDA
device field in the singlepoint dispatch fixture, and a second stale core-c
timeout-policy expectation. Update their current interface inputs; do not remove
their force-admission, history-marker or routing assertions.

The cooperative host-emulation test hit its 60-second subprocess limit while
repeatedly creating a new 32-thread simulated warp per block. Reuse the warp
across blocks with an explicit completion barrier. All arithmetic, parameter
combinations, sticky-failure injections and the finite timeout stay unchanged.
Its full 21-case module passes locally in 57.23 s; this host emulation is not
real-GPU evidence.

Hosted run `44224991871` subsequently exposes another 60-second emulation
timeout despite the local pass and an intervening hosted pass. The
[persistent sleeping-team decision](../performance/2026-11-18-persistent-host-cooperative-emulation.md)
supersedes that limited per-block worker reuse; no scientific case or timeout
gate is removed.

## Revisit when

RKS and UKS share a typed physical-state certificate with the export owner.
Do not replace executable oracle/lifecycle tests with source-shape or call-count
requirements, or reinterpret a failed or incomplete hosted run as acceptance.
