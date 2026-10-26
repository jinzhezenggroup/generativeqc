# Decision: shared native progressive-HF final acceptance

Status: implemented
Date: 2026-10-26

## Problem

The native bounded HF schedule ordered scientific stages, but Python still
decided whether the final target, physical audit, arithmetic and work budgets
established an acceptable result. SDK callers could not reuse that decision.
A native schedule alone does not complete runtime-policy ownership in #2227.

## Decision

Move the complete existing scalar final-verification policy into a constant-storage
native decision exposed through an additive C ABI. Typed frontends compare model
and provider identities, validate observable shapes and present reason strings.
Native code applies energy/density/physical thresholds, strict-FP64 cleanup,
independent accuracy disposition and budget precedence. It receives observations,
not a previously serialized final status. The core has no allocation, callback,
scientific reevaluation or global state.

The running controller passes its already selected target library. Standalone
public metadata verification uses the first configured library. Missing protocols
and integer metadata outside the ABI preserve the named Python compatibility
implementation; native refusal or malformed receipts never retry it. Arbitrary
Python integers must not wrap or truncate into a native work budget.

## Invariants

- A present fixed-density audit uses its maximum commutator for acceptance even
  when RMS is small; it supersedes the optional native residual diagnostic.
- Budget exhaustion takes final-status precedence while preserving every target
  failure and whether the physical target itself was established.
- Only matching independently observed accuracy can produce `verified`.
  Unknown or estimated accuracy remains uncertified.
- Automatic precision requires measured strict cleanup; explicit FP64 does not.
- Invalid API metadata leaves output untouched. Nonfinite actual results remain
  physical failures, not successful verification or a reason for a second solver.

## Evidence and remaining ownership

Native tests exercise inclusive gates, audit maximum/RMS distinction, nonfinite
results, strict cleanup, every accuracy disposition, simultaneous budget/physical
failure, full-width counts and transactional API errors. Python tests deny the
compatibility policy during native decisions and independent PySCF RHF/UHF
Direct/DF progressive endpoints; they cover selected old libraries, refusal,
arbitrary integer metadata, missing force evidence and concurrent decisions.

This moves final scalar policy, not fixed-density Fock/residual assembly,
general coarse-to-fine orchestration or all resource authority. Those remain
explicit #2227 R4/R6 work. No new CUDA scientific admission, kernel or equation is
introduced. Complete controller timing must still include preparation,
projection, both solves, physical verification and cleanup.

The final source passed 121 CPU CTests, 146 Python tests, four focused native
rechecks after formatting, and Sphinx warnings-as-errors. The completed endpoint
campaign retains all eight ABBA/BAAB work/timing comparisons (21 samples and
three warmups per block); its maximum candidate/baseline median ratio is
1.0202265228437473 against the unchanged 1.05 bound. All 1536 independent E/F/D
publications, including warmups, pass 1e-8/1e-6/1e-7 absolute gates. Maximum
errors are 2.731148640577885e-14, 8.287537323070637e-12 and
1.5955570198400437e-11, respectively. Full timing took 85.37 seconds after an
84.5-second pilot estimate; future campaigns should estimate cost before launch.

The [retained publication](../../../../benchmarks/results/native-hf-final-verification-20261026/README.md)
contains every sample, the independent reference, frozen scripts/source patch,
qualification receipts and an offline replay. The benchmark explicitly rejects
the Python final-policy fallback. Its physical audit still uses the existing
Python owner; this result does not close that remaining migration. Production
verification remains `unverified` when independent observed accuracy is absent.

## Revisit when

The fixed-density audit and complete progressive prepared lifecycle have native
owners. Preserve the same acceptance distinctions when connecting those owners;
do not replace the independent audit with an initialization or DIIS residual.
