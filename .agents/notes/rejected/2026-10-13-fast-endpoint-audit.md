# Decision: withdraw the unqualified Fast audit preset

Status: rejected
Date: 2026-10-13

The unavailable-Fast API is superseded by the
[single strict endpoint and exact source-reuse decision](../implemented/performance/2026-10-13-small-ks-reuse-fast.md).
The rejected double-solve design and its historical measurements remain unchanged.

The subsequent [budget-selected FP32 XC decision](../implemented/numerics/2026-10-13-budget-selected-fp32-xc.md)
replaces the temporary strict-FP64 alias with genuine mixed arithmetic. Explicit
target requests retain the full audit; qualification is not relabeled as a
per-call observable measurement.

## Problem

The initial application-precision facade labeled an AUTO candidate followed by
a complete independent FP64 solve as `fast`. The extra solve includes SCF and
forces, so it is not merely a cheap endpoint residual check. Independent
PBE0/def2-SVP measurements on an RTX 5090 show total times of 5.002/15.513 s
against ordinary FP64's 2.161/6.125 s for water/tetramer. AUTO itself takes
2.771/8.901 s. Lowering only J recurrence neither removes the strict K and
gradient work nor compensates for the additional iterations and audit solve.

The user correctly rejected this as an acceleration implementation. A caveat
that a speed-oriented preset has no speed promise does not resolve that mismatch.

## Decision

Withdraw `fast` before native construction; reject all case variants with
`NotImplementedError`, including requests with custom target allowances.
Retain the old numerical experiment only under the explicit `audit_auto`
profile. Document the entire facade as an experimental audit utility, not a
production accelerator. Balanced and Strict still describe audit allowances,
not throughput guarantees. Ordinary Calculator execution remains unchanged.

This corrects the public claim, not the runtime performance. No faster kernel,
cheaper verification algorithm or new speed measurement is delivered here.
Do not describe the name change as an optimization.

## Rejected alternatives

- Silently aliasing `fast` to the audit preserves the misleading contract.
- Dropping the second solve and claiming the same observed error-budget
  evidence would turn an unmeasured candidate into a falsely validated result.
- Loosening the model/grid or excluding audit time manufactures a speed ratio
  for a different scientific or execution endpoint.
- Treating one negative implementation as evidence against FP32 ignores the
  fact that K, XC and gradients were not lowered in this experiment.

## Evidence and invariants

The original failed/negative samples and their source identities in
`benchmarks/results/pbe0-precision-20261013/` stay unchanged. The runnable
benchmark uses schema v2 and the explicit `audit_auto` mode; it must not relabel
the historical v1 Fast records as a new result.

Tests exercise rejection before any backend work and preserve the explicit
audit's independent observable gates, work accounting, bounded fallback,
backend fidelity and native numerical references. No new GPU performance claim
is made by this API correction.

## Revisit when

A genuine accelerated candidate lowers the expensive J/K, gradient-integral or
XC contractions under independently measured error budgets. Its complete
endpoint, including all required refinement, verification and fallback work,
must outperform FP64 in its admitted workload domain. Verification should reuse
the converged state or qualified component evidence where scientifically valid,
rather than require an unconditional second full SCF as the acceleration path.
Any cheaper check needs its own scientific validation; omitting an audit is not
itself proof that an application error budget is met.

This supersedes the Fast selection in
`../implemented/numerics/2026-10-13-application-precision-audits.md`, not the
audit machinery or its historical measurements.
