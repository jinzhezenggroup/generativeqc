# Decision: defer eligible replay anchors by default

Status: implemented in the candidate; A100 requalification pending
Date: 2026-11-01

## Problem

A strict fitted range-separated solve can retain an audited range source for
warm replay while eagerly computing a second energy anchor for the incumbent
provider. The retained source often serves every subsequent call, making this
extra cold J/K action unused. The earlier opt-in experiment isolated about
12 seconds of this work in the A100 water96/WB97M-V endpoint.

## Decision

An unset `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY` now selects the existing guarded
deferral, like explicit `1`. Explicit `0` and other values preserve eager work.
Admission still requires strict FP64 fitted RKS, an exact factor-backed seed,
device-fused XC, and an audited source that is eligible for replay. The real
additional force-allocation envelope must fit currently free device memory.
There is no GPU-name or architecture whitelist.

No recurrence, precision, convergence threshold, public tile, source admission
budget, or independent reference changes. The frozen density's own XC scalars
and AO factor remain the authority for fallback reconstruction. Unexpected
source loss restores that incumbent anchor in a counted preparation Fock action;
force-memory eviction itself preserves the current stationary lease.

## Rejected alternative

Unconditional deferral moved the saved cold action into the first warm call on
the 5090, making that call 26.37% slower even though pooled warm timing hid the
cost. Retain the force-envelope guard and inspect each warm observation plus
the full cold/three-warm lifecycle. The earlier result is negative evidence,
not a measurement of this candidate.

## Invariants

- Default, explicit deferral and eager owners must independently pass energy
  and full analytic-force gates, including frozen seeds and invalidation.
- Preserve first-warm and lifecycle work counts. No preparation Fock is free.
- Imported seeds cannot borrow a former owner's deferred energy or source lease.
- Driver errors fail explicitly; lack of force headroom keeps eager preparation.
- Qualification on particular devices does not prove performance on every GPU.

## Evidence and limits

The candidate's native source identity is
`6be420fb4ee14175922194d61b41e3a20f13665968bc012b474b21cada96f4cc`.
The base commit alone does not contain the change; retain its dirty-source
reconstruction and measured binary identities with the receipts.

85 host checks pass. Both RTX 5090 and RTX PRO 6000 pass 66 scheduled GPU cases,
including explicit and unset-default frozen seeds, policy/capacity/factor
invalidation, eviction, failed begin/retry, exports/imports, residency tiers and
moved geometry. Independent small-system GPU4PySCF references retain the original
energy/full-force limits. PRO6000 cold latency improves 5.999% and the complete
cold/three-warm lifecycle improves 3.793%; 5090 is unchanged within 0.016%.
Every warm call remains one iteration / one Fock. The larger 108-atom 5090
control completes both six-call lifecycles with matching work and paired errors
below 1.37e-12 Eh and 6.76e-13 Eh/Bohr.

Native-only memcheck passes five cases with zero errors/leaks against separately
produced, authenticated independent GPU4PySCF observables. Combined instrumented
reference/native execution still fails; a reference-only process reproduces the
retained allocations without constructing a native owner. Explicit fixture GC
and pool cleanup did not fix this instrumentation interaction. Its root cause
remains unresolved, and all negative results remain in the evidence package.

A100 SSH currently resets before authentication, preventing this exact source's
requalification and recovery of previously submitted allocations. Earlier
opt-in measurements retain their original identities; they must not be relabelled
as this candidate. This change does not establish all-functional parity with
GPU4PySCF. The separate range-source construction cost remains an optimization
candidate, not a delivered improvement.

## Revisit when

A complete endpoint exposes first-warm work transfer despite the force envelope,
new consumer allocations invalidate its resource estimate, or a new owner or
precision contract needs explicit scientific qualification.

## References

- Supersedes the default-off decision in
  [the guarded opt-in proposal](../../proposed/2026-11-01-deferred-range-warm-anchor.md).
- Current behavior: [native XC](../../../../docs/developer/xc_native_cuda.md).
- Source-matched evidence: [default-anchor controls](../../../../benchmarks/results/range-anchor-default-20261101/README.md).
