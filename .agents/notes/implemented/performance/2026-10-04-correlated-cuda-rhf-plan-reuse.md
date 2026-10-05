# Decision: retain CUDA RHF executable plans across correlated prepared replay

Status: implemented
Date: 2026-10-04
Agent: dot

## Problem

Prepared MP2, RCCSD and RCCSD(T) already retain a validated HF density through
the shared correlated warm-reference lifecycle.  Their conventional CUDA
reference, however, entered through the single-system RHF adapter, which built
and destroyed a `CudaRhfBucketPlan` on every endpoint.  Same-topology replay
therefore rebuilt the arena, Graphs, provider handles, topology/schedule state
and optional reference-ERI residency before correlation started.

## Decision

Give each conventional-CUDA correlated prepared owner a separate RAII owner for
the existing RHF bucket plan.  The single-system adapter now has a cached form
that delegates to `run_rhf_cuda_bucket_cached`; compatibility remains entirely
owned by the SCF bucket contract.  Geometry-bound prepared method owners transfer
that executable plan to their replacement owner before a same-topology
coordinate replay.

Warm density and executable reuse remain independent.  A retained executable
plan never proves that a prior density or physical reference is valid, so every
correlated endpoint still executes and validates RHF at the requested geometry.
A failed/nonconverged cached RHF attempt destroys the executable plan before a
cold retry, preventing partially advanced executable state from being published.

The plan's live device bytes are subtracted from the correlation-phase budget
and added back to complete endpoint capacity/peak diagnostics.  The public
correlation diagnostic reports executable reuse and retained-plan bytes
separately from the existing batch warm-state flags.

## Invariants

- CUDA bucket compatibility remains the single source of truth for
  topology/basis, device, provider, precision, screening, reference policy and
  execution-schedule identity.
- Changed coordinates reuse immutable topology/schedule state but still rebuild
  geometry-dependent state and perform full RHF convergence.
- Scientific warm state is committed only after the complete correlated
  endpoint succeeds.
- A failed or nonconverged cached RHF attempt cannot leave a mutated executable
  plan live for the next endpoint.
- Retained RHF plan bytes coexist with and are charged beside MP2/CC/triples/
  response work; they are not folded into correlation-owned-device telemetry.
- CPU and density-fitted reference paths do not advertise CUDA plan reuse.

## Evidence

The native CUDA reference-residency test now checks first-run, stationary replay
and same-topology changed-geometry reuse.  Public MP2/RCCSD/RCCSD(T) CUDA batch
tests check that executable reuse is reported independently of warm-density
reuse, including a warm-disabled replay.  The host source-admission fixtures
cover the new cached-reference signature without requiring a GPU.

`benchmarks/correlated_reference_plan_reuse.py` times complete correlated
execute calls with warm-density reuse disabled.  It records cold, stationary
replay and changed-geometry samples plus the retained-plan diagnostics.  No
speedup claim is made until an allocated-GPU run is retained.

## References

- #1856
- #1503
- #1696
- #933
- #926

## Subsequent capacity correction

The device-only downstream reservation and exact-budget compatibility described
above are superseded by
[complete retained capacity and bounded retirement](2026-10-04-correlated-plan-capacity-fallback.md).
The independent scientific warm-state and executable-owner lifetimes remain.
