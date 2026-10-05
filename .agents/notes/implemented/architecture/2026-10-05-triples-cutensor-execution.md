# Decision: execute optional cuTENSOR through the existing triples W region

Status: implemented
Date: 2026-10-05

## Problem

The shared affine executor was callable in isolation, but the existing triples
consumer assumed exactly two providers through candidate-index parity. Its host
budget also assumed provider-independent descriptor storage. A third provider
must preserve the canonical W request and the complete scientific precision gate.

## Decision

Extend the existing portfolio to strict/mixed arithmetic across cuBLAS, generated
CUDA and cuTENSOR. Resolve algorithms from provider metadata and precision from
the emitted precision identity/index, never candidate ordinal arithmetic. Keep
panel work FP64, W casts explicit, W combination FP64 and every denominator,
epilogue and final reduction FP64. The original mixed qualification remains
`issue1764/df-triples-w-fp32-candidate-v1`.

The generated WExecution prepares one panel and two W plans, then replays them
through PreparedContractions. Charge all three workspace/provider ceilings and
qualified host reservations simultaneously before allocating the method arena.
Generated fallback is a subset of this already admitted storage. A rejection of
the third plan drains both the provisional W plans and the completed panel table
before the single generated retry. Execution and cleanup errors remain fatal.
The complete budget retains the initial admission after a preparation rejection,
because those provisional provider plans contributed to the endpoint's peak;
selected host/provider diagnostics describe the eventual executor.

Build availability supplies no production resource profile. The provider layer
currently returns an empty qualification; native selection records that rejection
and preserves the existing incumbent. Test builds alone inject synthetic resource
limits and complete ranking costs to exercise the production method path. These
costs are not timings, and the synthetic limits are not production defaults.

The validation probe gains a versioned ABI and diagnostic capacity. Inspection
found the older benchmark caller still used the pre-precision argument list and
a 14-entry output buffer although the probe had grown. Both current callers now
use the versioned signature and a checked 26-entry diagnostic publication, so
stale probe binaries fail lookup instead of misinterpreting pointers.

## Evidence

On n1 RTX 5090 / CUDA 12.9 / cuTENSOR runtime 20800 through finite Slurm jobs,
34 enabled-provider tests and 29 provider-absent tests pass (five optional cases
skip in the latter build). The enabled suite passes the pinned H2O/NH3/CH4 energies
in strict and mixed precision. Strict acceptance remains 3e-12 absolute/relative;
the mixed gate remains 2e-7 absolute plus 2e-4 relative. Random physical Gram
inputs verify original semantic summands, casts, provider/version and admission.

Repeated o=3/v=4/q=5 complete calls each perform 27,840 contraction summands.
The first measured strict call took 19.05 ms, subsequent calls 18.60/18.64 ms;
mixed calls took 19.21/18.63/18.89 ms. Each timing includes preparation, H2D,
panel/W/epilogue/reduction execution, D2H, synchronization and cleanup. These are
process-warm observations after earlier tests, not cold-process qualification or
a comparison proving speedup. Per-call records and cache statistics remain in
ignored `.artifacts/1887-triples-cutensor/`.

Budget-constrained selection and injected third-plan rejection reproduce the
same-precision generated result and work counts. The provider-absent CUDA build
also executes the existing strict/mixed/generated endpoint suite.

## Rejected alternatives and remaining scope

Do not assign test resource limits to all production shapes. Opaque host storage,
lazy device allocation, OOM, full cold/warm phase costs and another independent
method family still need qualification in #1887/#1889. Likewise, do not expose a
vendor Boolean on the CC API or claim completion of #1886 based on this slice.
Capture remains rejected; response/Fock callbacks remain separate #1890 work.

## References

- [Original joint triples lowering](2026-10-05-triples-joint-execution.md)
- [Shared cuTENSOR preparation](2026-10-05-shared-cutensor-preparation.md)
