# Decision: share prepared symmetric-eigen provider handle ownership

Status: implemented
Date: 2026-10-07

## Problem

The shared query/submission lowerer introduced in #2070 still borrowed handles
whose lifecycle was independently implemented by the embedded GFN2 context, the
RHF/UHF CUDA bucket, ordinary-stream KS, the DF metric/final plan, and the DF
persistent SCF solver. Each retained raw solver/Params/Jacobi ownership and its
own creation/configuration/destruction logic. Renaming the embedded directory or
forwarding its setup calls would not remove this duplicated ownership.

## Decision

`solver/cuda/symmetric_eigen_handles.{hpp,cpp}` owns the solver handle and optional
Params/Jacobi descriptors in a move-only prepared object. The five real owners now
retain this shared object, rather than separately owning those vendor resources.
The existing symmetric-eigen query/submission lowerer consumes its borrowed view.
There is no additional numerical implementation, provider selection or new GPU work.

Preparation is deliberately staged. GFN2 creates solver, Params, Jacobi, configures
machine-epsilon tolerance / 100 sweeps / sorted values, creates BLAS, then binds
both handles to its stream. RHF/DF create solver, bind it, then configure Jacobi
with 1e-13 / 100 / sorted values or create Params. Ordinary KS and the DF metric
plan retain solver / stream bind / Params ordering. Each method preserves its
original failure mapping and publication boundary; the shared object returns the
raw 32-bit provider status and stops Jacobi configuration at its first failure.

The shared header contains no NVIDIA or CuMetal types and no method status ABI.
This preserves GFN2's independent uint32 provider declarations across a separately
compiled translation unit using the official opaque-pointer/enum-status ABI.
Acquiring an already owned resource or configuring without a solver is rejected
without a vendor call. A partial owner retains acquired resources until reset;
retry after reset and move transfer cannot duplicate ownership.

## Lifetime and resource invariants

The enclosing owner still owns the CUDA device, stream and numeric allocations.
It explicitly resets shared handles at the original destruction position:

- GFN2: select owner device, drain if needed, destroy prepared state and BLAS,
  reset eigen handles, restore caller device
- RHF/UHF bucket: select owner device, reset eigen handles, destroy BLAS, queue
  original async numeric frees, drain/destroy stream, free host workspace
- Ordinary KS: select owner device, drain stream, free numeric workspace, reset
  eigen handles, restore caller device
- DF persistent solver: free its device/host workspace, reset eigen handles;
  persistent state destruction has already selected the owning device
- DF plan: release subordinate solver/state/frames and numeric storage, reset
  eigen handles, then destroy BLAS/stream and clear plan state

The shared object's automatic destructor is consequently empty after those
explicit resets, including after constructor cleanup and after device restoration.
Move assignment requires the same settled, owning-device precondition as reset
for its destination. It destroys a live destination exactly once before transfer.

Numeric workspaces intentionally retain their existing owners: GFN2's packed
iteration arena and pinned host storage; RHF's stream-ordered resource ledger
allocation also borrowed by cuBLAS; KS's synchronous resource allocation/vector;
and DF's synchronous allocations. Conflating these in one allocator would change
memory accounting, synchronization or allocator semantics. All capacities,
allocations, queries, masks, info buffers, graph eligibility, stream work, and
per-system result/failure handling remain in their existing services.

The DF plan's shared handle object adds one optional descriptor pointer. Its
existing `sizeof(*candidate)` host accounting automatically includes this actual
metadata. KS retains the existing 16 KiB metadata allowance and compile-time size
check. Numeric admission allowances and device/host workspace sizes do not change.

## Rejected alternatives

- One monolithic initialization function: changes GFN2's BLAS interleave or
  scientific Jacobi tolerance, and hides setup error boundaries
- Shared owner that automatically selects devices or drains streams: duplicates
  enclosing lifetime responsibility and changes teardown work/order
- Replacing numeric arenas with one generic workspace owner: loses the existing
  pinned, async-pool, cuBLAS-sharing and method-specific admission contracts
- CPU/generalized transform unification: the current LP64 xTB bridge, Cholesky
  generalized transform, Gaussian target eigenframes and overlap validation are
  distinct scientific contracts, not incidental handle lifecycle

## Qualification

Base: master `0ad23e7919ac632b4ecb680cec409198dc0343d4` (#2070).

- Fresh cached CPU build and all 76 native CTests pass
- Host probes compile the actual shared owner and query/submission lowerer with
  official opaque-pointer/enum-status and CuMetal explicit-stride fixture ABIs;
  a separate TU uses the real GFN2 private ABI
- 182 lifecycle/provider probe cases pass across the two fixture ABIs. Verbatim
  GFN2 setup/context teardown and SCF teardown bodies cover the BLAS interleave,
  all nine GFN2 setup failure points, exact error text, retry, reuse, selected
  device/restoration, and no late automatic-destruction vendor calls
- Lifecycle traces cover exact settings, partial failures and raw statuses,
  reset/retry, duplicate/prerequisite rejection, move construction/assignment,
  source-owner retirement and borrowed view identity
- Existing provider probes continue to protect exact-capacity/mode queries,
  non-monotonic workspace maxima, integer widths, per-item serialized failures,
  and GFN2/SCF/DF submission/mask behavior
- Vendor ownership inventory retires 43 method-local lifecycle references and
  classifies ten references in the one shared owner; vendor, shared-architecture
  and provider-selection checks pass
- CPU GFN2 endpoint/oracle tests: 13 passed, 10 CUDA/optional-provider skips
- Every one of the 13 device functions in the only modified CUDA source,
  `gfn2_cuda_execution.cu`, is byte-identical to the base; every other .cu/.cuh
  source is unchanged
- Ten generated GFN2 scientific headers are byte-identical. The solver metadata
  header changes only SHA-256 identities, with both new owner files included in
  the generator and CMake dependency closure

These are host-call/source-invariance checks, not GPU execution qualification.
The authoring environment has no CUDA compiler/device. Normal NVIDIA and CuMetal
compile/runtime CI remain necessary integration gates. This slice changes neither
scientific work nor schedule and makes no performance claim or new benchmark
campaign. Retained xTBloom attribution and scoped linking permission are preserved
in the shared implementation notice and `THIRD_PARTY_NOTICES.md`.

## Remaining work

This is a real prepared provider-handle ownership cutover, not complete embedded
runtime removal. GFN2 still owns generalized-overlap setup/cache provenance,
method-specific arena binding, validation and graph/device-tail orchestration.
The next semantic/resource migration must account for those contracts and actual
Gaussian consumers, or proceed to the common P/W tensor provider under #1879.
Johnson Broyden remains distinct from DIIS (#1882). CPU ownership remains under
#1240/#1890. None of those algorithms or state machines changes here.

## References

- #1240, #1890, #933/#934: migration and solver ownership
- #2063: shared bounded workspace helpers
- #2070 and `2026-10-07-shared-symmetric-eigen-provider.md`: shared submissions
- #1814/#1879, #1882: density and mixing ownership boundaries
