# Decision: finish method/provider ownership without changing qualified algebra

Status: implemented
Date: 2026-11-07

## Problem

The remaining #1890 inventory mixed three different things: actual scientific
consumer submissions, implementation switches in method APIs, and already
reusable provider/lifetime/ABI owners. Moving only the occupied metric Gram
would not resolve the method switches. Conversely, relocating every remaining
vendor token would duplicate legal providers without improving ownership.

## Decision

- DF solver and Lambda method APIs no longer choose GEMM implementations.
  Production admits compiler layouts from their flattened dimensions, complete
  host/device budgets, provider availability and existing bounded fallbacks.
  The admitted occupied/virtual/auxiliary dimensions are retained in the plan
  and consumed when preparing the existing shared Tensor descriptors.
- `DFContractionQualification` is an explicitly diagnostic exclusion interface
  for matched ordered-graph comparisons. Benchmark/probe callers migrate
  together; there are no compatibility aliases for the retired booleans.
  Exclusions cannot promote a vendor, change precision or weaken residual gates.
- `DFPreparedRegions` owns admitted response/audit/primal Q extents independently
  of diagnostics. All execution and fallback decisions consume that state;
  reporting only copies observations out. The independently expanded audit and
  fresh-primal replay keep their own admission, descriptors and bounded scratch.
- RHF's primitive matrix work/roofline/context admission moves from scientific
  policy to `tensor::CudaMatrixProductAdmission`. The original threshold and
  calibration math are retained, not retuned. Resource availability may exclude
  the optional context; it cannot promote one. Existing canonical immutable
  `CudaMatrixProductBinding` requests remain the execution authority.
- `CudaResources` owns matrix create/stream/HOST-pointer/workspace preparation
  alongside its existing teardown. Partial failures retain their original
  owner and native status. Neither device/stream lifetime nor math mode changes.

## Retained-path audit

| Remaining file | Real responsibility and retention decision |
| --- | --- |
| `src/scf/cuda/eigensolver.cpp` | Reusable symmetric-eigen preparation/execution; direct symbols only query linked cuSOLVER version for canonical provenance. Retain provider. |
| `src/scf/cuda/matrix_library.cpp` | Bounded matrix/structured executor plus optional context owner; actual selection uses immutable shared Tensor bindings. Retain provider and its allocation fallback. |
| `src/scf/cuda/resources.cpp` | Prepared context/arena lifetime, explicit status-bearing retirement and legacy destruction. Retain provider/lifetime owner under #933. |
| `src/scf/cuda_eigensolver_policy.hpp` | SDK overload imports and dependent signature adapters delegate contiguous calls to the shared compatibility layer. Retain ABI boundary and exact-stack graph evidence types. |
| `src/scf/cuda_rhf.cpp` | Remaining create-symbol addresses authenticate loaded provider objects for explicit initialization provenance. Retain diagnostic references, not executable LA calls. |
| `src/xtb/native/src/backends/cuda/gfn2_eigensolver.cu` | Bucketed generalized-eigen provider, overlap factorization, tridiagonal alternatives and scoped capture-state restoration. Reduction/recovery and symmetric-eigen submission already use shared solver services. Retain this bounded executor; it contains no occupation/SCC/force-model selection. |
| `src/xtb/native/src/runtime/gfn2_cuda_execution.cu` | Transactional context creation/stream binding/rollback/destruction under #933; numerical work uses the prepared `Gfn2SccSetupEigensolver` binding and charged arenas. Retain infrastructure, not another method selector. |

The vendor manifest records these reviewed contracts rather than a directory
allowlist. The selection manifest retains only one-way diagnostic observations;
its scanner also covers audit/primal flags that the older inventory omitted.
Reintroducing a method-owned implementation flag is rejected even in a formerly
classified migration file. Existing CC/post-HF contractions continue through the
shared canonical Tensor requests; this does not reopen their merged migrations.

## Invariants and rejected alternatives

Keep FP64 publication, independent physical replay, all precision-epoch gates,
resource allowances, partial-failure prefixes and capture/stream identities.
An optional provider allocation miss retains the original ordered schedule;
numerical, launch and driver failures do not trigger a concealed retry.

Merely renaming a boolean or reclassifying scientific submissions was rejected.
So was replacing a reusable GFN eigensystem executor with a duplicate generic
engine solely to reduce manifest counts. Generic joint cost/provider optimization
remains #1889; this change makes no endpoint speedup claim.

## Evidence

Host tests execute the actual providers/planners with injected partial setup,
allocation, descriptor and teardown failures. Independent numerical/oracle and
real-device gates remain required; standalone matrix tests are not molecular
E/F acceptance. The source-matched occupied-Gram PR build at `9c7cf3036` passed
22 Slurm GPU molecular response cases, including cold/warm/moved geometry and
independent PySCF comparisons. The combined final tree requires its own complete
build and endpoint qualification before #1890 can close.

### Combined-source qualification — 2026-11-07

The combined native/compiler source at
`d2f031681ad5e0d22068e0616f56745113fc5a56` now builds successfully on n1
with CUDA 12.9.86, GCC 13.3.0, sm_120 and verified sccache 0.16.0 launchers.
The qualified `libgenerativeqc.so` SHA-256 is
`b19f9e4b91efc703bef96e43fe081070e5f96e808d0f2dcacb184692932f0f37`.
Subsequent commits update test probes/formatting and this receipt only;
`git diff d2f031681..HEAD -- src include python tools cmake CMakeLists.txt`
contains no production or compiler changes. These identities are evidence,
not permanent CI source-fingerprint assertions.

- The final unique 27-file host suite passes **1706 tests**, with 38 explicit
  device/endpoint skips. The four late-discovered cache/admission/upload probe
  consumers are migrated without removing their original failure/lifetime gates.
- Slurm **16711**, visibility `0`: **60 passed** across occupied DF E/F replay,
  RHF resident block response and UHF response. Independent PySCF/dense oracles,
  cold/warm/moved geometry, both spins, bounded capacity, cleanup and transfers
  remain checked; this rerun uses the combined source, not the original Gram DSO.
- Slurm **16709**, visibility `3`: **124 passed, 15 expected skips** across
  native DF solver, Lambda, resident response and complete DF-CCSD(T) force
  endpoints. This includes ordered/prepared comparisons, independent determinant
  and molecular energy-difference oracles, full-coordinate water forces, exact
  budget fallback, tail batches and no publication after failure. The skips are
  CPU parameterizations of CUDA-only schedules, not failed GPU gates.
- Slurm **16708**, visibility `2`: all four native precision-policy,
  stream-eigensolver, Fock-composition and Fock-provider executables pass.
  The extended Python run also passes **140 RHF frame cases**, **26 native CPU
  precision cases** and all seven GFN CUDA molecular oracles. It is **not** an
  all-green run: the baseline failures and CPU prerequisite below are retained.
- Slurm **16717**, visibility `3`: **14 passed** for all GFN CPU/CUDA pinned
  tblite molecular E/F fixtures, including d shells. The CPU half requires the
  installed LP64/local-thread-control provider at
  `/data/jzzeng/portable-integration-20261101/provider/scipy_openblas32/lib` on
  the loader path. The initial seven CPU failures were missing-provider errors;
  setting a CMake-only provider variable at runtime does not configure that DSO.
- All three modified endpoint/qualification benchmark translation units compile
  against the combined generated headers through sccache. This is an interface
  check, not a benchmark or performance improvement claim.
- Vendor, selection, CUDA ownership, SCF dependency and Direct-J/K tuning guards
  pass. The complete changed-file pre-commit run passes after formatting.

#### Unchanged extended-test failures

The expanded run intentionally retains two pre-existing red groups rather than
weakening their assertions or silently reporting a completely green suite:

1. Three nonresident Direct multi-RHS cases fail the unchanged Python
   `device_workspace_bytes` contract. Slurm **16710** reproduces the same three
   failures with the preserved pre-completion Gram-only DSO at `9c7cf30367080d815a9ebe3da8a6d8e43c14a5bc`.
2. Twelve positive-screening/materialized fixed-mask cases fail the validation
   probe's `requested response schedule was not exercised` assertion. Slurm
   **16713** reproduces exactly the same twelve failure identities with that
   baseline DSO; its other 36 fixed-mask cases pass. The numerical/provider
   source and probe responsible for this schedule assertion are not modified by
   this migration.

Neither group is a new numerical discrepancy, and neither is fixed or removed
by this ownership change. They are reported separately from the changed-path
acceptance above. Raw logs, source/library identities, launcher commands and
before/after cache statistics remain in the ignored qualification directory
`/data/jzzeng/qc-1890-gram-20261107/` (the `final-*` receipts and `evidence-gram/`)
and local `build/issue1890/`. All device executions use finite Slurm allocations;
device visibility is preserved. No speedup, approximation change, precision
promotion or new Libxc qualification is claimed.

## Revisit when

New compiler layouts, precision variants, provider resource profiles or graph
capabilities invalidate an admission contract. New scientific consumers must use
semantic requests/prepared bindings rather than extending these retained vendor
exceptions with method policy.
