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

## Revisit when

New compiler layouts, precision variants, provider resource profiles or graph
capabilities invalidate an admission contract. New scientific consumers must use
semantic requests/prepared bindings rather than extending these retained vendor
exceptions with method policy.
