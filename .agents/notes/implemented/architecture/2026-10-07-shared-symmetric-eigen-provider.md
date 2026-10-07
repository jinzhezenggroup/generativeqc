# Decision: share symmetric-eigen provider submission across GFN2 and SCF

Status: implemented
Date: 2026-10-07

## Problem

The embedded GFN2 eigensolver and the SCF/DF CUDA services independently formatted
and submitted the same FP64 cuSOLVER symmetric-eigen queries and solves. Merely
moving GFN2's directory would retain that duplicated ownership. Calling the whole
SCF dispatcher from GFN2 would instead add mask/sanitization work and force vector
mode, changing the existing execution contract.

CPU services are also not interchangeable: GFN2 uses a host-isolated LP64 backend,
column-major work APIs, caller-owned allocation-free scratch, Cholesky transforms,
condition checks, and transactional publication. The current CPU HF target is a
fixed scalar solve with its own orthogonalization and validation. None of those
CPU numerical/provider contracts changes here.

## Decision and deletion boundary

One shared host-only lowering in `src/solver/cuda/symmetric_eigen_provider.*` now
owns symmetric-provider query and submission for these six actual consumers:

- GFN2 `gfn2_eigensolver.cu`
- Shared SCF `cuda/eigensolver.cpp`
- RHF/UHF `cuda_rhf.cpp` setup
- DF `cuda/df_scf_library.cpp`
- Ordinary DF `cuda/df_eigensystem.cpp` setup
- DF metric `cuda/df_plan_setup.cpp`

The six consumers delete 18 direct provider call sites. The common implementation
owns Jacobi-batched and generic-batched submission, generic single-matrix
serialization, FP64/lower-triangle/leading-dimension formatting, and the raw
provider status return. It is used in production by both GFN2 and HF/KS/DF; it is
not a forwarding header around duplicated method-local dispatch bodies.

The existing NVIDIA contiguous/CuMetal explicit-stride signature adapter moves
unchanged, except namespace, to `solver/cuda/cusolver_compat.hpp`. SCF retains its
compatibility aliases for remaining diagnostic callers. This avoids a new shared
solver-to-SCF reverse dependency. The independent exact-stack graph qualification
probe keeps its direct calls as diagnostic evidence.

The shared resource/request header contains no CUDA/vendor declarations or
method statuses. Opaque borrowed handles cross the boundary as `void*`; the sole
provider implementation casts them to the official provider types. This avoids
including official `cusolverDn.h` and GFN2's private `uint32_t` status ABI in one
translation unit. Raw 32-bit statuses are returned without method translation.
The private ABI itself remains unchanged.

## Preserved contracts

- Caller-owned family selection, native small/maximum-pivot routes, GFN2
  tridiagonal route, capture eligibility and exact-stack qualification
- Caller-owned resources, stream binding, allocation/lifetime, masks, sticky
  errors, generalized transforms, scientific validation and publication
- GFN2 value-only versus vector mode; mode-major/capacity-minor setup queries,
  every reachable exact capacity, non-monotonic workspace maxima, and no partial
  requirement publication on a failed query
- Existing Jacobi element counts versus generic byte counts; callers retain
  their original validation/conversion and pass the exact previously selected
  work count, rather than deriving new work sizing inside the provider
- Generic dimensions remain int64; Jacobi narrows to the same original int API
- Serialized Xsyevd matrix/value/info offsets and stopping on first provider
  failure, with no added stream or device calls
- SCF profile/sanitization order and failure mapping, DF trace boundaries, and
  independent GFN2/SCF status mapping

Unknown internal provider families fail with raw invalid-value status without
submission or output mutation. This is an internal fail-closed guard, not an
alternate algorithm or provider switch.

## Identity and qualification evidence

Base: `89978df757a78559364170bb8d77d07e6b1a4f61` (same source tree as the first
workspace-sharing slice's published head). This is a stacked continuation of
#2063; its independent stale include-root test repair is also carried forward.

- The host test suite compiles the actual lowerer under official opaque-pointer,
  enum-status and CuMetal explicit-stride-only stubs: 71 tests passed
- Verbatim production GFN2, SCF and DF host adapter bodies run against call traces,
  including both modes, failure at each capacity, non-monotonic workspace sizes,
  existing maxima, invalid Jacobi scratch, tridiagonal/native routing, mask and
  profile order, raw/method status mapping, int64 dimensions and invalid family
- A separate translation unit includes the real GFN2 private NVIDIA ABI header,
  then calls the real shared implementation compiled with the official/CuMetal
  stub ABI. All families preserve borrowed handle identities and raw failures
- Fresh cached CPU build and all 75 native CTests passed
- CPU GFN2 endpoint, independent tblite/xTB fixtures and retained runtime lifecycle:
  20 passed; 13 CUDA opt-in cases skipped
- Source comparison proves all 26 GFN2 kernel bodies and every byte outside the
  three migrated GFN2 host functions are unchanged after removing the new include
  and namespace alias; ten generated GFN2 scientific headers are byte-identical
- The ordinary solver metadata header changes only SHA-256 identity strings.
  It must change: its source-identity closure now includes the new shared provider
  and compatibility files, and is protected in both generator and CMake inputs
- Shared dependency, SCF dependency, vendor ownership, provider-selector and CUDA
  ownership inventory checks remain enforced; only reviewed retired references
  and the exact new shared contract are updated

Host tracing is not CUDA compilation, numerical GPU validation, or graph replay.
This environment has no CUDA compiler or device. Normal NVIDIA/CuMetal build
checks remain required before considering the branch qualified for integration.
No numerical implementation, work schedule, reduction order, or provider choice
changes; no performance improvement is claimed or new performance campaign needed.

## Remaining work

Generalized-eigen semantic request/prepared-provider migration and CPU provider
ownership remain under #1240/#1890. Density/weighted-density and Broyden follow-ons
remain under #1879/#1882. Method policy and persistent resource/graph owners still
exist in the embedded runtime. They require real consumer cutovers before that
runtime can be deleted; this slice does not claim wholesale retirement.

## Prepared handle ownership follow-up

The subsequent `2026-10-07-shared-prepared-eigen-handles.md` cutover transfers
provider handle/descriptor lifetime out of the five method owners. Numeric
workspace, generalized transforms and method graph/state ownership remain with
the original owners; the query/submission contract above is unchanged.
