# Decision: consolidate reusable xTB infrastructure under shared owners

Status: implemented
Date: 2026-10-15

## Problem

After the shared CUDA device-scope migration, the method still carried duplicate
NVIDIA ABI declarations, shared LP64 wheel packaging, generic CPU execution
interfaces and FP64 synchronization support. An unused device resolver and CPU
feature policy obscured which execution paths were actually active.

## Decision

Use the shared runtime NVIDIA host API across xTB and other consumers. Extend
the SDK generator's live-call inventory to xTB and regenerate with CUDA 12.9.
Select only the actual exported C-linkage declarations: real SDK headers also
contain static C++ convenience overloads such as `cublasGemmEx`, which are not
provider ABI alternatives. Conflicting exported signatures still fail closed.
Keep driver allocation-query declarations in a separate shared minimal header.
Preserve unsigned 32-bit raw provider status transport, opaque handle layouts,
graph minimum-version admission and all runtime fallback checks.

Move LP64 shim source and packaging configuration to the shared tensor/CMake
owners. Retain the installed shim SONAME, configuration options/macros and
reviewed input resolver paths as compatibility contracts. The shim alone has a
provider `DT_NEEDED` edge; the main runtime retains isolated lazy loading. This
is not permission to replace provider provenance verification with a test double.

Move only the CPU kernel identity and borrowed synchronous chunk interface to
`runtime/cpu_isa.hpp` and `runtime/parallel_executor.hpp`. Remove the unused
feature/environment selector rather than activating it. CPU execution remains
baseline and serial unless an existing explicitly supplied executor is present;
the interface itself creates no pool or workers. Retire the unused private CUDA
device resolver without changing live descriptor admission.

Move FP64 atomic addition to `runtime/cuda_atomic.cuh`. Retain native sm_60+
addition, integer-CAS fallback, return-old semantics and NaN-safe integer retry.
Rename its unconsumed internal forced-CAS qualification macro to
`GENERATIVEQC_FORCE_FP64_ATOMIC_CAS`. This is ownership consolidation, not a
precision or scientific reduction change.

## Rejected alternatives

- Do not share the handwritten method ABI by copying its declarations into a
  second generated inventory. The real SDK remains the function-signature oracle.
- Do not choose the last SDK overload or suppress conflicting signatures. A
  C++ wrapper may have a different parameter type from the exported C symbol.
- Do not rename installed wheel siblings just to remove historical method names:
  that changes loader/repair contracts without improving runtime ownership.
- Do not activate AVX/FMA, create a pool, change summation or delete scientific
  coverage as part of an infrastructure move.

## Invariants

Shared runtime/tensor owners do not depend on method headers. Public ABI,
scientific/generated parameter identities, upstream attribution and independent
oracles remain intact. Removed imported files retain their upstream hashes and
retirement reasons in `CUDA_SOURCE_PROVENANCE.json`; current vendored hashes
remain enforced. No performance gain or real-GPU qualification is implied.

## Evidence

Qualification uses a fresh CPU CMake build with the verified ccache launcher,
independent checked-in tblite molecular energy/force fixtures, frozen spectral
oracles, loader isolation/error-injection tests and executable packaging probes.
The packaging probe builds/installs the production CMake helper and shim against
a synthetic provider and checks SONAME, `DT_NEEDED`, exports, installed RPATH,
custom-provider fallback and non-wheel rejection.

The host-only runtime primitive probe checks synchronous/disjoint chunk outputs,
native/fallback FP64 return-old behavior, NaN/infinity, and an injected CAS race.
It is not a GPU memory-model test. CUDA 12.9 compilation and device linking of
the complete GFN2 CUDA archive, plus native/forced-CAS compile probes, qualify
the changed includes and device code. Real-device numerical/atomic validation
requires Slurm; the available GPU node was drained during this qualification.

Raw build/test logs and compile artifacts are ignored local data under
`build-cuda/qualify-xtb-infrastructure/`; they are not published release assets.

The bounded host regression run passed 607 tests with 42 skips, including the
independent tblite and spectral gates, provider ABI/status tests, device-scope
failure injection and shared resource-lifetime suites. SDK regeneration/check
used an isolated Clang 14 toolchain and actual CUDA 12.9 headers. The real
reviewed scipy-openblas32 input also passed production CMake resolver/shim
configuration and ELF dependency inspection. Sphinx completed with warnings as
errors. CUDA ownership inventory covers 362 files. A one-off token comparison
confirmed ownership-only transformations in 24 live xTB files and preservation
of the atomic function body after its internal macro rename; this is not a new
permanent source-shape/fingerprint test.

## Revisit when

A new consumer needs automatic ISA dispatch, persistent worker ownership, a
different provider ABI, or deterministic/compensated accumulation. Each is a
separate behavioral change requiring independent acceptance gates.

## References

- [Shared CUDA device-scope decision](2026-10-15-shared-xtb-cuda-device-scope.md)
- [Current native ownership](../../../../docs/developer/xtb_native_ownership.md)
- [Generated host API](../../../../docs/maintainer/generated_cuda_host_api.md)
