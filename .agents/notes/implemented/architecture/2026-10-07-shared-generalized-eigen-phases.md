# Decision: share generalized-eigen reduction and recovery

Status: implemented locally; device qualification and publication are separate
Date: 2026-10-07

## Problem

The ordinary symmetric-eigen provider, workspace and CPU LP64 admission were
already shared. GFN and canonical SCF still independently composed the basis
reduction and coefficient recovery around that provider. Moving only another
pointer adapter would leave those equations method-owned.

## Decision

`src/solver/generalized_eigen.hpp` owns separately callable numerical phases for
three explicit representations:

- Canonical X: `temporary = F X`, `reduced = X^T temporary`; recover `C = X U`
- Cached lower L: left/no-transpose TRSM, then right/transpose TRSM; recover with
  left/transpose TRSM
- Identity: a CPU copy or explicit existing CUDA-storage alias

CPU and CUDA lowerings contain tensor primitives. The existing prepared
symmetric-eigen service remains the independently invocable spectral phase and
retains the already-selected provider/workspace binding. No callback returns to
an unchanged method-owned numerical implementation.

The domain distinguishes actual solves, available solve-storage capacity and
physical system cardinality. Spin solves may exceed physical systems; compacted
submissions may be smaller than allocated capacity. Layout and square matrix
extents are explicit. Contiguous bindings validate only the storage needed for
the current phase. Device pointer tables carry their borrowed host-visible
extents; their individual device entries remain owner-prepared and validated.

Real consumers are GFN CPU; GFN CUDA fixed, spin and exact-capacity solve/recovery
bodies; canonical CPU target eigen; ordinary CUDA DF eigen; and CUDA DF RHF/UHF.
GFN's raw batched TRSM adapter is retired. Shared CUDA tensor lowering owns that
submission, and the vendor-debt ledger records the retirement.

## Preserved boundaries

- CPU DPOTRF/DPOCON and CUDA overlap eigenvalue-ratio/POTRF admission remain
  different method policies. Cache seals, geometry epochs and failure scopes stay
  with their owners
- GFN's averaging/validation remains between reduction and spectral execution.
  Compacted graph recovery remains independently submitted after successful-peer
  filtering. No CPU-style read of deferred CUDA `info` was introduced
- Provider family selection, the singleton tridiagonal path, fallback admission,
  thread scopes, stream binding, capture and completion remain unchanged
- Occupations, free energy, density/weighted-density and atomic publication are
  unchanged. No generic SCC host driver is adopted
- Canonical transform scratch is destroyed before the owned scalar solve. A
  phase binding does not retain scratch across that lifetime boundary
- Valid canonical aliases are preserved: CPU `X == F` and CUDA's second GEMM
  writing back over F. The scratch itself stays disjoint
- The GFN CPU n=2 / prepared maximum N=5 solve still binds exactly 81 doubles and
  28 integers. Borrowed transforms allocate nothing and preserve raw `info`

## Evidence

The local source baseline is `6a02e4b184a27301e06fe36762ec0c0c6902fdbf`, with the
same source tree as the reviewed PR2096 head. The new host-call gate compiles
actual production bodies with fake vendor ABI for both NVIDIA and CuMetal
signatures. It compares complete argument/order transcripts and injected first
failures against retained pre-migration bodies, including GFN's real tridiagonal
host path. Canonical source expansion additionally checks unchanged surrounding
upload, trace, solver, download, synchronization and alpha/beta order.

CPU gates include the actual LP64 consumer with independent prescribed spectrum,
generalized residual and metric oracles, scalar target-eigen analytic tests,
phase extent/alias/allocation tests, and inherited fused-publication failures.
An integrated CPU build passes all seven native GFN energy/force comparisons to
the independent tblite fixtures. Its runtime LP64 cohort is the existing
SciPy-prefixed OpenBLAS binary with SHA256
`8fb864c29cac4b25f6e2c139491ea96f2724dde42d51394f84e9c4a622e34790`.

`tests/python/test_gfn_cuda_byte_preservation.py --base 6a02e4b1` compares all
645 embedded device bodies (314 kernels, 331 helpers), ten regenerated science
headers, and the generated density include byte-for-byte. Ordinary tests of the
gate do not require Git history; historical comparison is explicitly opt-in.

These are CPU numerical and GPU host/source preservation results. There is no
GPU numerical, graph replay, complete-endpoint timing or performance claim.
`test_df_eigensystem.cpp` and normal NVIDIA/CuMetal device qualification must
still execute on their supported device lanes before such claims are made.

## Rejected alternatives and revisit conditions

An all-in-one generalized solve would hide GFN's interleaved validation and
successful-peer compaction. A callback to unchanged method-owned solve bodies
would not transfer numerical ownership. Blanket whole-operation alias checks
would reject existing canonical consumers. A rectangular basis or rank-truncated
frame is not inferred from this square contract; add it only with a distinct
domain and independent scientific/lifetime evidence.

The two pre-existing weighted-Gram source-fixture failures observed at the
PR2096 baseline are reported separately and are not repaired in this slice.

## Review follow-up: unused borrowed capacity

Independent review found that the original domain incorrectly applied vendor
integer and active matrix byte bounds to the whole borrowed capacity. GFN's
pointer-table descriptors can include an unused tail larger than those bounds
while submitting a small, valid prefix. Capacity now only bounds the requested
solve count; the vendor integer and matrix byte limits apply to actual solves.
Existing per-phase matrix spans and pointer-prefix requirements remain checked.

Regression probes execute small CPU canonical/Cholesky problems and trace CUDA
canonical/Cholesky phases plus real GFN restricted/spin consumers with capacity
above INT_MAX and with a tail that would overflow a capacity-sized matrix byte
product. Actual solve-count/active-byte overflow still rejects before vendor
submission. These probes fail against the recovered original domain, closing a
gap in its original preservation gate without changing the numerical phases.

## CUDA private-ABI boundary correction

The NVIDIA 12.9 build of PR #2110 exposed a header-boundary regression: the
generalized CUDA phase interface included `tensor/cuda_square_linalg.hpp`, which
imported official `cublas_v2.h` into GFN's translation unit after its independent
`runtime/nvidia_host_api.h` declarations. The duplicate enums and incompatible
status declarations prevented compilation. Existing phase host probes used only
the official-header fixture, so they missed this collision.

The interface now borrows the BLAS handle as `void*` and returns unmodified
`uint32_t` statuses, following the existing symmetric-eigen provider boundary.
Only `solver/cuda/generalized_eigen.cpp` includes the tensor/vendor primitives.
The shared phase composition, validation, actual-solve bounds, raw failures and
owner-side deferred numerical info remain unchanged. The new implementation is
registered in CUDA target sources and the generated solver identity closure;
the runtime source-identity inventory already recursively covers `src`.

The regression compiles the real private NVIDIA header alongside the phase
interface in its own translation unit and calls across that seam into the real
lowering under both NVIDIA and CuMetal host fixtures. It compares complete
success/first-failure traces for canonical, lower-Cholesky and identity phases,
large unused capacities, and invalid bindings. The existing full consumer-body
preservation checks remain in place. Host ABI coverage does not replace the
normal NVIDIA/CuMetal compile and device qualification lanes.
