# Decision: shared prepared CPU spectral resources and GFN electronic adapter

Status: implemented; CPU qualified, CUDA source unchanged
Date: 2026-10-08
Base: 5c02b1ba3c69fbee712697565dcb79da6b768606

## Problem

CPU primitive loading, ordinary symmetric eigensolution and generalized phases
already have shared owners. The embedded GFN eigensolver still owned immutable
ragged dimensions, resource planning, overlap-cache binding and transactional
factorization alongside occupation and electronic-result policy. Relocating all
of that code into a generic namespace would hide method policy instead of
removing the dependency.

## Decision and retirement ledger

`solver/cpu/prepared_spectral.*` owns a method-free prepared spectral service.
Its inputs are orbital partitions, caller-selected conditioning threshold/status
encoding, and borrowed matrix/cache/work storage. Its immutable metadata owns
exact LP64 resource counts, packed overlap-cache offsets and plan identity.
Factorization and single-spectrum execution reuse the existing shared LP64,
ordinary-eigen and generalized-phase services.

`methods/gfn2_electronic_update.cpp` is the actual method consumer. It retains
electronic admission, five-field layout, the combined numerical arena, spin
packing, both per-spin CPU occupation solves, weighted densities, thermodynamics
and staged result publication. The former
`src/xtb/native/src/model/gfn2/eigensolver.cpp` (1,872 baseline lines) is retired
from production and build registration. Much of that file remains as the method
adapter; this is not a claim of 1,872 generic lines or net project lines deleted.
The new shared implementation/header and method adapter total 2,543 lines,
versus the retired 1,872-line implementation: the production source delta is
+671 lines before tests/documentation/build wiring. The shared owner is 633
implementation lines plus a 181-line API; the method adapter is 1,729 lines.
The legacy `eigensolver.hpp` remains byte-identical for CPU consumers and CUDA
HostPlans. The occupation policy header remains method-owned and unchanged.

## Admission and transaction contracts

The shared matrix admission phase validates the complete requested ragged batch
without allocating or invoking a provider. Its opaque token binds a live plan,
input extent, system range and immutable borrowed matrix-offset/multiplicity
metadata. Retain a plan copy and all borrowed storage until the last execution.
Admission rejects token storage aliasing inputs, metadata or plan. Execution
rejects writable cache/scratch aliasing the token, any admitted input matrix or
layout metadata. Convenience APIs preserve structural/cache admission before
reading matrix values, including stale-factor status precedence.

GFN admits each overlap/Hamiltonian exactly once before entering its existing
sequential BLAS scope, then calls admitted execution phases. A first refactor
that repeated the finite/symmetry checks inside the shared service was rejected:
it added complete matrix scans and moved those scans into the thread scope.
Instrumented frozen-consumer tests protect both counts and placement.

Overlap backend failure preserves the entire committed batch cache. Numerical
failure commits that member's generation/status but retains its old factor
bytes; successful peers commit. Any nonzero caller stamp remains valid,
including repeated or decreasing stamps. `runtime::AsyncGeneration` is not an
appropriate substitute: its revocation/monotonic rules differ. Scientific
invalidation and SCC result reuse remain method responsibilities.

## Storage, work and numerical invariants

- Cache/worker/full-batch byte sizes, offsets and 64-byte alignment are unchanged.
  Successful transitions allocate no numerical storage and never replace buffers.
- Plan setup adds one immutable shared metadata allocation. Resident accounting
  includes both metadata payloads and their vector capacities; as before it does
  not estimate allocator/control-block overhead.
- Plan copies retain O(1) identity. Failed construction preserves the previous
  plan; partial bindings and result batches are never published.
- Each spectrum retains left/no-transpose TRSM, right/transpose TRSM, explicit
  symmetrization, DSYEVD and left/transpose recovery, with maximum-plan work
  counts even for smaller systems. Failed eigensolution suppresses recovery.
- GFN retains all occupation root/fallback arithmetic, two CPU occupation solves
  even for equal restricted populations, ordered weighted-density/free-energy
  arithmetic, numerical status translation and per-system/whole-call publication.
- Shared plan preparation performs no provider loading, preflight, workspace
  queries or numerical execution, including when called by CUDA HostPlans.
- CUDA sources, compatibility headers and ownership/provenance manifests remain
  unchanged. This slice makes no GPU numerical/performance qualification claim.

## Evidence

- Final focused selection: 189 passed, 3 explicit CUDA/NVCC qualification skips.
  This includes 12 frozen differential cases, 8 method-free resource/token cases,
  existing real LP64/Gaussian numerical and method consumer gates, source/binding
  fingerprints and CMake/ownership boundary tests.
- Release CPU library/CLI build passed with verified ccache 4.14.1, preserving
  the existing shared cache. All 78 native CTest cases passed. Unrelated stationary
  CPU force AOT was disabled, matching the preceding ownership qualification.
- Complete molecular tblite goldens (including Cl/Si d shells), analytic-force
  differences, covariance, retention and orbital/public API tests: 35 passed,
  13 explicit CUDA qualification skips.
- ASan/UBSan passed all 8 method-free contract cases. LeakSanitizer was disabled
  because this executor's ptrace prevents it from running; no leak qualification
  is claimed.
- The frozen setup probe observes 11 to 12 allocations, 900 to 940 allocated
  bytes and 884 to 908 reported resident bytes. Successful bind/factor/
  solve operations remain allocation-free.
- Source inventory and loaded library identity match
  `e0ab21632721a422291b8e43a5da273886a336cb09c92e930541d3254b69ce30`.
  The unchanged admitted SciPy OpenBLAS provider hash is
  `8fb864c29cac4b25f6e2c139491ea96f2724dde42d51394f84e9c4a622e34790`.
- Compiler, CUDA ownership, cross-method/SCF/vendor/provider boundaries passed,
  as did the exact CI ty command (0.0.82), Ruff 0.16.10 and clang-format 23.1.2.

Task-local receipts and reproduction logs live under `.artifacts/spectral-owner/`.
The shared cache counters are provenance, not a task-isolated cache-hit metric.

The SHA-guarded `spectral_prechange_5c02b1ba` test fixture contains exact baseline
source dependencies and a baseline-generated weighted-Gram header. It is test-only
and never registered in the production build. The same public-API harness runs
against frozen and candidate owners, comparing resource layout, all cache/output
bytes, provider traces, actual occupation/finite-symmetry function-call counts,
thread-scope placement, allocation counts and failure/recovery semantics.

The method-free owner contract compiles without native-method include paths and
uses arbitrary status encoding, ragged orders, multiple independent spectra,
hostile bindings and independently prescribed spectra/residuals. Supplemental
sanitizer builds link the real shared tensor implementation because the existing
generalized lowering's inactive canonical branch can survive instrumentation;
no method dependency is added to satisfy that link.
