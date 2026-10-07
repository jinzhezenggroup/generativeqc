# Decision: shared CPU LP64 provider and prepared spectral ownership

Status: implemented
Date: 2026-10-07

## Problem and decision

Moving the embedded loader alone would leave GFN owning its ABI, primitive
dispatch and numerical execution boundary. Routing GFN through canonical
cpu_linalg would instead change runtime admission, work ownership, layout,
thread policy and fallback behavior.

The shared CPU owner now contains the LP64 ABI, coherent table and loader,
preflight, provenance and process/thread lifetime. Typed Cholesky, condition
and TRSM operations and opaque eigen/GEMM bindings remove method-side raw
pointer extraction. The table implementation is private to the provider TU;
test injection goes through the shared complete-cohort factory.

The prepared ordinary-eigen service is genuinely consumed by canonical
cpu_symmetric_eigen/Gaussian cpu_target_eigen and GFN solve_one_spin. Its scalar
body preserves arithmetic/order text with the required tensor:: result-type
qualification; the original canonical translation-unit inclusion is retained.
Canonical linked LAPACKE retains its existing allocating row-major callback.
GFN's opaque binding submits caller-owned column-major DSYEVD work.

Against the reconstructed prepared gate, the GFN .cpp retires 840 lines and
adds 27; its header retires 110 lines and adds 11. These include actual ABI,
table, symbol-loader, self-test, factory, thread-scope and primitive-argument
ownership, not merely a directory rename. The method factories remain thin
status adapters.

## Retained invariants

- Gaussian fixed scalar/task-parallel/one-thread/1e-14-cap admission is unchanged
- Canonical auto/scalar/build-linked admission and fallback are unchanged
- Runtime LP64 still requires the complete five-function cohort and has no
  scalar/build-linked fallback
- Ordinary configured-runtime failure still tries known system SONAMEs;
  configured private cohorts fail closed
- LP64 checks, standard/prefixed cohesion, self-tests, Linux namespace isolation,
  successful handle retention and inherited partial MKL retention are preserved
- One outer GFN sequential scope remains at each original method boundary;
  no nested scope, optimal workspace query, allocation or buffer replacement
  enters the borrowed leaf
- N-derived DSYEVD counts remain 1+6N+2N² doubles and 3+5N int32 values for n<=N
- CPU completion is host-return, without CUDA-stream semantics
- Generalized transforms, residual/conditioning thresholds, occupations,
  thermodynamics, density arithmetic, caches, and publication remain method-owned
- No GPU source or generated scientific algorithm changes; opaque weighted-Gram
  binding is CPU-only
- Inactive MKL, desktop-private and Pyodide arms remain unqualified and receive
  no new admission/build wiring

The 865-line provider implementation intentionally retains the complete
platform-loader/state closure in one TU. Verification, once-only publication,
namespace/TSS handle retention and failure cleanup share that state and ordering.
Keeping the inherited bodies together makes their transfer auditable and avoids
splitting inactive arms into a purported new support surface. This is a narrow
exception to the shared-module size review target, not a new numerical monolith.

## Recovery and source identity

An earlier local prepared gate was lost during environment replacement.
Only its production/fused-fixture cutover was reconstructed from retained
authored content, against verified published d84e4006. The full old test/docs
tree and its aggregate proofs were not recovered or relabeled as new evidence.

Reconstructed production blobs:
- Canonical cpu_linalg.cpp: 071e8f5e702af5301cacb8cb9c4f28e4fc74fb05
- Prepared header: 00d421f22e9ef97a3e220f3c2d82c61de6465b66
- GFN eigensolver.cpp: 3ea53219fe1d31c313736a0e3cb25a8aa8d4537c

The new extraction starts from published d27132b9. Those existing source files
were unchanged between d84e4006 and d27132b9; later reported master changes
concerned DFT projection reservation, with no migration overlap. MINAO/default
and CUDA build-pool ownership were left alone. The separate held Johnson
manifest is neither included nor modified.

The integrated library qualification below uses production commit
096b24f92a577a1deb2ae77cb3bf778824ef94e6, tree
e0a480e70718a05d3551ccce2b67f2eef498ca22. Subsequent durable-test and documentation
changes leave that production source closure unchanged. Earlier checkpoint
462e305 exposed Access in a public header; review caught this and 096b24f moved
its implementation into the provider TU.

## Current evidence

Focused executable coverage includes:
- 29 loader/source cases covering complete standard/prefixed cohorts, mixed or
  missing symbols, LP64 configuration, every self-test failure, 12-thread
  once-only initialization, configured-path precedence/fallback and real
  dlmopen private-namespace behavior
- Actual typed GFN n=2/N=5 pointer, primitive-order, 81/28-work, raw-info and
  no-extra-thread-scope checks, plus opaque weighted-Gram and cleanup lifetime
- Independent prescribed spectra and original generalized residual/metric
  checks through the actual GFN real-LP64 and Gaussian scalar consumers
- Existing fused numerical/publication tests using the shared test factory
- Durable borrowed bounds, all six active-buffer alias pairs, capacity,
  alignment/address-wrap, preparation atomicity and host-return effects
- Compile-negative raw-access checks and no implicit conversion from opaque
  bindings back to function pointers
- Durable owned row-major LAPACKE pointer/thread/failure/fallback/cap and
  allocation-failure checks under local and global thread-control profiles

A separate reviewer passed 67 focused cases on the repaired production source.
Its independent canonical comparison covered 1,000 deterministic matrices
(sizes 1–24, ordinary/large/tiny scales and scalar caps), with matching output
bits and allocation counts, plus eight exception cases. Another 48 old/new
build-linked LAPACKE spy combinations matched pointer ownership, call count,
thread restoration, exception behavior and allocation failures. Linked spies
are not real linked-library numerical qualification.

Five independent architecture checks passed. The reviewer verified exact loader,
cohort, self-test, namespace and TSS-bootstrap bodies; the factory changes only
its neutral status enum. The linked LAPACKE body is unchanged, and the scalar
arithmetic/order is preserved modulo its result-type qualification.

The CPU Release build completed and all 77/77 native CTests passed. Additional
endpoint tests passed 23 cases, with 10 explicitly CUDA-only skips. Four Gaussian
fixtures pinned the core seed and checked energy/force versus energy-only
execution, totaling eight executions. No production fix was needed.

Library SHA-256:
669eb85f5fa199f08e68a7da619199489e80e748807c45523275bfdee49efeee

Real runtime provider SHA-256:
8fb864c29cac4b25f6e2c139491ea96f2724dde42d51394f84e9c4a622e34790

Toolchain: GCC 14.2.0, CMake 4.4.4, CPU-only Release, -O3 -DNDEBUG -std=c++20,
PIC and normal warning flags. sccache 0.16.0 was verified but its resumed
server/compile path failed with Operation not permitted before processing
requests. The verified ccache 4.14.1 fallback completed the existing build tree.
No cache was cleared or relaxed. Actual Ninja commands verify launcher use and
the moved provider's original configured-runtime definition. Shared-cache
counters recorded 289 cacheable calls, one hit and 288 misses during that build
window; these are shared snapshots, not an isolated profitability measurement.

Ignored qualification receipts preserve 1,921 production-input hashes, 71
binary/library hashes, the actual compile command, complete build/CTest/endpoint
logs, provider observation and cache provenance.

## Remaining boundaries

This retires CPU LP64/provider ownership from the embedded eigensolver; it does
not retire GFN generalized-overlap policy, SCC orchestration or the complete
embedded runtime. No CUDA/device, performance, wheel-distribution or inactive
platform qualification is claimed. Durable guards use checked-in fixtures and
do not require an unpublished/transient Git ancestor. Publication authority and
the Johnson full-file approval boundary remain separate.

