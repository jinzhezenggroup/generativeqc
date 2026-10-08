# CPU linear algebra and prepared symmetric eigen execution

CPU methods share primitive implementations and ordinary symmetric-eigen
execution while retaining distinct provider admission and scientific policy.

## Owners

- `src/tensor/cpu_linalg.*` owns the canonical scalar/build-linked OpenBLAS
  plan, dispatch and thread policy
- `src/tensor/cpu/lp64_abi.hpp` declares the borrowed LP64 primitive ABI
- `src/tensor/cpu/lp64_provider.*` owns runtime loading, complete-cohort
  verification, provider provenance/lifetime, typed Cholesky/condition/TRSM
  calls, opaque eigen/GEMM bindings and sequential thread scopes
- `src/solver/cpu/symmetric_eigen.hpp` owns immutable ordinary-eigen
  preparation, work binding, scalar Jacobi and borrowed DSYEVD submission
- `src/solver/cpu/prepared_spectral.*` owns immutable ragged spectral dimensions,
  exact LP64 resources, borrowed overlap caches and transactional factorization,
  and single-spectrum generalized execution
- `src/methods/gfn2_electronic_update.cpp` adapts GFN electronic layouts, occupation
  policy, density/thermodynamic construction and result publication

The canonical `cpu_symmetric_eigen` facade and shared prepared spectral owner both
execute the prepared service. Gaussian `cpu_target_eigen` reaches it through
the canonical facade. GFN compatibility factories only translate shared
initialization status to the existing method status and error boundary.

The raw provider table is private to its implementation. Method code uses
typed operations and opaque callable bindings. Weighted-Gram accepts the
opaque CPU GEMM binding without changing its scaling, accumulation or
publication semantics. This does not broaden compiler CPU tensor offers,
which retain their existing strict-FP64, compact, unbatched domains.

## Three admission profiles

The primary Gaussian CPU HF target uses an explicit scalar, task-parallel,
one-thread plan and a `1e-14` absolute off-diagonal cap. It neither changes
provider nor retries through another eigensolver. Setup, reference exports
and other method routing retain their own existing eligibility.

Canonical `CpuLinalgPlan` resolves scalar, automatic or build-linked
OpenBLAS. Task-parallel OpenBLAS requires local thread control.
Provider-parallel ownership can use local control or the existing
mutex-protected global control. Automatic selection retains scalar fallback;
an unavailable explicitly requested OpenBLAS provider is rejected. The linked
eigen entry remains allocating row-major `LAPACKE_dsyevd`.

The runtime-LP64 profile requires one complete DPOTRF_WORK, DPOCON_WORK,
DSYEVD_WORK, DTRSM and DGEMM cohort, matching standard or SciPy-prefixed
symbols, a valid LP64 configuration and the existing preflight. It provides
no scalar or canonical build-linked fallback. Native configured/system
OpenBLAS requires local thread control.

An ordinary `GENERATIVEQC_XTB_CPU_LINALG_LIBRARY` path has priority; failure
retains the existing known-system-SONAME search. A configured private wheel
cohort instead fails closed. The current Linux private-wheel path opens its
sibling shim in a new glibc link-map namespace. Native configured/system
discovery does not claim that stronger namespace isolation.

The inherited MKL, macOS/Windows-private and Pyodide source arms remain
inactive/unqualified in the integrated build. Their presence does not add
platform admission or install/build wiring.

## Storage and completion

Owned row-major execution moves its input vector and may allocate scalar
vectors or use LAPACKE's internal allocation. It is not an allocation-free
provider contract. The scalar routine remains header-included in its
original canonical translation unit so its compilation context is retained.

Borrowed column-major execution uses caller-owned matrix, values, double work
and int32 work arrays. For maximum order N, preparation preserves exactly:

- Double count: `1 + 6*N + 2*N*N`
- Integer count: `3 + 5*N`

A smaller system with n <= N receives those N-derived counts. Padded capacity
does not become submission lwork, and no optimal-workspace query is added.
Preparation and binding reject invalid extents without publishing a partial
descriptor. The existing method arena layouts and alignment remain owned by
their callers.

Completion is host-return: provider writes and raw info are available when
the call returns. Borrowed submission introduces no allocation, buffer
replacement, thread-setting scope, asynchronous completion or provider
selection. Opaque provider-owned storage is not represented as caller
numerical workspace.

GFN retains one outer sequential scope around each existing transaction.
The scope restores the calling thread's prior setting, including exceptional
exits. Successful runtime handles remain loaded for process lifetime;
backend copies do not unload them. Thread-resource cleanup remains an
explicit action at the runtime owner's existing teardown point. Inherited
MKL partial-initialization retention remains unchanged.

## Method-owned policy

Gaussian retains its symmetric-orthogonalizer transform, back transform,
original-F/S residual checks and metric validation. The shared spectral owner
executes Cholesky reduction/recovery, explicit symmetrization and overlap
conditioning using the caller's threshold. Its cache accepts any nonzero caller
generation. Backend failure preserves the complete committed cache; numerical
failure commits that member's generation/status while retaining its old factor
bytes. Scientific cache invalidation remains the caller's responsibility.

GFN retains occupation/thermodynamic policy, density construction and transactional
publication, including its two CPU occupation solves for restricted systems.
Its compatibility header and CUDA topology setup API remain unchanged. Plan
construction adds one shared immutable metadata allocation, included in resident
accounting; numerical cache and workspace sizes, offsets, and alignment are
unchanged. Successful factor/solve transitions use only caller-owned storage.

The shared borrowed service returns raw LAPACK info. GFN preserves negative
info as a call-level internal error and positive info as per-system
eigensolver failure. Canonical LAPACKE keeps its existing invalid-argument
and nonconvergence exceptions. Neither service moves method acceptance
thresholds into the shared provider.

## Qualification

`tests/python/test_cpu_lp64_provider.py` compiles actual provider and method
sources. Its gates cover coherent/missing/mixed/ILP64 cohorts, once-only
initialization, configured-versus-private fallback, Linux namespace isolation,
typed primitive arguments and pointers, exact work counts, status mapping,
thread/lifetime behavior, opaque access and owned-LAPACKE admission.

Set `GENERATIVEQC_TEST_LP64_LIBRARY` to an existing verified provider's absolute
path for the explicit real-LP64/Gaussian prescribed-spectrum oracle. Add the
provider dependency directory to `LD_LIBRARY_PATH` when needed. The linked
LAPACKE tests use spies; they are distinct from the real runtime numerical
oracle. No performance or inactive-platform qualification follows from them.

Reviewed scientific/admission source contracts are stored in
`tests/data/cpu_lp64_source_contract.json`. They work without Git history.
An intentional change to a protected region requires reviewing its numerical
or policy effect and updating that explicit fixture. The existing GFN
molecular, runtime-retention, canonical eigenframe and native CTest gates
remain separate integration requirements.

`test_prepared_spectral_owner.py` compiles the shared owner without method include
paths and checks resource admission, independent spectra, failure transactions
and allocation-free execution. `test_gfn2_spectral_preservation.py` compares the
real method adapter against a SHA-guarded pre-extraction consumer, including
exact cache/workspace layouts, occupation-call counts and staged publication.
