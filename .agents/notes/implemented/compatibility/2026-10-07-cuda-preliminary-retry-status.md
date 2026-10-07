# Decision: retry CUDA preliminary seeds only on numerical failure

Status: implemented
Date: 2026-10-07

## Problem

The single-system KS adapter passed `reuse_warm=true` on its first execution and
mistook that permission for an available resident seed. This silently skipped
cold MINAO preparation. It also raised a returned CUDA physical-failure flag
before reaching the existing one-Hcore-retry path.

CUDA exceptions need narrower classification than a generic `runtime_error`
catch. Some device utilities still use untyped runtime exceptions, while the
KS status mapper formerly erased an explicitly numerical provider status into
that same untyped class.

## Decision

Expose `CudaKsPlan::has_warm_start()` as a host-only query of the retained
last-good seed flag. Explicit input and an available, permitted resident seed
continue to precede preliminary preparation; permission alone does not.

A prepared target attempt can consume exactly one fresh Hcore retry after
nonconvergence, a returned physical-failure flag, or a typed
`generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE)`. Preserve the
numerical type at the provider status mapper and seed-eigensolver convergence
and frame-validation checks. A negative seed-eigensolver `info` remains an
invalid-argument error; only positive `info` is numerical nonconvergence.
Propagate allocation, invalid-input, CUDA, logic,
overflow and untyped runtime exceptions without a hidden retry. The Hcore
attempt is outside the numerical catch and cannot trigger a third attempt.

The retry releases the discarded exported history and seed first, disables
resident reuse, and lets the ordinary `begin()` reset DIIS and revoke final-state
eligibility. It does not clear an independent last-good warm seed or override
the caller's warm-publication permission. Returned attempts preserve their
iteration/Fock counters; a numerical exception has no terminal result and marks
the diagnostic work census incomplete.

## Evidence and limits

`tests/python/test_preliminary_cuda_control.py` compiles the actual adapter
branch, CUDA run loop, status mapper, warm query and invalidation statements
against explicit host doubles. It covers precedence, retry bounds, exception
classification, diagnostics, history release and publication permission. These
are control-flow checks, not CUDA-device or MINAO-numerical qualification.

`tests/native/test_ks_cuda.cpp` also checks real-owner warm availability after
cold creation, convergence, final-state revocation, freezing and clearing, with
zero matrix-download/synchronization deltas for the query. Real-device execution
is still required to qualify that native coverage.
