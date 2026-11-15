# Decision: share SciPy LAPACKE declaration adaptation with capability probes

Status: implemented
Date: 2026-11-15

## Problem

The SciPy OpenBLAS LP64 wheel used by the PR 3084 acquisition,
`scipy-openblas32==0.3.30.0.6`, exports `scipy_LAPACKE_dpotrf` and
`scipy_LAPACKE_dsyevd` but declares the ordinary names in `lapacke.h`.
The existing CMake probe calls the exported names without declarations and
incorrectly disables LAPACKE. Simply forcing the capability true also leaves
the production translation unit uncompilable. Newer wheel headers can already
declare prefixed names, so the fix must support both layouts.

## Decision

Use one private declaration adapter in the production owner and CMake link
probe. For the already-selected SciPy provider, rename only the two consumed
ordinary declarations to the exported prefixed names before including the
vendor header, then remove those aliases. Already-prefixed declarations are
unchanged. Ordinary OpenBLAS continues to use its ordinary declarations.

Keep capability detection as a real compile/link check. Missing declarations,
headers or exported routines still disable LAPACKE and retain the existing
scalar fallback. Do not infer ABI support from a wheel version, manually enable
cached capabilities, add a second BLAS dependency, or hand-declare signatures
that could disagree with the vendor's integer ABI.

## Validation boundaries

Synthetic shared-library tests exercise both SciPy header spellings with
complete and incomplete prefixed exports, building and executing consumers of
the same private adapter. Existing real-provider tests compile the production
CPU owner and execute scalar/OpenBLAS numerical contracts, including Cholesky,
non-positive-definite failure and symmetric eigensystems. Cached-capability and
provider-discovery tests retain their existing behavior.

Local validation passes all 33 pkg-config cases and 63 adjacent capability-cache,
Python-provider-discovery, CPU-boundary and packaging cases with the installed
`scipy-openblas32==0.3.34.0.0`. The three real-provider cases also pass against
an isolated copy of the actual `0.3.30.0.6` acquisition package, including both
production-owner builds and numerical-contract execution. Before the fix, the
new complete-export/unprefixed-header case reproduces the false negative.
Compilation reuses the existing ccache store; commands and before/after cache
statistics are retained with the local validation logs outside Git.

The native Libxc archival checker also tracks live source-input identities.
Regenerate its derived snapshot and reference page with the owning checker
after this source change. Only the two checked input identities change: the
retained campaign, measured identity, binary identity, stale status, qualification
summary and all negative evidence remain unchanged. This is archival consistency,
not new Libxc qualification, and no acknowledgement or scientific gate is relaxed.
The live checker and all 75 native-catalog receipt/context/regression tests pass
after this derived refresh.

This independent fix does not change AOT profile selection, PR 3084 data or
figures, CUDA code, numerical acceptance gates, or provider ownership policy.
The earlier single-translation-unit endpoint experiment is diagnostic evidence,
not performance qualification of this new source identity. No GPU speedup or
complete application-cohort rerun is claimed here.
