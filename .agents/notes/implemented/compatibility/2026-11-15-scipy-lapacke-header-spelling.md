# SciPy LAPACKE header spelling and linked capability admission

Status: implemented; GPU endpoint performance qualification in progress
Date: 2026-11-15

## Problem

Some LP64 SciPy OpenBLAS distributions export `scipy_LAPACKE_dpotrf` and
`scipy_LAPACKE_dsyevd` while providing conventional unprefixed declarations in
`lapacke.h`. The existing configure probe called prefixed names directly and
therefore failed at compilation despite both symbols being present in the
selected library. The production adapter had the same header assumption.

The retained 2026-11-14 SM120 build selected OpenBLAS but recorded
`GENERATIVEQC_OPENBLAS_HAS_LAPACKE=0`; the earlier qualified build had it enabled.
An original-node4 same-device comparison reproduced the cold regressions with
unchanged solver settings and work counts. Diagnostic host traces found overlap
orthogonalization about 146 ms versus 13 ms. These intrusive observations
motivated capability investigation; they are not publication timings or proof
that this is the only endpoint regression.

## Decision

Use one private inclusion adapter in configure probes and the build-linked
runtime. For a SciPy-prefixed provider, rename only the two used LAPACKE tokens
during inclusion of the provider header. Preserve the provider's declarations
and argument types, rather than inventing a second ABI declaration. Already
prefixed headers and ordinary providers retain their existing interfaces.

Both actual link symbols remain mandatory. Missing symbols still leave LAPACKE
unavailable and retain the existing scalar fallback. Do not force the cached
capability to true, skip numerical validation, change solver controls, or infer
LP64 support merely from a package label.

## Evidence

The executable static-provider regression covers prefixed and conventional
headers independently from exported symbols. Before the fix, the conventional
header/prefixed-symbol case failed while three controls passed. After the fix,
all four passed, including missing-symbol rejection. The real installed provider
and related host capability suites passed 33 tests; 28 pkg-config cases were
unavailable because that host lacks pkg-config. Corrected configure output
actually discovers LAPACKE, without a command-line capability override.

GPU E+F checks and the complete independent-process cold/warm matrix must still
qualify any replacement performance figure. Retain the prior failed timing
cohort and every numerical failure; this compatibility fix is not itself a
64/64 performance claim or overall scientific acceptance.
