# Decision: isolate the real OpenBLAS pkg-config test import

Status: implemented
Date: 2026-10-12

## Problem

The real pkg-config test imported the build-input package `scipy_openblas32`
inside a persistent pytest worker. Its import calls `ctypes.CDLL` with
`RTLD_GLOBAL`, exposing internal OpenBLAS symbols to libraries loaded later.
The ordinary local loading mode of GenerativeQC does not undo this side effect.

On PR #2342 CI run 38204607750, only worker gw0 ran the real provider test.
That worker later faulted in PySCF's `NPdgemm` during the independent LDA
reference calculation. Its native record identified `libscipy_openblas.so`
at file offset `0xa90650`. For the reviewed scipy-openblas32 0.3.34.0.0 Linux
x86-64 wheel, this is `dgemm_oncopy_SKYLAKEX+0x1f0`, a destination store.
The other workers did not run that import, and a compared unaffected worker
did not contain the unhashed provider.

A bounded local reproducer establishes actual interposition: importing this
official provider before PySCF 2.14 redirects PySCF's old OpenBLAS bindings
for `gotoblas`, `dgemm_nn`, and `dgemm_tn` to the new provider, while its
public `dgemm_` remains bound to the old provider. A 12-by-12 matrix product
then terminates with SIGSEGV. The clean process returns the expected matrix.
The new provider reports LP64 configuration; this evidence demonstrates
mixed internal OpenBLAS generations, not an LP64/ILP64 integer mismatch.

## Decision and invariants

Import the build-input package and inspect its real symbol capabilities in a
bounded subprocess. Return its own pkg-config text and capability results
to the test worker. Preserve the production CMake configuration, compile,
link, and native consumer assertions. Check that metadata acquisition leaves
the worker's existing global BLAS bindings unchanged.

Do not replace the real provider probe with synthetic-only coverage, change
scientific tolerances, skip reference tests, or treat retrying the crash as
a repair. A local `CDLL` elsewhere is not sufficient namespace isolation.

## Evidence

- All 28 tests in `test_cmake_cpu_linalg_pkgconfig.py` passed, including the
  real provider compile/link/runtime probe and namespace guard
- The unchanged old-source real probe fails a before/after namespace
  assertion; the repaired probe preserves all inspected bindings
- Running PySCF's matrix product in the repaired probe's process succeeds
- Compilation used verified ccache 4.14.1; no full project build was needed
- The official dependency wheel and library matched the repository's pinned
  `tools/xtb/scipy_openblas32_manifest.json` checksums; CI artifact binaries
  were never executed

## Consequences and revisit conditions

The real provider probe starts a short Python subprocess. Keep that boundary
unless the upstream import no longer globally exposes conflicting native
symbols and compatibility with later reference-library loading is qualified.

## References

- https://github.com/jinzhezenggroup/generativeqc/pull/2342
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38204607750
