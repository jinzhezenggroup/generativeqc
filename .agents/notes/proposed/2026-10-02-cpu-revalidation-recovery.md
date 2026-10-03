# CPU source recovery and fresh qualification

Status: proposed; new native qualification complete, endpoint comparison pending
Date: 2026-10-02

## Scope and scientific contract

This candidate integrates four CPU changes on master `9a5871dc`:

- Value-only Cartesian ERIs are written directly to scalar output storage;
  derivative requests retain the Jet producer and derivative storage
- Exact J/K shares a contiguous source-major traversal, retaining the scalar
  accumulation order for each output and the existing restricted J-only BLAS path
- The RKS feature owner specializes ingredient masks 1, 7, and 15 and fixed
  derivative axes; its runtime mask fallback and axis accumulation order remain
- The AO polynomial recurrence has twelve bounded through-f fixed shapes,
  retaining one runtime recurrence body and the radial-zero guard

These remain shared integral, Fock-provider, XC-feature and AO owners. No
method-specific HF/DFT scientific implementation, additional tensor copy,
screening policy, convergence change or new fast-math option is introduced.

## Recovery boundary

The old execution environment and its raw timing payloads/binaries were lost.
Scalar, J/K and XC source files were recovered with retained source identity
checks. The AO optimization is a newly reconstructed candidate against the
current baseline, not a recovered historical source identity. Historical timing
numbers are not new qualification receipts and are not asserted here.

## Fresh evidence

The current baseline and candidate both build with GCC 14.2.0, CMake Release
`-O3 -DNDEBUG`, verified ccache 4.14.1, CUDA disabled, and the same SciPy OpenBLAS
0.3.34 provider. Each passes all 66 current native CTest suites. Recovered scalar
storage and XC operation-order Python probes pass. The new AO reconstruction
has 158 focused checks across O2, strict O3, and production-style O3/PIC,
including a long-double independent derivative oracle; see its separate note.

Fresh 24-AO water HF and PBE endpoint energies agree with independent PySCF
using exact primitive inputs and, for DFT, identical quadrature. Full-size,
all-repeat, isolated endpoint qualification and broader endpoint regression
tests are still pending. No speed claim is made before that gate completes.

## Acceptance and rollback

Require unchanged scientific settings, all-repeat independent energy gates,
matched iteration/Fock/grid work, complete cold/process-warm/changed-geometry
timing, exact source/binary identities, and retained losing samples. A future
source-major compiler schedule must preserve the same bounded ownership and
reduction semantics. Any AO code-size/stack increase must be weighed against
complete endpoint performance, not an isolated kernel benchmark.
