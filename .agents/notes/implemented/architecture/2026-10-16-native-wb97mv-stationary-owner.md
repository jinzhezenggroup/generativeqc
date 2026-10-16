# Decision: native stationary WB97M-V owner and Python-free CLI

Status: implemented, with explicit linked-closure and qualification limits
Date: 2026-10-16

## Problem

Compiled derivative leaves and individually qualified sources did not remove
Python/CuPy from PR #2527's C++ CLI runtime. The missing consumer was the actual
accepted-state stationary assembly around native coupled-spin CPKS. A second W
formula, frozen DF metric, exact-for-DF substitution or coordinate-major source
tensor would have changed the scientific/ownership contract rather than migrated it.

## Decision

`CudaKsWb97mvHessian` borrows one live accepted batch and composes the existing
native physical induced Fock, H/S, exact/LR J/K, raw DF plus fitted-primary AD,
residual ECP and moving-grid owners. It requires CUDA success **and** CPKS
convergence. Reference W exposes the same primal underlying the existing W JVP
in `method.ks_cpks`; the existing solver arena is reused exclusively to compute it.
RKS occupation two and UKS physical spin panels retain the existing CPKS policy.

The compiler owns total D/dD, negative total W/dW overlap weights, frozen Fock
source addition and final HVP addition. Nuclear repulsion reuses the existing
scalar 1/r gradient-JVP, with generated charge/displacement/motion maps and
signed physical incidences. Native gathering uses direct triangular indexing
and one neighbor reduction per atom: O(N squared), not O(N cubed).

The frozen source pass supplies dH/dS/dF and zero-dD moving XC to the RHS.
After CPKS, moving weights feed a second source pass. DF keeps raw dA/dV
resident, differentiates the full inverse/source weights with solved dD, then
contracts the raw source. Nuclear-grid preparation retains bilateral VV10
directional work once per right direction; left-coordinate mixed contractions
reuse it and the cached AO spatial jets. Both passes and every grid direction
remain visible in source receipts; bounded storage is not a speed claim.

All independent preparation arenas, generated scratch and candidate outputs
share an aggregate response-byte bound. Actual receipts are charged and later
preparations are clamped to remaining bytes. Ordinary borrowed SCF storage is
outside this separate response budget. Actions allocate nothing. The complete
Hessian is privately staged and raw symmetry is checked, never repaired. A
failed column, nonconverged response or revoked owner cannot publish a matrix.

The CLI now calls this owner directly and serializes the complete canonical
MethodIR model at build time. It rejects obsolete interpreter/library/cache/
worker-timeout options. Missing AOT closure and CPU CLI execution fail explicitly,
without Python/JIT/CPU numerical fallback. The existing Python Hessian APIs are
separate consumers and are not claimed to have migrated by this change.

## Rejected alternatives

- Keeping a Python worker after precompiling leaves would not meet the runtime
  boundary, even if compilation happened only once or CuPy remained resident.
- Reconstructing W on CPU would introduce numerical downloads and another
  maintained convention; the compiler already owns the required primal.
- Publishing columns incrementally would leak a partial Hessian after a later
  response/source/symmetry failure.
- Replaying VV10 pairs for every left coordinate would hide avoidable work
  behind a memory-bounded implementation.
- Retaining the old CPU CLI bridge as a fallback would silently violate the
  requested Python-free CLI contract. Its Python API remains available explicitly.

## Evidence and limits

The reference-W implementation passed 55 Slurm native CPKS cases, including
empty/full spin populations and signed orbital energies. A values-only molecular
oracle separately reconverges libcint/Libxc/direct-space VV10 SCF at displaced
geometries and checks the native HVP at two displacement steps. The initial
Slurm 3146 run passed six RKS/UKS Direct/DF all-electron cases, including two
raw full Hessians, with native child interpreter/compiler discovery unavailable.
These are small-grid qualification fixtures, not quadrature convergence or
complete high-angular shell closure. Additional final combination/CLI receipts
must be retained before extending claims.

### Independent auxiliary admission regression

The original CLI UKS/DF/ECP spherical-d auxiliary comparison disagreed by
0.0405525 Eh/Bohr squared. Comparing primals first exposed a 0.00683416 Eh SCF
energy difference. A new independent all-electron d-auxiliary fixture then
reproduced the issue without ECP. Borrowing the actual retained raw source
showed only two auxiliaries where seven were requested: this was an admission
error, not an ECP derivative or an independent-oracle discrepancy.

`ks_auxiliary_template` previously required a nonnull opaque C ABI auxiliary
handle even when an internal native caller supplied its already decoded
`core::System`. Consequently it silently substituted the orbital basis. The
decoded native source is now authoritative; a nonnull ABI request without a
resolved native source still fails explicitly. Single and batch CPU native
regressions compare decoded-only preparation against the resolved ABI-marked
preparation and a numerically distinct default-fit negative control. The CUDA
probe also checks the retained auxiliary dimension and compares the physical
projected raw-source ERIs independently, rather than relying only on an HVP
computed in a self-consistent but unintended fit space.

The repaired actual CLI tests passed both Direct RKS full-Hessian/HVP and
UKS/DF/ECP with the independent spherical-d auxiliary (Slurm 3163, 65.55 s).
Both SCF energy and HVP must agree with the separate CPU consumer; its original
HVP tolerance is unchanged. The native child has no interpreter/compiler
discovery path. Host native auxiliary and CLI contract tests passed (29.29 s).
Local map/response/benchmark tests passed 81 cases, with 92 explicit optional
GPU skips; skipped cases are not GPU evidence.
The separate spherical-d fit passed all four RKS/UKS and all-electron/ECP
values-only reconverged energy gates (Slurm 3162, 172.29 s), including raw
source/metric projection checks against independent libcint ERIs.

The selected development package still does not contain practical complete
orbital-f / auxiliary-g ERI/DF closure. Broad molecular qualification and matched
external complete-endpoint GPU4PySCF timings remain outstanding. No source/probe
timing or arena size demonstrates completion or comparative endpoint speed.

## Revisit when

The package has practical complete high-angular closure and broad molecular
gates, or measurements show that duplicate independently charged grid owners,
paired integral passes or per-column response dominate a complete endpoint.
Any optimization must preserve the accepted-state and whole-result gates.
