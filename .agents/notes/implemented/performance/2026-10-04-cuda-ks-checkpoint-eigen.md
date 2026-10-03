# Decision: use the idle CUDA KS eigensolver for checkpoint admission

Status: implementation candidate; device qualification and endpoint comparison pending
Date: 2026-10-04

## Problem and measured scope

The private GPU LDA preliminary-density experiment charges source export and
target import to complete WB97M-V cold startup. At 48 atoms, its original
source export/import costs about 12.7 seconds. A separate finite n5 Slurm 1420
component audit distinguishes the ABI calls: 24-atom export query/copy take
0.000138/0.000103 seconds, while three imports take 1.231363/1.225954/1.225731
seconds. At 48 atoms, query/copy take 0.000495/0.000400 seconds and imports take
15.684234/15.631609/15.632248 seconds. Imported densities remain bitwise equal
to the converged source. The audit does not execute a WB97M-V target endpoint.

Slurm 1421 wraps the unchanged restore ABI in the existing host observer. Each
48-atom import makes two actual `reference_eigensolve` calls with reason
`seed_validation`. Their combined time is 15.346126–15.363690 seconds out of
15.625583–15.642699 seconds of native restore. This identifies scalar reference
diagonalization as the dominant import cost, rather than density transfer.
It does not measure an optimized endpoint or a general checkpoint workload.

The audit uses qualified integration source 678f7eb88, native identity
`d198acd713d7c4218bb4cacd9acbd7eb646d5fc2b89c77d04bdb47901c74ad36` and library
`fe826ad7eb9293686d5bb3b5e97456f3a9c884a13256603e7573ea3a1703f25b`.
The implementation is separately based on master d9431c913. Evidence, scripts,
ccache receipts and the diagnostic wrapper remain in the integration checkout's
ignored `.artifacts/seed-transfer-audit-20261004/`. The first audit attempt,
Slurm 1419, fails during library loading because importing CuPy first binds an
older private cuSOLVER. The corrected driver matches the comparator's
native-first load order; that failure is retained, not treated as qualification.

## Decision and invariants

Allow the existing shared warm-density guard to receive a borrowed eigen
operation. An idle prepared CUDA KS owner supplies its ordinary eigensolver;
CPU callers and owners without a prepared CUDA plan keep the existing route.
Source overlap is still rebuilt from the checkpoint's coordinates, not silently
substituted with the target metric. Electron/spin trace, shape, finiteness,
Hermiticity, singular-overlap and ensemble-occupation gates remain unchanged.
The common guard retains its existing reference fallback for legacy inputs
which satisfy its 1e-7 asymmetry contract but not the GPU frame's 1e-12 rule.
Runtime/provider failures propagate instead of selecting a numerical retry.

The callback borrows the already charged Fock/residual DIIS-history matrices
and the iteration solver's temporary eigenvalues/status while no iteration is
active or pending. Every future `begin()` discards that history. It does not
borrow `tmp1/tmp2`, which may hold a published stationary D/W lease. Final
coefficients/eigenvalues/status, density, warm state and generation tokens remain
untouched, including on rejection. Graph capture is explicitly rejected.
No device allocation or extra solver owner is introduced. Synchronous downloads
drain before host input/frame/status storage can expire.

Column-major device eigenvectors are converted to the common row-major frame.
The shared independent eigenframe residual/metric validator checks every frame
before the existing ensemble guard consumes it. Native code binds the provider
and storage; it does not reimplement an occupation formula or repair a density.
The optional host trace reports actual `cuda_ks_seed_eigen` executions.

## Required qualification

The new independent-metric tests cover native-small/provider solver sizes,
restricted/unrestricted analytic ensemble densities, charge-preserving invalid
occupations, atomic multi-item rejection, continued last-good warm execution,
exact density preservation and legacy near-symmetric admission. Device runs,
live stationary-weight lifetime checks, existing checkpoint/resource regressions
and complete source-plus-WB97M-V E/F comparisons are still required. This slice
does not expose a public CUDA preliminary-SCF API, lower resource bounds or
claim that the remaining source/target capacity contract is solved.
