# Decision: prepare ordinary DFT eigensolves through the shared lowering contract

Status: implemented
Date: 2026-10-05

## Problem

DFT/KS already borrowed `OrdinaryStreamEigensolver`, but that resource owner
selected a native/cuSOLVER family on each launch without the canonical
request/candidate boundary required by #1886/#1890. Moving vendor calls between
files would not fix that gap.

## Decision

Reuse the existing `LibraryRequest` execution vocabulary. Its canonical owner
moves from `integral.runtime_backend` to `common.library`; the old import
re-exports the identical class. `common.solver_lowering` projects the ordinary
symmetric operation into `LoweringRequest`, `LoweringPrecision` and two
`LoweringCandidate` records. This is an execution adapter, not a second IR or
a generalized Fock/overlap eigensolve.

The full-symmetric FP64 input is overwritten by column-major eigenvectors;
eigenvalues are ascending. Active masks and asynchronous device-info publication
are explicit effects. Both CPU and CUDA adapters retain the same science and
semantic identities. This change implements only the ordinary CUDA binding;
it does not migrate CPU solver execution.

Native preparation resolves the existing disjoint size domains (native AOT
through 16, ordinary Xsyevd above 16), actual cuSOLVER runtime version, device,
driver/runtime versions and queried numeric workspace. Shared selection retains
the qualified incumbent with unknown endpoint costs. Replays use the immutable
selected family; there is no heuristic search or library preparation per solve.
The diagnostics' canonical identities describe an AOT template, with actual
dimension and device/version facts retained separately. There is no cross-owner
executable cache or claim that the template hash alone authorizes reuse.

## Residual and resource boundaries

`LibraryRequest` retains its default runtime-residual contract. The existing
ordinary algorithm explicitly requests the new `qualification` policy, which
requires independent numerical qualification but promises no per-launch residual
measurement. A request for a runtime residual is rejected by this adapter.
This preserves the existing numerical workload and avoids claiming that a
cuSOLVER status code proves an eigensystem residual.

Keep the existing host and device numeric allowance separately:
`1 MiB + 16*n*n*sizeof(double)`. A 16 KiB host reservation covers the prepared
binding, including native object storage; native and Python KS resource owners
charge it even for tiny matrices. The selected candidate reports queried numeric
workspace plus that metadata reservation. Opaque cuSOLVER allocations remain
explicitly unknown and are not covered by the numeric allowance.

## Preserved behavior and remaining work

Inactive provider matrices are sanitized, serialized batches reuse one workspace,
info/error behavior stays unchanged, and small solves remain capture-safe.
Provider-backed ordinary solves reject capture instead of silently switching to
maximum-pivot Jacobi. All preparation now rejects capture before querying library
resources. Stream/device lifetime stays with the existing owner.

RHF and DF callers that directly select `launch_solver` families remain #1890
migration debt. The vendor inventory deliberately retains its migration
classification for this partially migrated module. No new algorithm, precision
variant, default promotion, SCF speedup or force speedup is claimed.

## Evidence

Retain build, compiler-cache, source/binary provenance and qualification logs in
ignored `.artifacts/1890-eigen/` on the allowed compute node and local checkout.
The first host qualification passed 216 contract, native-selector, compiler
structure and ownership tests. Real-device qualification is recorded in the PR
and appended here after the current-branch native build completes.

The native oracle constructs analytic Householder eigensystems and checks their
spectrum/eigenvectors and extended-precision residuals at sizes 7, 16, 17, 24,
192 and 768, including two spin slots and repeated execution. It also checks
changed-input graph replay, inactive NaNs and explicit provider capture rejection.
Complete DFT qualification uses the existing native KS suite and 24-AO PBE/r2SCAN
RKS/UKS PySCF gates, including warm replay, changed geometry and final-state export.

Revisit provider competition only with independent numerical qualification and
complete endpoint evidence; do not infer a faster choice from workspace or a
kernel-only timing.
