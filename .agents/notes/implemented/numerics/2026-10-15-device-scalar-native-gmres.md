# Decision: keep native Hessian-response solver scalars on CUDA

Status: implemented (generic solver only; molecular native endpoint incomplete)
Date: 2026-10-15

## Problem

PR #2527 must eventually execute the complete WB97M-V Hessian/HVP without a
Python runtime. Its explicit Python/CuPy worker fails that contract. The existing
native resident GMRES controller also cannot establish an all-CUDA numerical
contract: although dimension-sized vectors remain resident, Hessenberg/Givens
arithmetic executes on the host. Do not relabel either implementation as the
requested replacement.

Existing mixed host-scalar consumers retain their unchanged controller and
compatibility contract; this opt-in solver does not silently migrate them.

## Decision

Add a separate opt-in native generic FP64 GMRES primitive to the existing AOT
development library. Compiler-owned CUDA schedules implement stable scaled
norms, one/two-pass modified Gram-Schmidt, right diagonal preconditioning,
Givens rotations, triangular solves and candidate reconstruction. The borrowed
native operator runs on the same stream. Every candidate is accepted only by
its explicitly evaluated true, unpreconditioned residual, including on Arnoldi
breakdown. No solver numerical operation executes on the host.

The native controller owns only checked size arithmetic, workspace binding,
launch sequencing, discrete-status checkpoints and transactional publication.
Only the terminal scalar receipt is downloaded after iterative status checks.
Vectors and scalar workspaces are caller-owned; no internal CUDA allocations,
BLAS handles, JIT, interpreter or CPU fallback are used. The public solution is
untouched on numerical failure, breakdown, iteration exhaustion and operator
failure. Admission rejects invalid options, overflow, insufficient budget,
misalignment, undersized workspace and input/output/workspace aliasing before
GPU work. Read-only RHS and diagonal may alias each other.

This solver does not authenticate an SCF token or construct a physical response
operator. Native coupled UKS orbital maps, metric transport, D/W reconstruction,
XC/integral/DF responses and complete CLI wiring remain required. It registers
no molecular Hessian capability.

## Rejected alternatives

- Calling the existing host-scalar resident controller would violate the stricter
  numerical CUDA boundary, even though it is already Python-free.
- Downloading projection/norm coefficients or using a CPU small least-squares
  solve is not an acceptable hidden fallback.
- Trusting a projected residual or happy breakdown can publish a wrong solution
  for an inconsistent/nonfinite operator. Independent true-residual acceptance
  is mandatory.
- Squaring raw vectors for norms overflows/underflows well inside representable
  FP64 input ranges. A deterministic single-block scaled sum of squares avoids
  that failure without atomics or a second reduction workspace.

## Evidence

Source base `c5554f97a` plus this change; Release CUDA-sm120, `--fmad=false`,
verified ccache 4.5.1 with explicit CXX/CUDA launchers. Retained Ninja commands
show the actual launcher. Shared cache before/after statistics are retained;
their aggregate changes are not attributed exclusively to this build.

Allowed node2, `main` partition, `gpu:pro6000:1`, finite Slurm job **3063**,
inherited CUDA_VISIBLE_DEVICES=0: **30 tests pass in 10.90 s**. Independent NumPy
linear-solve and true-residual oracles cover nonsymmetric coupled matrices,
restarts, signed preconditioning, empty/zero systems and RHS scales 1e-250 and
1e200. Failure tests include nonfinite inputs/actions, a preconditioner overflow,
gap separation, singular operators, iteration exhaustion, invalid workspace and
callback failure after candidate construction. A deliberately inconsistent
operator has a zero projected residual but a nonzero true residual and must
leave the output unpublished. A standalone solve succeeds with an empty PATH
and unavailable Python paths. Build-time generation blocks public-runtime,
CuPy, PySCF and Torch imports.

Final combined native leaf/solver regression, finite node2 Slurm job **3066**:
**63 tests pass in 27.57 s**. Focused CPU/tooling regression: **58 pass, 29
explicit opt-in tests skip** in 19.33 s. Compiler structure checks 551 modules
with zero dependency errors; pinned ty 0.0.82 and the complete pinned prek 0.3.1
pre-commit suite pass, including the newly staged native and compiler files.

Finite node2 Slurm job **3065** runs a restarted/preconditioned solve under
Compute Sanitizer memcheck, initcheck, racecheck and synccheck: zero errors or
hazards. An earlier telemetry-only job 3064 stops before sanitizers because
`nvidia-smi` reports an unrelated NVML driver/library mismatch; no numerical
results are attributed to that attempt. Pending node4 job 995 is cancelled
before execution because another allocation holds its GPU. No scheduling or
device-visibility bypass is used.

Standalone probe SHA256:
`bc404905f09e416b82a76d9385a7a3a52fa87e589b6e52ce82d581e887891a94`.
Dependencies are cudart, libstdc++, libgcc_s and libc, not libpython or the public
GenerativeQC runtime. Receipts are in the ignored local
`.artifacts/wb97mv-hessian-aot-20261015/gmres-*` files. Test/sanitizer durations
are not complete Hessian endpoint timings or speedup claims.

## Revisit when

An independently qualified multi-block reduction or native library primitive
materially improves complete response endpoint timing while preserving device
scalar ownership, deterministic acceptance and explicit workspace accounting.
The solver is only a foundation for the outstanding
[complete native Hessian owner](../../proposed/2026-10-15-python-free-wb97mv-hessian.md).
