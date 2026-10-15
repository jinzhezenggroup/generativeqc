# Decision: native AOT execution for the complete WB97M-V Hessian

Status: proposed (native leaves and generic CUDA solver implemented; endpoint incomplete)
Date: 2026-10-15

## Problem

The user clarified that PR #2527 must have an entirely Python-free runtime.
The previously implemented explicit Python/CuPy CLI worker does not meet that
requirement, even though differentiation belongs to the compiler and numerical
arrays stay on CUDA. Precompiling kernels while preserving Python orchestration
does not close this gap. The PR is draft while the native replacement is absent.

## Direction and delivered boundary

Reuse existing compiler mathematical roots and generate native CUDA at build
time. Native subsystem owners must eventually execute the complete prepared KS
state, integral traversal, tensor contractions, coupled CPKS and stationary
assembly. Do not create a second integral/XC derivative algebra or substitute
finite differences, displaced SCF, PySCF or CPU numerical providers.

The first delivered slice is an explicit static CUDA leaf library with a typed
native descriptor registry and standalone C++ numerical probe. Identifier aliases
isolate translation units without rewriting emitted arithmetic. The registry
contains immutable logical identity, exact scalar input order, component indices
and center/record/output packing. It performs no runtime compilation, allocation,
device selection, interpreter loading or hidden package discovery. All callbacks
borrow the caller's buffers and CUDA stream. The caller owns status checking and
transactional publication. Missing package entries fail instead of invoking JIT.

The checked-in development inventory deliberately does not pretend to cover the
complete shell-component closure. It includes all scalar leaf families, the
existing through-f AO policy and representative S/T/V, range ERI and raw DF
tiles. Raw integral gradients and their directional derivatives retain all
mathematical centers before physical-atom scatter. Unit-charge attraction keeps
the actual nuclear charge in native primitive scales, avoiding a charge-specific
kernel family. Range kernels retain the actual fixed WB97M-V omega=0.3 operator.

## Leaf-package evidence

On source base `7f231f1c8635c219917174b2b4d6d0b780daffda` plus this AOT patch,
the Release CUDA-sm120 leaf archive/probe builds through verified ccache 4.5.1
(no usable local sccache). Generated CXX/NVCC commands invoke the configured
launcher, the cache is reused and before/after statistics are retained. The
package links 35 explicitly named programs. `readelf` lists cudart, libstdc++,
libgcc_s and libc, not libpython or the GenerativeQC/Python runtime.

Allowed node4 RTX 5090 Slurm job **990**, finite 10-minute allocation, inherited
CUDA_VISIBLE_DEVICES=0: **33 tests pass in 11.53 s**. The native probe SHA256 is
`c9cab6606a688da1604d4bd66af1b0e90b7c40560d19af359b95b183f16e294f`.
Tests exercise all selected leaf families: independent Gaussian/Boys interval
quadrature and complex-step first-gradient differences for integral AD, Libxc
energy/gradient differences for RKS/UKS feature Hessians, independent real-space
VV10 partials, polynomial AO jets including real-spherical f polynomials,
mixed-chain primitives, ECP radial measure and harmonic orthonormality.
They also verify missing-package/null/overflow/jet rejection, no partial output
after a numerical failure, and numerical execution with unavailable Python
paths and an empty executable search path. These are leaf, not endpoint tests.

Focused CPU regression: **57 pass, 33 explicitly opt-in GPU tests skip**.
Build-time generation succeeds with imports of generativeqc/CuPy/PySCF/Torch
blocked. Compiler structure checks 550 modules with zero dependency errors;
pinned ty 0.0.82 and the complete pinned pre-commit suite pass. An older unrelated system ty 0.0.15 produces existing
compiler diagnostics and is not the qualification tool.

Node1 initially rejects a newly queued job with Slurm InvalidAccount. Pending
job 8769 is cancelled before execution and work moves to allowed node4 rather
than bypassing the scheduler. No maintenance node or visibility override is used.
Raw build/test/cache/dependency receipts are retained in the ignored local
`.artifacts/wb97mv-hessian-aot-20261015/` directory. Test duration is not an
endpoint timing, scalability result or speedup claim.

## Remaining mandatory work

An additional generic native GMRES primitive now retains every iterative
numerical scalar on CUDA, including norms, Hessenberg/Givens coefficients and
triangular solves. It accepts a borrowed native operator and publishes only
after a true unpreconditioned residual gate. It does not supply the physical
coupled-spin response maps, source operator or D/W reconstruction. Its
[implemented solver note](../implemented/numerics/2026-10-15-device-scalar-native-gmres.md)
records independent linear-solve, Python-free execution, resource and sanitizer
gates. This does not change the incomplete endpoint status.

- Bind/authenticate a live native prepared-state token and source inventory.
- Implement native CUDA tensor execution/reductions, coupled UKS CPKS and D/W
  reconstruction with true residual and complete finite-publication gates.
- Supply complete admitted integral classes and bounded primitive traversal,
  orbital/auxiliary spherical transforms and full-rank DF metric response.
- Execute moving-grid product/normalization branches, bilateral VV10 reduction
  and the scalar-ECP projector coarse/refined acceptance on native CUDA.
- Replace the CLI worker rather than describing it as native after AOT kernels
  become available; qualify actual complete hessian/hvp commands with Python,
  CuPy and compiler executables unavailable.
- Re-run independent endpoint accuracy/resource gates for RKS/UKS, DF, ECP,
  spherical bases and combinations, then benchmark the complete native endpoint.

The existing response backend that retains Hessenberg/Givens numerics on the
host must not silently weaken a stricter all-CUDA numerical contract. Host
control/scalar stopping decisions and host small-system numerical work are
distinct boundaries that need explicit acceptance.

## Rejected shortcuts

- A standalone Python bundle or embedded interpreter is not Python-free.
- Build-time generated CUDA with Python/CuPy tensor scheduling is insufficient.
- A native leaf probe is not a molecular Hessian or production capability test.
- Exhaustive unbounded per-component compilation is not automatically a usable
  production schedule. Package completeness and bounded work/storage need their
  own admission rules and qualification.
- The earlier GPU4PySCF bridge timings are not evidence for the native runtime.

## References

- `src/dft/hessian_aot.hpp`
- `tools/generate_hessian_aot.py`
- `manifests/hessian_aot_development.json`
- `tests/python/test_hessian_aot_cuda.py`
- `docs/developer/dft_hessian.md`
- Prior decision: `../implemented/architecture/2026-10-15-wb97mv-cli-compiler-runtime.md`
