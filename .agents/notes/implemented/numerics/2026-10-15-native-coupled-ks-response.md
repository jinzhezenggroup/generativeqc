# Decision: build-time AD and native coupled KS response

Status: implemented component, not a complete molecular endpoint
Date: 2026-10-15

## Problem

The Hessian bridge still owns OV packing, metric/RHS construction, CPKS and D/W
reconstruction in Python/CuPy. AOT point/integral leaves and a standalone GMRES
do not remove that runtime dependency. The user requires Python only at build
time, not merely cached Python-generated kernels at execution time.

## Decision

`method/ks_cpks.py` owns one-spin tensor maps, composed by a native controller
that passes both UKS density blocks to one induced-Fock callback. Density and
energy-weighted density are differentiated by TensorIR `linearize`. The base
occupied energy is a full diagonal matrix whose tangent may be nondiagonal;
using only eigenvalue-vector tangents would lose occupied-space response.
Same-space degeneracy requires no gauge divisions. Only OV gaps are divided by
the already qualified signed right preconditioner in native device GMRES.

The full-range dynamic affine emitter is intentionally narrow. Semantic space
identities map to runtime extents; equal witness sizes do not identify spaces.
All supplied definitions, including primal AD checks, are materialized and
finite-checked. Unsupported operations, sliced domains and nonuniform witness
literals fail at generation. Empty contractions produce zero. This correctness
schedule is not a high-performance replacement for planned GEMM contractions.

The native owner authenticates before device work and again before publication,
checks finite/symmetric physical matrices without projection, and stages D/W
privately until all gates pass. All numerical arrays, Krylov reductions and
response algebra remain on CUDA. Only discrete status and terminal GMRES scalar
receipts reach host. Failed providers drain outstanding work before returning
borrowed workspace. Final-copy CUDA failure invalidates both outputs.

`CudaKsPlan` additionally exposes a full resident orbital-frame binding through
the existing scalar-only physical/canonical proof. It preserves native
column-major C and row-major D/F/S without detached export. Cached preparation
does not redo proof work. The binding is not an autonomous lifetime lease:
the enclosing method must retain the owner and reauthenticate its exact token.
The prepared-batch port exposes that proof without a concrete-owner downcast in
the response layer. A nonmovable native adapter reborrows the actual live owner
at both CPKS gates and verifies every pointer, dimension, stream and current
device; copied caller labels cannot substitute for this reauthentication.

## Rejected alternatives

- A Python/CuPy callback around native GMRES: still a Python runtime endpoint.
- Handwritten native derivative formulas: a second algebra authority beside AD.
- An eigenvalue-only W tangent: wrong for occupied off-diagonal response.
- Symmetrizing a bad physical callback: hides provider/scientific errors.
- Treating AOT witness dimensions as production dimensions: conflates distinct
  spaces and fails empty/full or differently sized spin populations.
- Advertising molecular support after component tests: the physical native
  induced-Fock, integral/grid/VV10/DF/ECP assembly is still not integrated.

## Evidence and gates

`tests/python/test_ks_cpks.py` uses independently reconverged displaced generalized
eigensolves with a weak self-adjoint spin-coupled functional. No response equation
is used by that oracle. Central steps 1e-4 and 3e-4 gate compiler maps at absolute
3e-8 / relative 3e-7. Native tests use the 1e-4 oracle and both frame layouts,
RKS/UKS, unequal/empty/full spin populations, occupied degeneracy, overlap motion,
cross-spin response, budget/alias/alignment rejection, numerical/provider errors,
stale entry/publication and iteration exhaustion. The executable also runs with
Python/compiler executables unavailable. Finite differences exist only in tests.

`tests/native/test_ks_resident_frame.cpp` exercises real prepared CUDA PBE/global
hybrid RKS/UKS owners: scalar-only proof, cached borrowing, capture/stale rejection,
independent detached numerical/layout comparison, corruption revocation and warm
recovery. These are reference-binding tests, not molecular Hessian qualification.

Raw build, Slurm, cache and sanitizer receipts are retained locally under
`.artifacts/wb97mv-hessian-aot-20261015/cpks-*`. No end-to-end speed ratio is
attributed to these components; previous bridge timings describe older sources.

Component qualification uses Release/sm120, strict FP64 `--fmad=false` and
verified ccache 4.5.1. Node2/main/PRO6000 Slurm job 3069 passes 65 tests in 19.06 s,
including multiblock runtime tensor sizes and Python/compiler-unavailable replay.
Job 3068 passes memcheck, initcheck, racecheck and synccheck with zero errors or
hazards on the restarted coupled-spin solve. The standalone qualification binary
SHA256 is `b3722226d5eace081674743757905d04405346d38283f4561191d0317e31dbaa`;
its dynamic dependencies contain no libpython or GenerativeQC public runtime.

The pre-existing broader `generativeqc_ks_resident_weights_tests` target cannot
compile at base `372696077`: it references the removed `PreparedBatch` member
`ks_transport_diagnostic`. The new focused frame test deliberately avoids fixing
that unrelated test/API migration while retaining real-owner numerical and
lifecycle coverage. The failed broader build is not qualification evidence.

## Remaining integration

Connect the authenticated native frame to the physical coupled-Fock owner,
complete integral/spherical/DF/grid/VV10/ECP response and stationary assembly,
replace the production CLI worker, then rerun molecular accuracy, Python-free
endpoint and matched GPU4PySCF endpoint performance qualification. PR #2527
must remain draft until those contracts are actually delivered.
