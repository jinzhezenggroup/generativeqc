# Decision: expose bounded DF residual contractions as FP64 matrices

Status: implemented; one 230-AO internal energy/force endpoint qualified
Date: 2026-10-04

## Problem

#1769 already reduced auxiliary intermediates before expensive T2 consumers.
Its generated CUDA contractions still assign each output a serial reduction.
The remaining leading virtual ladder costs 2 q o² v³ semantic summands per
residual evaluation. #1792's complete energy endpoints spend 265.24/418.41 s
and 3737.38/4224.52 s in CCSD on PRO6000. Those phase measurements motivate
improving the residual before pursuing the much smaller energy-only (T) phase.

## Decision

The compiler derives matrix groups from the existing binary TensorIR einsums.
Explicit transpose nodes pack only existing tensors, then restore each output
axis order. The shared arena allocator accounts for every packing buffer.
No intermediate has more than two virtual axes, and no all-Q tensor graph,
ovvv, vvvv or amplitude Jacobian is introduced. Shared value numbering can
reuse newly identical packed contractions; final IR queries count this actual
work. The leading asymptotic contraction cost is unchanged.

The native residual owner supplies stream-bound FP64 cuBLAS with pedantic math,
host alpha/beta and zero workspace. It audits each matrix result for nonfinite
values before arena reuse. The sticky flag spans preparation, all Q slices and
the core, preserving failure even when later outputs are finite. The unchanged
expanded replay remains the independent convergence check.

## Admission and fallback

Admission sums the existing host/reference/amplitude/DIIS state, packed arenas
and a 96 MiB provider allowance, following the existing RHF matrix owner policy.
Compiler-derived flattened dimensions must fit the provider's signed integer
interface. Handle initialization measures device allocation against the allowance.
An oversized allowance use, provider allocation failure or optional arena OOM
selects scalar hoisted execution; a tighter caller budget can further select
the original bounded Q schedule. Arithmetic and driver errors do not select a
fallback. Every fallback is replanned against the same complete numeric budget.

Diagnostics report selected matrix execution, actual GEMM calls and scalar
summands, packing read+write bytes, provider allowance and complete capacity.
The internal cold endpoint can explicitly select scalar or matrix residuals
in the same library, independently of Lambda reduction and energy/force mode.

## Validation and boundary

Four random-amplitude shapes, including o != v and unit dimensions, agree with
the independent expanded residual within 2e-12. Runtime work and rank checks
cover (9,221) and (21,243); these are compiler checks, not large-device timings.
Native solver tests compare amplitudes and energy with the scalar schedule and
determinant oracle, and exercise exact-budget fallback and sticky overflow.
RTX5090 Slurm job 12236 passes all 14 CUDA solver tests in 13.02 s, including
the exact-budget matrix-to-scalar fallback and all sticky-overflow cases.
Job 12238 passes all 12 complete-force tests in 31.09 s, including independent
energy finite differences and every water coordinate. Frozen-library test mode
links the probe to the same full library as the endpoint benchmark; the default
fixture retains standalone code-generation coverage.

On PRO6000 job 2197, scalar/matrix complete water forces differ by at most
3.5527e-15. RTX5090 job 12240 completes the entire matrix water force endpoint
under Compute Sanitizer memcheck with zero errors.

The 230-AO cold energy pair completed sequentially on the same PRO6000 (UUID
GPU-4b4be14f-ec84-6736-a7d8-968d62900c72). Both use 38 primary residual
evaluations plus independent replay. This is one paired measurement, not a
median or a cross-GPU calibration:

| Phase | Scalar seconds | Matrix seconds |
| --- | ---: | ---: |
| Exact RHF | 145.935925 | 159.010956 |
| DF source | 0.969436 | 0.974524 |
| CCSD | 263.134257 | 170.829741 |
| (T) | 4.919768 | 4.894072 |
| Complete energy | 414.959598 | 335.709511 |

The CCSD and complete energy speedups are 1.5403x and 1.2361x. Total energies
differ by 6.3949e-13 Eh. Complete numeric capacity rises from 2,922,911,728 to
3,278,103,280 bytes. Actual contraction summands are 44,105,252,811,046 and
44,063,877,939,352, so this is an execution improvement rather than a large
arithmetic reduction. The matrix run records 317,528 GEMM calls,
39,019,400,687,592 GEMM summands and 7,347,698,382,880 packing read/write bytes.
The same job and GPU subsequently completed the cold 230-AO force endpoint
with matrix residuals and reduced Lambda in 1,750.505951627 s. All 24 force
components were returned. The geometry/basis-only input has o=9, v=221, q=488
and SHA256
9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e.
The exact invocation was:

```text
./df-force-endpoint ethane230.input ethane230-matrix-force.json 1 1 1
```

| Complete force phase | Seconds |
| --- | ---: |
| Exact RHF | 122.324041 |
| DF problem/source | 0.976212 |
| CCSD | 170.737427 |
| Triples amplitude/parameter/Fock response | 109.987902 |
| Lambda and parameter response | 675.219956 |
| DF source/nuclear response | 3.676782 |
| Exact orbital/nuclear response | 667.559231 |

Lambda and orbital/nuclear response now account for 38.57% and 38.14% of
this complete force measurement. CCSD accounts for 9.75%. The measured
DF source fraction cannot establish the cost of conventional post-HF source
re-preparation, which follows a different ownership path.

The Lambda residual is 6.1158e-13, the orbital Z residual is 1.3632e-13,
and stationarity is 1.3321e-11. There are 21 Lambda iterations, 42 reduced
actions, one preparation, 22,448 auxiliary visits and 1,497,478 Lambda kernels.
Lambda semantic work is 65,352,182,732,706 summands; neither visits nor
kernel counts are FLOPs. The complete numeric capacity bound is 5,741,890,195
bytes, including the Lambda phase bound of 5,478,438,027 bytes. There are
28 J/K response actions and zero explicit orbital Hessian elements.

For the independent carbon stretch direction (carbon 0 z=+1/sqrt(2),
carbon 4 z=-1/sqrt(2)), -dot(forces,direction) is
0.01808581310788394 Eh/bohr. The independent retained #1809 energy differences at
1e-4 and 3e-5 bohr give absolute errors 5.6251e-9 and 4.8909e-9 Eh/bohr,
both below the 3e-7 gate. The largest translational force sum is 8.6742e-12.
Against #1817's scalar-residual force endpoint, the maximum component
difference is 3.2682e-9 Eh/bohr and energy difference is 1.8474e-13 Eh.
Those force runs used different GPU UUIDs, so no force speedup ratio is
inferred. One large geometry and one independent direction at two steps do
not establish all-system or all-coordinate large-force qualification.

Ignored local evidence is retained under .artifacts/df-gemm/:

| Artifact | SHA256 |
| --- | --- |
| ethane230-matrix-force.json | 794bf8854246241f04c6f946d3d2bc828a8db78f5a95ed3d7c59f3e66d2c8600 |
| ethane230-force-fd.json | 54e11eae63336750e4be0c4b32c4ef5a29a3e8a9c5a5036d48a9c57866dab89c |
| ethane230-matrix-force.trace.jsonl | d2d25e912227a41cb79d9df453270e9285d960dc7512c311fe1c91661e29d15d |

Frozen library SHA256 is
a92445a75857471438c9a752554a27a32964abc09bbc02b7e680ff5d6c9d7539;
build source identity is
a28ce46d0ba8b65868503b20e51c977e2b9dfb7e8985a090ee55a848c1931bd3.
The frozen endpoint executable SHA256 is
4caa2610580ccaf16341b6e9b1ddbd34aba24441eb222a50d8fe7f992d624fd5.

After the #1809 base rebase, the complete library was rebuilt with ccache.
RTX5090 Slurm job 12242 passes 26 native solver/complete-force tests in
43.10 s; job 12243 completes the matrix water force endpoint under memcheck
with zero errors. The refreshed library SHA256 is
cb9cbcd07edae94374f0ab70854f7bad541a7e6e2e45bfe3d565e121ec3a1a7b,
with source identity
0a3dbd45cb27799a755a3d637db150e4c3cce2a97f5b34deb199804edd5cd35c.
The rebase changed prerequisite integral paths, not the measured CC/compiler
equations. Refreshed small-device regressions are not substituted for the
frozen large-endpoint timing provenance above.

The compiler/runtime queries give the following per-residual figures. Arena
bytes include preparation, auxiliary, accumulation and core, but exclude
inputs, replay, DIIS, retained host state and the provider allowance; they are
not complete endpoint capacities.

| o,v,q | Scalar summands | Matrix summands | Scalar arena bytes | Matrix arena bytes |
| --- | ---: | ---: | ---: | ---: |
| 9,221,488 | 1,037,733,074,263 | 1,036,644,261,850 | 895,480,896 | 1,150,009,248 |
| 21,243,666 | 11,836,059,199,641 | 11,828,905,472,340 | 5,919,537,792 | 7,604,262,624 |

Packing adds data movement and may lose on small shapes. The measured speedup
is limited to the paired energy endpoint above. This changes execution only: exact RHF
orbitals plus DF correlation remain the method. Factor precision and retained
source provenance limitations from #1781/#1782/#1785 remain separate gates;
the strict large-factor atol=rtol=3e-10 gate is not qualified by these endpoint
results. Global RHF stability is also uncertified (diagnostic flag 0).
Total energy agreement does not certify factors.

## Rejected alternatives

Do not repeat #1769's hoisting work, expand vvvv, cache each Q's T2 graph, infer
orbital symmetry from equal shapes, lower reductions with one-sided labels as
GEMM, or interpret task counts/packing bytes as FLOPs. Do not calibrate PRO6000
with #1787's RTX5090 model. Optional precision and approximate correlation
methods require separate scientific contracts.

## References

- #1769, #1792, #1809 and Lambda reduction #1817.
- python/generativeqc_compiler/cc/df_gemm.py
- tools/generate_df_ccsd_hoisted.py and src/cc/cuda_solver.cu
- tests/python/test_df_gemm.py and test_df_cc_native_solver.py
- benchmarks/df_ccsdt_force_endpoint.cpp
