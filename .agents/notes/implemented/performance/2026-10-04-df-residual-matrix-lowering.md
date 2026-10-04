# Decision: expose bounded DF residual contractions as FP64 matrices

Status: implemented; native qualification pending
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
Native and complete force qualification is pending at this checkpoint.

Packing adds data movement and may lose on small shapes. No speedup is claimed
until matched complete endpoints pass. This changes execution only: exact RHF
orbitals plus DF correlation remain the method. Factor precision, retained
source provenance, and global stability limitations from #1781/#1782/#1785
remain separate gates. Total energy agreement does not certify factors.

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
