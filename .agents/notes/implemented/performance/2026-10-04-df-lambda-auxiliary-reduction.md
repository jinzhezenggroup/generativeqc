# Decision: reverse the auxiliary-reduced DF residual in stages

Status: implemented
Date: 2026-10-04

## Problem

#1769 reduced auxiliary intermediates before their expensive T2 consumers in
the CCSD residual. #1789's Lambda owner still differentiated the expanded
one-Q virtual correction for every auxiliary and every Krylov action. The
230-AO force attempt recorded by #1809 was paused during corrected Lambda;
it is not a completed endpoint or a failed convergence result.

## Decision

Differentiate the existing compiler-owned prepare/Q-reduction/core cuts. A
DFLambdaActions owner copies immutable Problem/T inputs once, prepares tau and
all six complete auxiliary sums once, and retains them for its lifetime. Every
physical J-transpose action reverses the core once, copies its cut adjoints
before scratch reuse, accumulates each Q's direct T1, direct T2 AND tau
cotangents, then reverses tau once. Omitting direct auxiliary T2 dependence
was an incorrect early prototype; the expanded-AD gate exposed it.

Retained parameter VJPs use the same cuts. Virtual-factor publication reverses
the core for the final Lambda seed once and publishes each auxiliary row
separately. Retained Gram/source, metric, orbital and nuclear response stay
with their existing owners. The energy seed is included in retained parameter
VJPs; virtual energy dependence is absent in this CCSD Hamiltonian partition.

The original expanded physical primal replay and independent Lambda transpose
audit remain unchanged. No CC iteration, DIIS update, numerical denominator,
metric cutoff or eigensolver is differentiated by this change.

## Admission and diagnostics

The existing fallback arena remains live for the independent audits. Optional
storage contains tau, its cotangent, six primal cuts and six cut cotangents;
the cut sizes are v², four o²v² tensors and ov. A larger staged graph arena is
charged only when the old arena cannot contain it. Admission includes the old
host bound, packed coordinates, GMRES, detached parameters and borrowed state.
Every offset/product uses checked arithmetic before allocation.

The owner selects reduction only when its per-action contraction count is
smaller and the complete numeric bound fits. A smaller caller budget or a
CUDA allocation failure retries the original bounded schedule. An explicit
Lambda option disables reduction; the internal complete force entry point
passes that option through for same-binary endpoint comparisons. Arithmetic
and convergence errors never trigger this fallback.

Diagnostics count the extra preparation traversal, actual Q visits, generated
and accumulation kernels, contractions, selected schedule and reduced actions.
Transfers and host synchronization retain the existing owner accounting.
Cache identities cannot outlive or migrate between scientific owners.

## Evidence and qualification boundary

The compiler prototype is checked against expanded AD at arbitrary amplitudes,
independent NumPy factor-direction finite differences at two steps, and all
eight retained parameter VJPs. The corrected five-test host suite passes.
The native owner and generated CUDA translation units compile with CUDA 12.9,
strict FP64 scalar arithmetic and ccache; native numerical qualification is
still pending at this checkpoint.

Per-action semantic contraction summands, excluding one-time primal staging:

| o,v,q | Original | Reduced | Work ratio |
| --- | ---: | ---: | ---: |
| 9,221,488 | 6,432,349,619,038 | 1,210,132,747,322 | 5.3154 |
| 21,243,666 | 67,469,492,672,406 | 15,146,844,625,908 | 4.4544 |

These are sums over emitted IR contraction labels, not hardware FLOPs or
measured speedups. Stage preparation, RHS, parameter response and independent
audits must be included in complete-endpoint work and timing.

The refreshed #1809 base had lost the SCF structure allowlist edge for #1792's
pure reference_eri_policy.hpp. Restore only that planning-header edge; device
implementation and reference-export dependencies remain forbidden.

## Rejected alternatives and remaining gates

Do not retain a full amplitude Jacobian, all per-Q intermediate graphs, full
ovvv/vvvv or T3. Do not use convergence-state energy equality as an adjoint or
factor-precision oracle. Small-force qualification does not establish a large
force endpoint. The separate strict large-factor gates and source provenance
limitations from #1781/#1782/#1785 are unchanged; no public capability expands.

Complete force comparison uses benchmarks/df_ccsdt_force_endpoint.cpp. Its
input is normalized geometry/basis metadata only, with the existing finite
resource budget. The final argument selects reduced (1) or expanded (0)
Lambda in the same compiled library. All real-device execution must use Slurm.

## References

- #1769, #1789, #1792, #1809; paused compiler prototype resumed by user request.
- python/generativeqc_compiler/cc/df_lambda_reduction.py
- tools/generate_df_lambda.py and src/cc/df_lambda_cuda.cu
- tests/python/test_df_lambda_reduction.py and test_df_cc_lambda.py
