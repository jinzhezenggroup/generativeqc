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
The full library compiles with CUDA 12.9, strict FP64 scalar arithmetic and
ccache. The rebased prerequisite also contained the generic AOT query twice
in a single stub; removing its duplicate definition repairs AOT-disabled builds.

RTX5090 Slurm job 12234 passes 22 tests in 53.50 s across
test_df_cc_lambda.py and test_df_complete_force.py, including independent
H2/water/LiH force finite differences, every water coordinate, exact-budget
fallback, and failure without publication. Job 12237 passes the seven native
Lambda tests under Compute Sanitizer memcheck with zero reported errors.
On n2 PRO6000 job 2196, paired complete water endpoints have identical energy
and maximum force difference 5.3291e-15. The existing conventional stack also
passes six CUDA tests in job 12239: 14/28/56-AO water-cluster complete forces,
warm/moved geometry, energy directions, and exact/near-degenerate methane.

Frozen library SHA256:
461bd4f0e7defcc4c513bf1ecb6989ba9dce786cd2552ef6b05c1d1a2b99232a.
Build source identity:
1ac0c9753b8ce92a5c2b56ca025921c679391fa935210b788f2418733ca1ebc0.
The geometry/basis-only ethane230 input SHA256 is
9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e.

### Completed 230-AO force endpoint

PRO6000 job 2196 (GPU-54595246-dbdc-a633-dc38-7bd8eea3831a) completes the cold
geometry/basis-only ethane endpoint with all 24 nuclear force components.
Timings in seconds, from the frozen library above:

| Phase | Seconds |
| --- | ---: |
| Exact RHF | 158.875826 |
| DF source | 0.975416 |
| CCSD | 264.614006 |
| (T) amplitude/parameter/Fock response | 110.651630 |
| Corrected Lambda and parameter response | 681.013329 |
| DF factor/source/nuclear pullback | 3.691032 |
| Exact orbital and nuclear response | 671.674373 |
| Complete force endpoint | 1891.519925 |

The analytic energy derivative along the retained normalized opposing-carbon
z direction is 0.018085814482828037 Eh/bohr. Independent oracle energy central
differences at h=1e-4 and 3e-5 bohr are 0.01808580748274835 and
0.01808581799878084, giving errors 7.0001e-9 and 3.5160e-9 against the 3e-7
gate. This qualifies that large endpoint and direction, not all large systems
or a per-factor accuracy gate. The largest translational force sum is 6.8e-12.

Lambda performs 21 iterations/42 reduced actions, one preparation, 22,448 Q
visits and 1,497,478 generated/accumulation kernels. Its complete reported work
is 65,352,182,732,706 semantic contraction summands, including audits and
parameter response. Lambda independent residual is 6.1146e-13; orbital
residual 1.3606e-13 and maximum stationarity defect 1.0399e-11. There are 28
exact J/K actions and zero explicit Hessian elements. The reported complete
numeric bound is 5,741,890,195 bytes; Lambda reports 1,603,607,336 owned device
bytes and a 5,478,438,027-byte complete numeric bound.

Global RHF stability remains uncertified (flag false), and strict large-factor
precision/source-provenance gates remain open. The old expanded large-force
run was cancelled, so no measured large-force speedup is inferred from it.
After this reduction, Lambda and exact orbital/nuclear response each account
for about 36% of the completed force endpoint; energy-only (T) priorities do
not describe this force profile.

The prerequisite was rebased during qualification to 73a14a696. Its incoming
changes concern range-separated force screening/roots; the CC, methods and
compiler sources above are unchanged. Frozen measurements retain their exact
source identity. Refreshed build/regression validation is recorded separately.

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
