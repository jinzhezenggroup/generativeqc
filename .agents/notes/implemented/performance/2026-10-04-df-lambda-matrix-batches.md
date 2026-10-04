# Decision: compiler-owned FP64 matrix adjoints with bounded Q batches

Status: implemented
Date: 2026-10-04

## Problem

#1818 matrix-lowered DF CCSD residuals, but its Lambda response still issued
one scalar staged program per auxiliary slice. The 230-AO force record had
675.220 s in Lambda/parameter response, 21 iterations/42 actions, and 488
auxiliary factors. The converged iteration count does not explain the per-action
cost. Orbital response remains a separate 667.559 s phase.

## Decision

Reuse the staged adjoints from #1817 and the packing/GEMM lowering from #1818.
The compiler lifts only factor-dependent values to an explicit leading Q axis;
amplitudes and seeds remain shared. The three auxiliary programs (primal cuts,
amplitude adjoints, factor cotangents) execute in bounded batches. The core,
prepare and retained-parameter adjoints use the same matrix lowering. Shared
free Q axes lower to strided batched GEMM; one-sided Q axes can fold into M/N.

The default requested batch is 8, capped by naux. Admission tries successively
smaller batches, then scalar staged execution, then the original expanded path.
The original physical replay and expanded independent Lambda audit are preserved.

A generated consumer fuses all output accumulations with an ordered Q sum in
one kernel per batch. Factor cotangents retain their Q rows. No scientific
formula, orbital frame, method, cutoff or screening threshold changes. Host
Krylov transfers and unscreened orbital response remain unchanged.

## Capacity and arithmetic invariants

- No all-Q intermediate graph or ovvv/vvvv tensor. IR packing/broadcast arrays
  participate in normal liveness planning. Matrix scratch is optional and fully
  charged alongside scalar replay storage, retained cuts, host GMRES, borrowed
  reference/CC inputs and detached publications.
- Add a 96 MiB provider allowance, measured on handle creation as in #1818.
  Use FP64 pedantic cuBLAS, host scalars and zero optional workspace.
- Only allocation/admission failure selects the scalar fallback. CUDA/BLAS
  execution and arithmetic errors propagate. Tail batches use their actual Q.
- A finite audit follows each GEMM; every fused accumulation addition updates
  the sticky error flag. Only a complete audited action may be published.
- The generated operation hash distinguishes matrix/shared/independent programs.
  Contraction summands and GEMM calls are not FLOP counts. Generated-kernel
  counts include finite audits, but exclude unknown provider-internal kernels.
  Packing-output bytes count written transpose/broadcast elements, not a measured
  memory-traffic total. Smaller Q batches can have MORE kernels because of packing.

## Validation

Compiler tests compare unequal occupied/virtual extents, tail batches, factor
rows and the complete adjoint with the independent expanded transpose (9 passed,
including existing reduction tests). Slurm RTX 5090 native tests cover full
integral parameter/factor oracles, determinant and resolved-energy derivatives,
exact/short memory boundaries, and complete DF-CCSD/(T) nuclear forces including
all water coordinates (22 passed). Separate nonfinite/memcheck validation and
same-GPU large-force endpoint results are retained with the PR evidence.

The n2 endpoint pair uses one immutable library/input and one Slurm allocation;
large performance results must be reported only after both complete. Small
correctness tests do not qualify large per-factor precision or global RHF
stability. The existing unqualified factor `atol=rtol=3e-10` gate and false global
stability diagnostic are not changed by agreement in total energy/force.

## Alternatives and next boundaries

Device-resident Krylov could remove repeated dense seed/action transfers, but
it cannot replace optimizing the existing contraction graph. Fixed Schwarz
screening of orbital J/K changes integral truncation and requires independent
force/Z/FD qualification; it is not enabled here. All-Q caching is rejected due
to large resident intermediates. Revisit batch size/cost selection using measured
complete endpoints on each GPU, rather than importing another GPU's calibration.

References: #1769, #1792, #1809, #1817, #1818; #1781/#1782/#1785 reviews retained.

## Completed large endpoint evidence

n2 Slurm 2198 used one frozen binary and one PRO 6000 UUID for matrix/scalar
force calls and a separate cold energy-only call. Lambda/parameter response
fell from 679.436 to 274.307 s (2.477x), with unchanged 21 iterations/42 actions.
Complete force time fell from 1796.022 to 1352.053 s (raw 1.328x). Reference RHF
also differed by 37.284 s; do not attribute that variation to Lambda. The
standalone energy call took 307.998 s, including only 4.942 s for (T) energy.
The force call's (T) energy/response phase took 110.515 s. Its source response
took 3.685 s and exact orbital/nuclear response 668.622 s.

Batch/program visits fell from 22448 to 3660 while auxiliary slices remained
22448; expanded physical replay and independent audit still visit every Q.
Semantic contraction summands fell from 6.535e13 to 6.401e13. For one transpose,
the compiler-derived work only decreases 2.585%; asymptotic complexity is
unchanged. Gains therefore come mainly from matrix execution and launch
structure. Generated/packing/accumulation/audit kernels fell from 1454004 to
372357, plus 78066 GEMM calls whose internal kernels are unknown. Complete
numeric capacity rose from 5.102 to 6.433 GiB; this is not measured GPU peak.

Both complete cold forces pass the independent 3e-7 Eh/bohr directional FD gate
at h=1e-4 and 3e-5. The matrix errors are 2.60e-9 and 7.92e-9. The *additional*
3e-9 cold scalar/matrix force sanity line failed (max 5.7838889233607915e-9).
Job 2198 therefore exited nonzero after completing all endpoints. Retain that
failure explicitly; total-energy agreement (2.84e-14 Eh) cannot erase it.
The follow-up benchmark compares both complete Lambda/parameter solves with one
shared physical Problem/T/triples seed, rather than attributing cold-force
variation to an unproven cause or silently relaxing the line.

The production branch was rebased onto #1818 at 7fb64d4c4. n2 job 2201 passes
24 native/complete-force tests, and job 2204 completes the 230-AO force in
1359.881 s with FD errors 4.24e-9 and 6.27e-9. Its different GPU is excluded
from paired timing ratios. Frozen source patch, input, library identities,
compiler polynomials and numerical records are retained under
`benchmarks/results/df-lambda-gemm-20261004/`. Two upstream host probes were
adapted to the current CC signatures and cuBLAS ownership; both pass. No
production library changes were needed for those CI test adapters.

Shared-state n2 job 2207 passes all 12 response arrays at the predeclared
`atol=rtol=3e-10` gate. The largest absolute difference is 1.39e-17 and both
independent expanded Lambda residuals are 6.11e-13. Matrix/scalar Lambda take
274.363/679.254 s with the same 21 iterations/42 actions. This bounds the
changed schedule difference at fixed inputs; the exact source of that
small cold-force variation remains unlocalized. It is accepted here against
the independent complete-force FD gates and shared-state comparison, while
retaining the failed stricter cold-pair line and the separate outstanding
large source-factor qualification. No extra node3 job was submitted.

## Review follow-up: explicit matrix selection pending promotion qualification

The matrix schedule remains available through explicit options and the retained
benchmark commands, but ordinary Lambda and complete-DF-force defaults now keep
the scalar staged schedule. The shared-state diagnostic explicitly requests the
matrix solve before its scalar comparison; it must not depend on a production
default to select the candidate.

This is a temporary promotion boundary, not a finding that the matrix equations
fail the formal force gate. The failed supplemental 3e-9 cold-pair comparison and
the passing formal 3e-7 directional gate retain their original meanings.

The internal complete-force API is cold-only: it reconstructs RHF/source/CC state
on every call and exposes no prepared warm-force API. Promotion evidence should
therefore characterize repeated calls to that supported cold API, including the
same geometry with warmed runtime caches and changed input geometry, alongside
the supported batch/budget fallbacks. It must not claim warm source reuse or
require inventing a new warm endpoint. Retain matched scientific settings, work,
complete timings and unchanged independent numerical gates. The one cold pair
and fixed-state Lambda comparison above remain useful evidence for their stated
domains, rather than an unrestricted default-promotion record.
