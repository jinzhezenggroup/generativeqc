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
