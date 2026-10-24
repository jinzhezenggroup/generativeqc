# Decision: distinguish budget-Fock Fast from strict-density mixed gradients

Status: implemented (local worktree; endpoint qualification recorded separately)
Date: 2026-10-14

## Problem

The public PBE0 Fast 384-AO limit belonged to derivative qualification, not
FP32 Fock capacity. Application XC warmup kept J/K strict, ordinary AUTO kept
K strict, and the Python facade clamped Fast to FP64 stopping tolerances.
Timing that implementation did not answer a request for FP32 Fock with relaxed
SCF accuracy. Earlier 96-atom strict-SCF/mixed-gradient measurements must not
be relabeled as budget-Fock speedups.

## Decision

Keep ordinary FP64/AUTO and strict-density derivative admission unchanged.
Add a private native capability for canonical exact PBE0 on the reference grid,
with independent Fock shape admission through 768 AOs/96 atoms. Both J and K
consume the existing compiler FP32 recurrence; contraction/global accumulation
stay FP64. Strict generated/resident routes remain available to the comparator.

The facade selects relaxed candidate controls only after capability/model/grid
and budget checks, honoring explicitly stricter controls. A separate optional
physical stationarity allowance preserves default final-state gates and never
loosens trace, idempotency, metric or canonical diagonalization. Mixed proposals
are bounded and cannot publish without fresh FP64 physical refinement/projector
closure. Relaxed candidates keep FP64 forces, separate from the retained
strict-density derivative qualification. Missing arithmetic evidence or tighter
targets require an observable audit; numerical failures retain the failed
attempt and bounded strict fallback.

The first small endpoint exposed a second Python force boundary: its historical
residual and canonical-density gates rejected a state already accepted by the
native budget owner. Transport the verified optional allowance through a
token-checked private snapshot query, not mutable Calculator controls. Only
force callers opt in; default snapshot/response contracts remain strict.
Weighted-density and orbital-frame algebra stay strict in Python as well.

The next source-matched small endpoint exposed a retained compiler guard that
still rejected every mixed materialized call with a separate K sink. Its NaN
sentinel correctly forced a native strict retry, leaving zero FP32 recurrence
and exchange work. Admit restricted full-range separate J/K sinks together;
unrestricted K, range separation and mixed weighted-Fock products remain barred.
Add a real-device retained-FP64-ERI/independent-host-orbit probe, including a
cross-pair density with zero J but nonzero K, so either that guard or accidental
Coulomb-only pruning cannot silently remove exchange work again. Benchmark JSON
represents rejected nonfinite diagnostics as null with explicit path/value tags;
qualification continues to inspect the unmodified numerical endpoint.

## Invariants

- Actual positive recurrence and FP32 exchange work is required, not AUTO labels.
- A shared J/K recurrence scatters K exactly once. Coulomb-only zero-density
  pruning must not discard exchange-active canonical or materialized orbits.
- Legacy final-state consumers retain their gates when no allowance is supplied.
- Neither a relaxed tolerance nor a shape admission is a per-molecule error bound.
- Retain first calls, failed endpoints and complete forces; never handicap FP64.
- All real GPU qualification runs through finite Slurm allocations.

## Evidence

Host admission, schedule restoration, native product-policy, facade fallback and
benchmark observable/work-count tests exercise the new seams. The reusable
runner is `benchmarks/pbe0_fast_fock.py`; independent GPU4PySCF owns the oracle.
Source-matched raw local evidence lives outside this worktree in
`../evidence-pbe0-fp32-fock96-20261014/`. Qualification and timing are separate:
no speedup is claimed from source changes or effective-bit metadata alone.

Source-matched qualification completed on 2026-10-15, RTX 5090 / CUDA 12.9.1,
through finite Slurm jobs 8489–8492. The independent 96-atom / 768-AO exact
PBE0 endpoint passes all energy/force gates. Every Fast call executes one solve,
12 mixed J/K Focks and 5 strict physical Focks, with zero numerical retries.
Three alternating repeats give FP64 97.183134 s versus Fast 210.505000 s:
FP64/Fast is 0.4616666322, so this implementation is 2.166 times slower, not
an acceleration. Maximum Fast oracle differences, including first calls, are
2.34195e-9 Eh energy, 2.79617e-7 Eh/bohr maximum force and 1.04869e-7 force RMS.
The 25-AO Cartesian water and 96-AO spherical tetramer also qualify but are
slower. Arithmetic qualification must not be relabeled as performance success.

The mixed path recomputes integral sources where optimized FP64 can reuse them.
The first comparable recurrence census is 1,873,221,716 for FP64 versus
17,717,684,496 mixed plus 1,476,473,708 strict for Fast; missing recurrence rows
in later FP64 records are unknown, not zero. These counts and the source route
identify a work-reuse deficit, without claiming a complete profiler attribution.
Retain the optimized comparator and investigate separately qualified float
resident sources before promising acceleration. Final validation includes
245 passing host tests, analytic final-state probes, and the real-device
materialized J/K/derivative test with exchange-only active density coverage.

## Revisit when

Broader molecules, grids, spin/provider domains or actual lower-precision
contraction/storage have independent endpoint evidence. A float resident source
may avoid repeated recurrence but needs separate storage/error/lifetime gates.

The subsequent precision-bound cache design is recorded in
[the mixed-resident source decision](../performance/2026-10-15-pbe0-mixed-resident-source.md).
It supersedes the recompute-only implementation, not these historical timings.
