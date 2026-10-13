# Parallel bilateral VV10 pair traversal

Status: proposed (complete molecular qualification pending)
Date: 2026-10-13

## Problem

Large WB97M-V endpoints spend most of their time evaluating the self-consistent
VV10 all-pairs grid consumer. The 48-atom grid has about 905,726 active points;
96-atom pair evaluation alone took 116 seconds in one traced SCF iteration.
Ordered row ownership computes each off-diagonal pair twice. A previous
triangular design serialized tile waves and lost 3–10 times despite reducing
pair evaluations; symmetry alone does not imply parallel endpoint progress.

## Candidate

Expose both feature legs of the existing canonical TensorIR pair DAG. Generated
CUDA templates assign unordered 128-row tile pairs to persistent warps, with
four rows per lane. Off-diagonal pairs evaluate once and accumulate both ends;
diagonal tiles retain ordered evaluation. For N active rows and tile lengths
m_t, scalar evaluations are (N*N + sum(m_t*m_t))/2. No pair matrix or serial
wavefront is introduced. FP64 warp reductions and atomic additions change
summation order and bitwise reproducibility, requiring independent qualification.

The runtime borrows O(N) output rows for raw feature and geometry sums, then
finalizes them on the caller's stream. All live input spans and output spans
must be disjoint; an expired input density-gradient allocation may still serve
as point-derivative output. Final scalar energy is written after all consumers.
There is no workspace growth, allocation, CPU fallback or host synchronization.

The candidate selects large (at least 65,536 grid points) sm_120 VV10 feature
consumers with the molecular row mask. `GENERATIVEQC_VV10_PAIR_SCHEDULE=ordered`
retains the incumbent; explicit `parallel` also permits small diagnostic grids.
Selection never bypasses numerical or output-lease admission. Existing bounds,
coefficient limits and the original ordered fallback remain. Positive-zero
weighted rows require fallback because their feature/weight derivatives remain
observable although the compact partner list excludes zeros. Negative-zero
screened rows retain the molecular mask contract. Other variants and public
unmasked fixed-grid consumers retain their established schedules.

## Evidence before molecular qualification

The isolated 128-row prototype at 320,000 points reduced features from
3.123488 to 2.279462 seconds and geometry from 4.315792 to 3.229075 seconds.
At 80,000 points the reductions were about 31.8% and 30.1%. Large-array
comparisons to the ordered schedule passed rtol 2e-12 / atol 2e-11; small
129/257-point comparisons to an independent long-double rational oracle passed.
The earlier 32-row prototype improved features only 3% and slowed geometry 9%.
These are component diagnostics, not complete benchmark qualification.

The production-generated scalar roots are checked against independent 90-digit
Decimal derivatives, including admitted-scale extremes. Device admission tests
compare full feature/geometry arrays, partial tiles, signed/masked/zero rows,
out-of-bound fallbacks, error publication and output canaries. Host tests execute
the actual alias gate. Qualification has passed 139 host tests and 196 device
checks, plus 98 parallel cases under each of memcheck and synccheck (zero errors).
Initial fixture prerequisite/architecture failures are retained separately.
Complete molecular E/F, changed-geometry and fixed-seed
replays at original deadlines are still required before claiming issue closure.

## Invariants and revisit conditions

Keep canonical grid, density cutoff, convergence, FP64 semantics, independent
1e-8 Eh / 1e-7 Eh/Bohr gates and bounded fallback. Do not promote isolated
throughput to a claim that the 1800-second population completes. Revisit the
size/device crossover with complete endpoint measurements before widening auto.

## References

Issues #2417 and #2426; `docs/maintainer/performance_engineering.md`.
Ignored qualification records are retained under `.artifacts/qualification/`
in the benchmark integration checkout and `/tmp/qc-benchmark-audit-20261013/`.
