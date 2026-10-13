# Parallel bilateral VV10 pair traversal

Status: implemented; large complete populations remain under qualification
Date: 2026-10-13

## Problem

Large WB97M-V endpoints spend most of their time evaluating the self-consistent
VV10 all-pairs grid consumer. The 48-atom grid has about 905,726 active points;
96-atom pair evaluation alone took 116 seconds in one traced SCF iteration.
Ordered row ownership computes each off-diagonal pair twice. A previous
triangular design serialized tile waves and lost 3–10 times despite reducing
pair evaluations; symmetry alone does not imply parallel endpoint progress.

## Decision

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
weighted rows retain observable feature/weight derivatives. A separate cooperative warp evaluates
these rows once against the compact nonzero partner list, using existing generated
one-leg roots; their zero density factor has no back reaction on partner rows.
Negative-zero
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

## Rejected initial admission and live-route evidence

The first production candidate rejected positive-zero weights. Nsight Systems
on frozen integration `de4d7f564` showed all 34 feature sweeps and 12 geometry
sweeps taking the general ordered fallback (219/277 ms per sweep); the parallel
kernels returned after about 1 microsecond. This made prototype gains vacuous
on real molecules and also lost the incumbent admitted reciprocal closure.
Five independent complete molecular gates and small 3/12-atom populations
passed correctness, but do not qualify candidate throughput. The long 48-atom
attempt was stopped explicitly after this diagnosis, not recorded as a timeout.

The correction retains the stable nonzero compaction, bounded workspace and
ordered extreme-input fallback. For N nonzero live rows, Z positive-zero live
rows and tile lengths m_t, pair work is
`(N*N + sum(m_t*m_t))/2 + Z*N`. No zero row is silently screened. Device tests
now exercise positive-zero derivatives in the admitted schedule, including an
all-positive-zero population; independent molecular tests and live profiling
must be repeated before promoting the corrected candidate. The corrected
generated kernels passed 220 real-device cases, and all 110 parallel cases
passed memcheck and synccheck with zero errors; 139 host checks also passed.

## Cooperative zero-row follow-up (2026-10-14)

The initial correct positive-zero finalizer used one thread per output row.
Real 3-atom tracing showed about 40 ms per zero-row pass despite very few such
rows: scattered inactive lanes left much of the device idle. A warp now owns
one zero row, reduces its nonzero partners in FP64, and writes its three raw
sums before finalization. The zero-row predicate is uniform within each warp;
partial partner warps contribute zero. Bilateral atomics never write these
rows. There are still no extra allocations, global synchronization, or skipped
observable responses; work remains `(N*N + sum(m_t*m_t))/2 + Z*N`.

At 320,000 synthetic points with 25% positive-zero weights, the additional
schedule reduced feature/geometry time by 41.4% / 36.1% relative to the prior
correct candidate. The full arrays passed the ordered comparison; a small
independent long-double oracle also passed. These are component diagnostics.
A real 3-atom Nsight trace reduces the zero-row pass from about 40 ms to 1.55 ms;
feature/geometry bilateral kernels remain about 72 / 102 ms. It proves live
execution, not a 96-atom endpoint result.

Frozen integration `7001858f1adf0ce851b0feda7e92d5fcd48425fe`, library
`08b866dc5404feebf024011fe95a9e04b16ba55256078874d4fcda080e21fdd0`,
passes 165 host tests, 220 device admission/array cases, all 110 parallel cases
under both memcheck and synccheck with zero errors, and six independent complete
molecular E/F tests. Initial missing-ccache and shared-library fixture failures
are retained separately from successful device qualification.

The preceding correct noncooperative source `99e7c54b2` improved 48-atom cold
from 781.601 to 630.716 s and warm to about 72.9 s, but still timed out during
its final moved-warm call. It is not a completed point. The cooperative
candidate completes the full 48-atom population in Slurm job 8218 under the
original 1800-second limit: cold 576.151 s, warm median 67.707 s, moved 406.924 s,
moved-warm median 67.946 s. All 12 calls pass the independent gates, with maximum
energy error 4.548e-12 Eh and force error 6.528e-11 Eh/Bohr. A separately
reconverged moved-geometry reference supplies accuracy only; the reference
timing population remains incomplete. Ordinary-driver 3/12-atom controls also
complete all 12 calls. Their cold/warm/moved/moved-warm times are respectively
6.620/0.607/4.867/0.604 s and 43.101/4.804/32.271/4.830 s.

A 180-second diagnostic trace at 96 atoms records one 82.6 s bilateral feature
call. It is not a formal 1800-second timeout or completed population. Native/96
still needs repair and complete qualification, so #2426 remains open.

An ignored three-float expansion experiment at 320,000 synthetic rows passed
its small long-double check and baseline-array comparison, but was 3.19 times
slower for features and 3.99 times slower for geometry. Its domain had no
production overflow/underflow admission. This experiment is rejected; it does
not establish a new precision policy or qualify molecular execution. Source,
cache receipts and Slurm results remain in `.artifacts/qualification/vv10-triple-float/`.

## Invariants and revisit conditions

Keep canonical grid, density cutoff, convergence, FP64 semantics, independent
1e-8 Eh / 1e-7 Eh/Bohr gates and bounded fallback. Do not promote isolated
throughput to a claim that the 1800-second population completes. Revisit the
size/device crossover with complete endpoint measurements before widening auto.

## References

Issues #2417 and #2426; `docs/maintainer/performance_engineering.md`.
Ignored qualification records are retained under `.artifacts/qualification/`
in the benchmark integration checkout and `/tmp/qc-benchmark-audit-20261013/`.
