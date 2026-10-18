# Decision: reuse Direct force moments and factor Hermite contractions

Status: implemented on the qualification branch; source-matched qualification retained below
Date: 2026-10-16

## Problem and source boundary

The order-six component-local spatial consumer repeats low-degree rho/Boys
factors for each requested root. The order-seven cooperative consumer already
shares its Coulomb DAG across AO components, but expands six Hermite indices
for every independent atomic response. Its Coulomb root depends only on the
three sums of those indices. Task enumeration and page-width changes cannot
remove that repeated contraction arithmetic.

The qualification baseline composes master `9a92cad27`, #2531 `01ed327f4`,
and #2532 `fb37f6d1c` in `6d8192be8`. In particular, it includes #2532's
subsequent Direct/1088 resident-final-validation hold. Neither the user-supplied
historical table nor #2538's earlier frozen image is relabeled as this baseline.

## Changes and invariants

- Prepare the eight `(-2*rho)^n F_n` roots once per order-six spatial box.
  Its existing Wick consumer then borrows those roots and the already prepared
  coordinate powers, including for degrees zero through four. Nonspatial and
  order-four/five consumers retain their incumbent arithmetic. The box remains
  bounded by 64 doubles; no shell/primitive tensor or CTA synchronization is added.
- Feed the existing compiler-owned axis convolution into the cooperative
  derivative contraction. Compute each one-dimensional pair convolution before
  consuming the three-axis scalar Coulomb states and their response views.
  This changes floating-point association, not the mathematical integral.
- Retain independent-center translation, signed source coefficients, RKS/UKS,
  Cartesian/spherical packing, exact screening, all component slots and the
  shared recurrence's publication/retirement barriers. No functional, molecule,
  basis size or GPU name enters either algorithm.
- Existing cache, maximum-angular, recurrence-mode and bounded-domain guards
  continue to own admission and fallback. No approximation, CPU reference,
  density fitting, precision relaxation or convergence change enters production.

## Preliminary evidence, not endpoint speedups

Finite n4 Slurm jobs 1000--1003 preserve the assigned visible device. On the
32768-primitive ddds probe, the incumbent median is about 0.428 ms and
preweighted moments about 0.308 ms. On 512 synthetic dddp shell quartets
(8192 primitive products), the cooperative incumbent is about 285.51 ms and
the convolved consumer about 75.38 ms. These are isolated device-event probes,
not complete E+F observations or achieved occupancy measurements.

A compiled instrumented execution of the actual generated contraction over
all 648 Cartesian dddp components records the following source-level work.
The scalar wrapper counts entered operations, not GPU machine instructions or
FP64 FLOPs after compiler optimization; its arbitrary inputs are not an oracle.

| Work per component inventory | Six-index | Axis convolution |
| --- | ---: | ---: |
| Scalar/jet additions | 36288 | 27504 |
| Scalar/jet products | 194184 | 62244 |
| Hermite coefficient reads | 127008 | 13608 |
| Coulomb root reads | 36288 | 18432 |

Independent 70-digit Wick sums check every order-six box and response face,
including zero coordinates/rho. Existing all-center full/LR finite differences
remain required. The convolution's independent correlated-Gaussian quadrature
and raised/lowered-polynomial derivative oracle now also cover three dddp
axis distributions. The standalone real-GPU retained-AD/independent host-orbit
suite covers both spins, both source layouts, repeated atoms and exact work.
Memcheck and synccheck report zero errors. The actual linked candidate also
passes six independent Libcint GPU cases, including spherical/Cartesian,
RKS/UKS and separate/combined J′/K′ consumers with order-seven/eight work.
The final focused host suite passes 318 tests; other overlapping host-test
records must not be added to that count as disjoint coverage.

## Rejected alternative

A 28-case generated scalar DAG shared all roots of each order-six box without
CTA barriers. It passed independent root/force checks, and the primitive probe
used fewer registers (216 to 168), but median device time rose from about
0.428 to 0.443 ms. Its 390-kB generated specialization is archived rather than
retained. No hardware-counter evidence establishes why it regressed; fewer
source expressions/registers alone do not qualify a default.

The previously rejected mixed/cooperative-six and degree-seven six-only
scheduling experiments are not restored. Their retained negative evidence
remains authoritative for their original contraction arithmetic.

## Qualification records

Ignored raw sources, compile commands/cache statistics, failures, probes and
sanitizer logs live under `.artifacts/direct-reuse/`. Remote work is under
`/data/jzzeng/gqc-direct-reuse-20261016` on n1/n4/n5. Initial benchmark preflights
reject an accuracy-only reconstruction used as a complete timing reference and
an accidentally selected portable build; neither returns a timed endpoint.
The remedy is the retained complete independent references and matching
Release/sm_120 official profiles, not bypassing either admission guard.

The evidence retains cold, five warm, moved, five moved-warm calls, the
original 1800-second population limit, actual SCF/Fock counts and independent
1e-8 Eh / 1e-7 Eh/Bohr gates. Diagnostic host and device ledgers are separate;
neither nested wall intervals nor device kernels are added to parent wall time.

The matched 96-atom diagnostic reduces order-six device time from
21.754343 to 17.441551 seconds and order-seven from 28.034387 to
7.305988 seconds. Generated-low and materialized-order-eight times remain
about 4.626 and 5.772 seconds. Exclusive J′ dispatch/wait falls from
60.196200 to 35.154146 seconds. The intrusive complete diagnostic falls
95.189178 to 70.208878 seconds, distinct from clean population timings.

Entered scientific work remains identical: 2,400,504,912 order-six primitive
products, 1,250,544 order-seven primitive products/Coulomb preparations and
95,976 order-eight primitive products/Coulomb preparations. Cooperative
bra/ket preparation and component-contraction counts also remain unchanged.
Across all 28 unweighted spatial boxes, entered source products decrease
16573→15984 while additions increase 2300→2956. Those are source-algebra
counts, not endpoint totals or measured hardware instructions. Nsight Compute
cannot access hardware counters (`ERR_NVGPUCTRPERM`, finite n4 job 1008),
so no GPU instruction/FLOP reduction is claimed from the source counts.

Order-six launch metadata remains 255 registers/thread and 128-thread CTAs.
Order-seven registers change 222→214, retaining 256-thread CTAs and
416 static plus 42984 dynamic shared bytes. These are not measurements of
achieved occupancy, memory traffic or full process peak memory.

Both changes are automatic in their existing admitted generated consumers on
this branch. Master is unchanged; PR #2541 remains stacked on the frozen
composition pending #2532 and restacking. Requalify material source changes
after resolving that dependency. The next substantial force optimization is
compiler-owned primitive-pair preparation reuse across Cartesian components,
not another page-width or CTA-size experiment. Old cooperative-six trials
were removed and are not available behind a runtime tuning flag.

## Complete endpoint acceptance

The clean 96-atom complete cold/warm/moved/moved-warm observations are
411.131/90.752/260.226/90.920 → 386.093/65.555/234.750/65.669 seconds.
Warm and moved-warm medians improve 27.76% and 27.77%. The matched 48-atom
observations are 76.637/21.558/60.232/21.566 →
70.499/15.286/51.185/15.358 seconds. Its moved SCF trajectory changes
14→13 iterations/builds and is explicitly not normalized away. The 96-atom
cold/moved trajectories remain 24/24 and 13/13 in both arms; all warm calls
take one iteration/build. PBE0 controls improve about 6.3% warm; HF changes
are within 0.21% in every phase. All 96 primary calls finish, with independent
maximum errors 3.86535e-11 Eh and 6.01639e-11 Eh/Bohr.

Reviewed evidence is in `benchmarks/results/direct-force-reuse-20261016/`.
Every primary energy/force array and independent geometry oracle is retained
losslessly in standard NumPy/XZ with a standalone 96-gate replay. Detailed
native documents, iteration histories and raw profiling/build records remain
hash-bound locally. The compact public selection respects the existing
aggregate storage cap without changing unrelated historical evidence. Numerical
acceptance and scoped endpoint gains are explicit, not a universal interleaved
performance promotion with full compile/memory budgets. The candidate remains
about 2.47× slower than the historical 26.59-second GPU4PySCF warm target.

## Current-source requalification after dependency resolution

The preceding measurements remain historical. After #2532 merged, clean master
`5348b9e66` and synchronized candidate `4bdffe3ce` were built as official
Release/sm_120 images with verified ccache4.5.1. Native identities are
`4fef53cce012f4fcabdd307247350fc61b3f1d4d9d2e8999400f3d2e5532ff96`
and `ac850b60e98f89be8fa69922f7a2779d2da9c812a5a5dee61be5a88bcd14471b`,
from1753 manifest inputs each. Later source movement is not relabeled as this
campaign. PR #2541 now targets master; no merge is performed by this agent.

All96 current-source primary calls complete and independently pass1e-8Eh and
1e-7Eh/Bohr gates; maximum errors are3.86535e-11Eh/6.01426e-11Eh/Bohr. Full
per-call arrays, both independent geometry oracles, actual iteration/build
counts and original1800-second population limits are retained in the current
bundle. The original frozen bundle remains at immutable commit
`6369de44fe1f30759b3d07c8f267f4a02d0d70dd`, same repository path.

Current96 cold/warm/moved/moved-warm seconds are
411.282020/90.798641/260.105874/90.800574 to
386.116851/65.477918/234.961981/65.707564. Warm/moved-warm improve
27.89%/27.64%. Current48 values are75.817234/21.438233/56.877101/21.439160
to69.934962/15.226590/50.849639/15.253462. Unlike the historical pair, both
current48 moved trajectories are13/13; current96 remains24/24 cold and13/13
moved. All warm calls are1/1. PBE0 warm/moved-warm improve6.59%/6.95%; HF
changes are within0.58% in every phase. No trajectory is normalized away.

Separate matched n1 diagnostic profiles give order-six21.968210→17.549077s,
order-seven28.218229→7.375439s and order-eight5.809917→5.824592s. Primary96
populations use the same n4 GPU. Diagnostic stationary caches are isolated
from clean endpoint caches. Mutually exclusive host owners and separate device
windows reconcile; intrusive capture is not substituted for clean timing.
Entered primitive products remain2,400,504,912/1,250,544/95,976 for orders6/7/8.
No missing preparation count is inferred from enumeration.

588 focused host checks and six actual-linked independent Libcint GPU cases
pass after synchronization. Follow-up warp-local Boys/prefactor and subgroup
Wick-term sharing probes have no consistent useful device gain despite lower
preparation work; neither is enabled. The separately tracked response-factoring
continuation is PR #2549. This two-arm packet does not claim its results.
The existing aggregate storage cap is respected by superseding only this PR's
own packet while retaining its full earlier version in existing Git history.
