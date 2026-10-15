# Decision: reuse Direct force moments and factor Hermite contractions

Status: implemented on the qualification branch; complete-endpoint acceptance pending
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
Memcheck and synccheck report zero errors. Complete Libcint/endpoint gates and
source-matched profiles are still required before promotion.

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
`/data/jzzeng/gqc-direct-reuse-20261016` on n4/n5. Initial benchmark preflights
reject an accuracy-only reconstruction used as a complete timing reference and
an accidentally selected portable build; neither returns a timed endpoint.
The remedy is the retained complete independent references and matching
Release/sm_120 official profiles, not bypassing either admission guard.

The final evidence must retain cold, five warm, moved, five moved-warm calls,
the original 1800-second population limit, actual SCF/Fock counts and independent
1e-8 Eh / 1e-7 Eh/Bohr gates. Diagnostic host and device ledgers are separate;
neither nested wall intervals nor device kernels are added to parent wall time.
