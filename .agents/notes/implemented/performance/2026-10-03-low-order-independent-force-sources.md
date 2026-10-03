# Decision: share geometry across independent low-order J/K force sources

Status: implemented; qualified incremental improvement, not performance parity
Date: 2026-10-03

## Problem

The full-range stationary force source contract publishes independent J and K
gradients. Its bounded native force scheduler traverses the same order-zero
through-three shell task twice, including AO enumeration, primitive metadata,
Gaussian geometry and Boys evaluation. Combining coefficients would violate the
source-output contract, even if their final weighted sum matched PBE0 forces.

## Candidate and invariants

`direct_force_low_order_sources.cuh` binds both channels to the existing
compiler-owned weighted force roots. It shares only immutable geometry and AO
traversal. It retains independent density weights, generated force-root calls,
primitive reduction order, screening predicates and zero-channel behavior.
All nine closed low-order shell classes are supported. Scalar combined-force,
resident-bra, generic high-order and range-separated consumers remain unchanged.

The PSSS worker's unique-atom order is **raw shell order**, unlike the other
canonical workers. Retaining it is necessary to preserve which atom receives
the translation-restored final contribution. Generated stable canonicalization
retains equal-angular ties; no new integral recurrence is introduced.

There are no new device allocations, resident-state assumptions, resource
budgets, precision changes or approximation thresholds. Extra channel-local
weights and gradients can nevertheless increase register/stack pressure, so
reduced source work alone does not establish an endpoint improvement.

## Evidence

Base is freshly fetched master `9e9b938239d31564d6f92e9b99b922aee0dbb1cc`.
The prior e1-based moment-cache candidate is separate and is not included.

- Host execution of the real generated geometry/force algebra and old/new native
  schedulers compares **13,026,816 force components bitwise equal**. Cases include
  every ordered low-order angular tuple, all shell pair-of-pairs, repeated centers,
  nonsymmetric/zero/sparse/opposing-spin densities, coefficient masks, AO Schwarz
  rejection, inactive systems and nonzero tile rejection.
- Test-only counters observe **1,311,456 -> 1,043,280 geometry calls**, while force
  root calls remain **1,311,456**. Every task with two live sources uses 16 instead
  of 32 geometry evaluations; one/zero live sources retain 16/0. These are weighted
  **fixture counts**, not actual molecular benchmark work or a speedup.
- The same host harness passes AddressSanitizer and UndefinedBehaviorSanitizer.
- The independent native CPU-ERI qualification is extended to the real public
  shell `[J', K']` API, all coefficient masks, both spins, original/moved geometry
  and Cartesian/spherical representations through f. Node1 Slurm job5490 passes
  this expanded oracle for both baseline and candidate; candidate memcheck and
  initcheck also pass with zero errors.
- Certified baseline/candidate native identities are respectively
  `a80427eb5263cbcacacd36eb52f2768a4d6b114ef93961fa2dc48a6b70016cc0` and
  `14d47b97f7de8e34bea8990fd9a46e8d2b27fb2a165ac053b1f2ddb0b489480b`.
  Device symbols verify that the changed CUDA owner really compiled. Both spin
  force-screened kernels retain 255 registers and 90392-byte static stack.
- Latest-master baseline node5 job1405 passes all eight 24/96-atom diagnostic
  endpoints; clean warm medians are 5.200943/80.209254 seconds.
- Node5 job1406 completes baseline96/candidate24/candidate96/baseline96 on one
  GPU allocation. The four baseline96 clean warm calls take
  80.164096/80.153800/80.187680/80.181607 seconds; candidate96 takes
  77.709112/77.701921 seconds. All six take one SCF iteration. Pooled control
  median 80.172851 versus candidate 77.705516 is **3.077519% less wall time**,
  or 1.031752x. All 16 diagnostic calls pass the independent energy/force gates.
  Profiled calls are excluded from medians. Cold iteration/cache histories
  differ and do not establish a cold speedup.
- Node1 job5491 completes all **72 public endpoints**: cold, five frozen warm,
  moved and five moved-warm calls at 3/6/12/24/48/96 atoms. The retained independent
  reference is acceptance-only; no reference densities seed production.
  Every full force array is revalidated, not merely its stored gate flag.
  Maximum energy/force errors are 1.036824e-10 Eh / 3.036026e-11 Eh/Bohr, below
  the unchanged 1e-8 / 1e-7 gates. At 96 atoms, original/moved warm medians are
  79.803158/79.397174 seconds. Do not mix these cross-node measurements into the
  node5 performance ratio.
- The 96-atom grid work remains 2,359,296 points, 9,216 geometry batches and
  21,516,779,520 Becke pair-state evaluations; additional device/host bounds
  remain 357,022,976/197,047,712 bytes. Executed molecular low-order primitive
  counts are not measured; the host fixture counts above must not substitute
  for them. Both output channels still execute their own generated force roots.
- 91 focused host cases, the native host ASan/UBSan control, 147 ownership/
  dependency cases and 116 benchmark/retention cases pass. The new adapter stays
  classified as scientific composition in both ownership inventories; this
  optimization is not retirement of native scientific responsibilities.

## Reproduction and limits

The isolated checkout is `/home/jzzeng/codes/qc-pbe0-low-order-sources-20261003`.
Native builds run on n5 because local storage is critically constrained. Tools
and the existing compiler cache were copied, with a dedicated remote cache;
compiler identities, explicit CMake launchers, checkout-root `CCACHE_BASEDIR`,
statistics and source/binary identities are retained under `.artifacts/`.

The compact [qualification record](../../../../benchmarks/results/pbe0-low-order-sources-20261003/README.md)
binds raw-file hashes, build identity, semantic work, all repeats and reproduction
commands. Raw arrays, build/cache receipts and profiler files remain in ignored
local artifacts and the existing node5 mirror; no new external host or release
is used. A rejected stale incremental build and failed node4 driver probes
remain separate from successful measurements.

Subsequently fetched master `cd08953d5765d389baba62935f2025816be57075` is newer
than the measured base. Its changes are not relabeled as measured here. The
original user worktree and withheld README DFT chart remain unchanged.

This slice does not solve the approximately eightfold large-system PBE0 gap.
Integral derivative roots, XC geometry and native SCF still dominate. No
range/high-order behavior, numerical approximation, tile policy or resource
allowance is changed. Revisit the remaining source work rather than extending
the small geometry saving into an unsupported endpoint-parity claim.
