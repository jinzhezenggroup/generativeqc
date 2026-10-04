# Full TZVPD warm regression: integral work before further grid tuning

Status: diagnosis; optimization candidates require controlled qualification
Date: 2026-10-04

## Measured endpoint

The accepted [3/6/12-atom campaign](../../../benchmarks/results/wb97mv-tzvpd-cold-20261004/README.md)
uses one frozen native binary on n1 RTX 5090, complete spherical def2-TZVPD,
matched full semilocal/VV10 grids, and independent original/displaced energy
and force gates. Its native library SHA-256 is
`db20557ede9cd4471c0bf7659666f01431fda72eae2fc01434551fe90cc62b4c`.
All warm observations require one SCF iteration and accept the warm density.
Fock-build counts are unavailable in these public records and remain null.

The following component observations all belong to the **first warm call**
at each size, rather than combining a median endpoint with another call's
component timers. Units are seconds.

| Atoms / spherical AO | Complete warm | Force endpoint | Integral derivatives | Geometry/pair drain | Outside force timer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 / 58 | 1.901599 | 1.171211 | 0.864504 | 0.300824 | 0.730388 |
| 6 / 116 | 6.060485 | 3.441405 | 2.551732 | 0.880265 | 2.619080 |
| 12 / 232 | 62.843442 | 36.493232 | 33.160176 | 3.311353 | 26.350211 |

The last column is a subtraction, not a separately measured J/K timer.
The geometry/pair drain includes nonlocal pair work and must not be relabeled
as VV10 alone. Six to twelve atoms increases complete warm time about 10.4x
and integral derivatives about 13.0x. This identifies an integral hotspot;
it does not measure the distribution over angular classes or prove a global
scaling law. All three systems already contain f shells.

Five-repeat native/reference 12-atom warm medians are 62.961973/18.200958 s.
The native SCF AO discovery takes 0.136993 s in the first warm and its
point-AO-square domain is 8,101,125,376 versus 15,873,343,488 dense. These
are contraction-domain counts, not FLOPs or a measured speedup. Resident
active-AO mapping is already enabled, unlike the older public-path audit.

## Actual source routes

The inspected integral hot-path files are unchanged between production
commit `20543ad6f` used by the accepted points and `e60f60ecb` inspected here.

- `src/dft/cuda_ks.cpp` first requests fused RSH values, with separate
  full-J/K and LR-K enqueues on `NOT_IMPLEMENTED`.
- `src/scf/cuda/direct_jk.cpp` deliberately leaves through-f bounded value
  selection off. The admitted canonical Cartesian route already retains
  HF density/output projections, screened AO-pair rows and source metadata.
  `canonical_jk_kernel` in `direct_jk_kernels.cu` evaluates one contracted
  Cartesian ERI per admitted AO quartet. The compiler source owner is
  `python/generativeqc_compiler/integral/direct_source_contraction_cuda.py`.
- `src/scf/cuda_fock_execution.cpp` prefers the retained bounded shell RSH
  derivative owner. For omega 0.3, `direct_coulomb.cpp` executes full-range
  J/K derivatives and then LR derivatives, reconstructing separate SR/LR
  source channels. Full J/K already share their derivative recurrence.
- `direct_bounded_fallback.cu` consumes total orders 0--3 with weighted
  full/LR shell workers. `direct_force_quartet.cuh` shares explicit all-center
  full/LR gradients for orders 4--6. Higher orders still evaluate Dual3 for
  each of the first U-1 unique atoms, using translation invariance for the
  final atom. All-center reuse through six is already implemented.
- The full-range force launcher receives `p.bounded_block_domain`; the LR
  launcher does not. LR does reuse the sorted pair permutation and the
  density/exchange screening predicates, but still enumerates the complete
  block triangle without the retained prefix or 16-page claim subdivision.

## New warm capture

N1 Slurm 5724 completed the same-binary diagnostic after the exclusivity
repair described below. Its warm endpoint is 63.388783 s, with one iteration,
energy error 4.718e-12 Eh and force error 5.045e-11 Eh/Bohr. It is a profiled
observation on device 1, not another clean sample for the device-0 README curve.

| Executed kernel family | Device duration / s |
| --- | ---: |
| Canonical full J/K values | 11.078079 |
| Canonical LR K values | 12.385336 |
| Bounded full J/K derivatives | 16.734083 |
| Bounded LR derivatives | 16.529910 |
| All VV10 pair kernels, SCF and forces | 4.518127 |

Nsight identifies kernel names/order, not the radial argument. The value
trace exactly matches two repetitions of the source's 28 angular-pair buckets;
the force trace contains the two launches in the frozen full-then-LR order.
That source-order mapping assigns the first four rows. Their summed device
duration is 89.49% of this profiled endpoint. Do not add these new device
timers to the older clean host-wall component timers.

Canonical total orders 5--8 account for 17.840975 s, or 76.04% of the 23.463415 s
value-kernel duration. Order 12 alone is 0.007832 s. These are observed kernel
durations, not shell/primitive work counters or evidence about force-class
fractions. Both bounded force launches report 255 registers/thread and 256
threads/block. With the recorded 65,536 registers and 1,536 maximum threads/SM,
registers alone permit at most one such block/SM (a 1/6 theoretical thread
occupancy ceiling). This is not measured achieved occupancy or proof of spills.
Nsight's local-memory fields are inconsistent (zero per-thread and negative
signed total); no local-memory traffic estimate is taken from them.

The trace contains 14,422 kernel launches, including 56 canonical value and
two bounded derivative launches. Executed shell quartets, primitive products
and FLOPs remain unavailable. Machine-readable extraction and aggregate kernel
durations are retained as `diagnosis.json` and `kernel-aggregates.json` in the
diagnostic artifact directory. SQLite SHA-256 is recorded in `diagnosis.json`.

## Candidate order and validation

The capture establishes the integral-family priorities below. Force angular
class counts/timings still need a narrower census before selecting exact classes.

1. **Split derivative scheduling and reduce source repetition.** First
   count actual class work and isolate hot classes from the monolithic,
   register-heavy worker; a smaller per-class kernel can avoid paying the
   largest recurrence's resource footprint for every branch. Splitting alone
   preserves the physical integral count. Do not just cap registers and move
   the same storage into spills. For screened AO
   quartets q above total order six, the existing source work has the form
   `sum_q P_q (U_q - 1) C_dual3(L_q)`, where P counts primitive products and
   C represents recurrence cost, not a known FLOP count. A compiler-owned
   all-center contraction would instead use `sum_q P_q C_all_centers(L_q)`.
   Broader shell component reuse may additionally share primitive geometry
   across A_q Cartesian components. Neither expression proves a speedup:
   all-center register/local storage and occupancy must be measured. Apply
   only to classes shown hot, retain the Dual3 fallback, and independently
   qualify full/LR, repeated/distinct centers, both spins and displaced forces.
   The benefit can matter in f-containing or high-order d combinations; AO
   count alone is not an admission rule. Preserve bounded queues and measure
   all new per-class temporary storage rather than retaining all quartets.

2. **Improve canonical SCF source reuse without promoting the rejected
   bounded-value schedule.** Current full and LR value passes each visit
   admitted AO quartets, repeating primitive geometry and shell preparation
   across components. A shell source can produce its admitted components
   before eviction, sharing common geometry while keeping distinct full/LR
   radial moments. Expected work changes from per-component primitive setup
   toward one setup per shell/primitive product plus required component
   contractions; exact counts depend on screening and must be measured.
   Bounded scratch/queues and full/SR/LR source identity are mandatory.
   Start within the measured total-order 5--8 hotspot, select an exact class
   using actual work counts, and use an independent fixed-density matrix
   oracle, then complete cold/warm/moved endpoints and a larger size. The
   existing whole-route bounded toggle is not an acceptable shortcut.

3. **Pass the retained indexed domain to LR force scheduling.** With B
   shell-pair blocks and S geometry-admitted block products, replace the
   `B(B+1)/2` outer claims with `16 S` page claims and prefix lookup. This is
   a scheduling/work-distribution change, not a reduction of the surviving
   physical integrals, and can increase claim count in a dense domain.
   It requires no new retained index allocation when the full-range owner
   already has one; preserve the empty-domain bounded fallback. The exact
   predicates, orientation and coefficients must remain unchanged. Validate
   against independent CPU displaced range ERIs, indexed/triangular same-owner
   comparison, prefix-budget edges and sanitizer gates, then complete
   endpoints. At 12/96 atoms, dense block-triangle capacities are
   10,731/42,582,606; these are not measured surviving work or FLOPs.

## Cold consequence and rejected shortcuts

Twelve-atom prepared cold is 697.619542 s native and 181.337490 s reference,
despite 25 versus 45 target iterations. Warm includes a newly evaluated SCF
step and complete analytic forces, not a cached energy/force return. Costly
value sources recur across cold iterations; improving forces alone cannot
remove the entire cold gap. Per-iteration attribution still requires profiling.

The same-basis LDA source costs 262.989625 s (262.663043 s in its solve),
reduces target iterations 25 to 19, and worsens complete cold to 803.081423 s.
Do not promote it. Do not repeat VV10 variants with retained negative evidence
or duplicate Becke PR #1830, AO-density force PR #1798, or incremental KS
PR #1803 without resolving their own evidence and trajectory constraints.

The [through-f value policy](../implemented/performance/2026-10-02-through-f-value-policy.md)
retains an independently proven 10--18x bounded microbenchmark regression and
2.2--5x physical PBE0 energy-endpoint regression on an older controlled binary.
These are rejection evidence, not current WB97M-V calibration. HF Cartesian
reuse, LR low-order roots, LR all-center orders 4--6, and resident AO selection
are already present and must not be proposed as entirely new gains.

## Diagnostic and scheduling receipts

The completed n1 Slurm 5724 capture uses the exact accepted 12-atom native library,
builds its own converged cold density, and captures only the next complete
warm E/F endpoint with Nsight Systems. Independent E/F gates remain in force.
The ignored scripts and receipts live under
`.artifacts/tzvpd-warm-profile-20261004/`. Numerical checks and profiling pass;
this is attribution evidence, not a newly qualified optimization or speedup.

During its cold preparation, an orphan from cancelled n1 job 5722 was found
still computing on device 1, now assigned to 5724. Its command, PID and
`SLURM_JOB_ID` were checked before terminating only that task's Python/timeout
processes. The diagnostic was still in cold after both PIDs disappeared;
post-cleanup device receipts confirm its GPU had only the diagnostic process.
Its cold timing is contaminated and ineligible. Only a subsequent exclusive
warm capture can qualify for attribution. This is another physical board than
the accepted 12-atom campaign's device 0; do not claim cross-board speedups.

The orphan had also continued writing the shared reference48 output path.
Retain the detection snapshot and require the final paired48 reference and
native reports to identify job 5723/device 0, with every original/displaced
record, before accepting the point. Its live reference process retains its
own in-memory records and final write, but intermediate files are not evidence
of progress from the intended job. Future cancellation checks must verify
process exit as well as Slurm state before reusing a GPU or output directory.

The user prohibited further node3 jobs due to overheating. Its native24 job
12257 was already cancelled when checked, before any completed native row;
logs and the cancellation disposition are retained. Its completed reference
does not make a paired point. N1 Slurm 5725 restarts reference/native24 within
one allocation. Existing n1/n5 paired48/96 campaigns remain frozen and distinct
from this 12-atom diagnostic.
