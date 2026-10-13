# Decision: accelerate the bounded residual Fock domain without widening Rys admission

Status: implemented
Date: 2026-10-13

## Problem and decision

The [whole-shell reuse and f-Rys work](2026-10-13-bounded-rhf-shell-reuse.md)
removes repeated primitive preparation and supplies 22 f-containing value
recipes. Twelve other f classes still account for roughly 39% of two screened
C7 Focks. They exceed the qualified Rys root, fourth-center or component-tile
limits; changing a manifest is not sufficient to admit them safely.

Keep those limits and optimize the existing bounded whole-shell consumer:

- The first warp decodes/screens up to 32 candidates, then publishes a live
  mask. Rejected candidates do not enter recurrence barriers. Use
  `max(1, min(32, candidate_count / worker_count))` candidates per page so
  small domains retain enough pages to occupy the incumbent worker grid.
- Three lanes prepare independent Hermite axes from immutable cached geometry.
  The first warp raises independent states within descending Coulomb radial
  planes. Reuse the serial owner's Boys roots and axis operation; synchronize
  within the warp between planes and once with component readers afterward.
- Explicitly select the existing compiler-owned Hermite-axis convolution for
  order-five-and-higher FP64 components of this bounded owner. Summed bra/ket
  axis powers index the same Coulomb roots, replacing the six-index contraction
  with three axis convolutions and a three-index contraction. This is the
  [previously host-qualified reassociation](../../proposed/2026-10-04-hermite-axis-convolution.md),
  not another scientific implementation.

The last step is essential: reducing preparation synchronization alone did not
materially reduce complete reference time. This promotion is specific to the
bounded HF consumer, not the broader canonical/DFT value or derivative
experiment. Legacy packet and canonical whole-shell consumers keep their
existing serial preparation and contraction unless independently selected.

## Invariants and resource admission

Preserve exact full-range FP64 RHF/UHF, individual primitive coefficient
multiplication, shell/AO Schwarz and density gates, canonical pair orientation,
compensated exported-reference accumulation, final unscreened refinement and
physical-reference acceptance. Convolution changes floating-point summation
order; bitwise equivalence is neither required nor claimed. Every used
convolution weight is initialized. Partial tails, inactive lanes and empty
pages participate in the appropriate publication/retirement protocol.

No global numeric allocation, quartic tensor or changed resource budget is
introduced. Candidate storage is a fixed 32-entry shared page. Convolution
uses at most 15 FP64 local values. The existing 48-KiB shared-workspace bound,
pair-cache lifetime, generated/uncovered class masks and disabled/missing-cache
fallbacks remain authoritative. `GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=0`
still selects the old bounded consumer. Mixed precision and separately selected
reachable/convolved recurrence modes retain their existing owners.

Source-matched sm120 resource extraction is not a zero-spill claim. For the
order-twelve restricted kernel, REG/STACK/SHARED change from
`131 / 1648 / 21901` to `132 / 1776 / 22276`; the unrestricted variant matches.
Order eight changes from `131 / 256 / 11301` to `136 / 352 / 11676`.
The extra local traffic must be judged against complete endpoint results, not
hidden by reporting only the absence of new global allocations.

## Rejected preparation-only variants

All variants preserve the numerical gates; slower variants were not promoted.

1. A fixed 32-candidate page plus block-wide Coulomb plane barriers underfills
   small classes. In node4 Slurm 820, two Focks grow from 19.177886 to
   20.608983 GPU seconds; ffff alone grows from roughly 55 to 939 ms.
   Complete candidate C7 RHF takes 233.148246 s versus the 223.673319-s
   preceding resident-DF baseline. Library SHA-256:
   `6fcf3b93e4139d2df2bc65f89c33206bae402977441e70e625c7c1eec5b43164`.
2. Adaptive page width and warp-only Coulomb preparation repair that regression
   but leave contraction work unchanged. In node4 Slurm 821, the same-allocation
   two-Fock pair is 19.169721 / 19.260923 s (baseline/candidate). Complete RHF
   is 223.672883 / 223.199721 s, both 14 exact Focks. That roughly 0.2% endpoint
   difference does not establish a useful speedup. Library SHA-256:
   `d23fa7d526cf08b2a6bca963121afa6a7a0b042512fc7061ff971836f0ccbd47`.

The accepted candidate retains the adaptive/cooperative preparation and adds
the existing convolution. Do not repeat the fixed-page or preparation-only
experiments as if their negative evidence were unknown.

## Source, build and device identity

Parent source is `7497d35c1` plus this change. The three changed production
files match the deployed source byte-for-byte:

- Materialized emitter SHA-256:
  `ccca4974242549be1ae573373c7839a9f33ad4bb52fac1e17f138f32962726b9`.
- Shared Coulomb helper SHA-256:
  `879bc79f54fe6f9abbb88af40f627dc1be873dba91de930acebae36b5a8da2fd`.
- Native angular Fock owner SHA-256:
  `62720d54399d34dc7cb12a778178a5198da6faaee700db4827cd708104847918`.
- Frozen baseline library SHA-256:
  `6c5f76b76e0ab0587d2f2b39497dd127c3c1018578ee020df836fe2e59f30226`.
- Frozen candidate library SHA-256:
  `59b6e2094c99592498d3c60338c43a17210ddb9abbce2548ec0d3543ddf453c2`.

Build: Release CUDA 12.9.1, sm120/auto, eight AOT shards, fast compile off,
direct device link. Verified sccache 0.16.0 reuses the existing disk cache.
The accepted build uses a separate server port with a finite idle lifetime;
the earlier shared server exited during long CUDA compilations and its clients
fell back to local compilation. No shared server was stopped or cache cleared.
Accepted-build statistics record five hits, 28 misses, no cache errors/timeouts
and no compilation failures; launches alone are not counted as cache hits.

Timing uses finite node4 Slurm job 822, RTX 5090 UUID
`GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`, driver 580.178.04, 575 W.
The unchanged QM7 zero-based index 5604 fixture has 23 atoms, 58 electrons,
434 spherical/485 Cartesian cc-pVTZ AOs and 1047 correlation auxiliaries.
Input SHA-256 is
`696b783d661883745a71f15cb529e83802773d7ace6cbaf38f3e69f2e6cc16d0`.
Automatic DF preparation uses the same bundled 1464-function JK auxiliary.

## Semantic work and timing

The two-Hcore-iteration diagnostic is not a converged endpoint. Both libraries
execute exactly 109,665,513 admitted FP64 shell quartets across 55 classes;
all per-class counts match and no FP32 quartets execute. The twelve residual
f classes account for 2,728,802 admitted quartets in each run. Their GPU time
falls from 7.402387488 to 2.200539904 s; all-class time falls from
19.116786560 to 13.944170720 s. These counts are shell quartets, not primitive
or component counts. No tolerance was relaxed to produce the reduction.

| Class | Baseline two-Fock ms | Candidate ms |
| --- | ---: | ---: |
| fsdd | 863.572064 | 271.979968 |
| fpps | 781.399840 | 487.996640 |
| fpdd | 810.403584 | 237.817312 |
| fddp | 1481.478752 | 458.082016 |
| fddd | 840.004256 | 193.958464 |
| fdfp | 944.149472 | 192.537472 |
| fdfd | 470.572576 | 89.507872 |
| ffdp | 392.564928 | 115.572960 |
| ffdd | 232.663104 | 48.859968 |
| fffp | 257.866976 | 52.968288 |
| fffd | 272.921664 | 44.149312 |
| ffff | 54.790272 | 7.109632 |

Four fresh-process complete RHF endpoints run in candidate/baseline/baseline/
candidate order within the same allocation, including DF preparation and final
exact physical-reference publication:

| Run | Complete RHF s | DF preparation s | Exact target s |
| --- | ---: | ---: | ---: |
| Candidate 1 | 171.782120225 | 34.540314733 | 137.241805492 |
| Baseline 1 | 223.627221975 | 34.544281795 | 189.082940180 |
| Baseline 2 | 223.623230538 | 34.527944254 | 189.095286284 |
| Candidate 2 | 171.818198853 | 34.516646141 | 137.301552712 |

Medians are 223.625226257 / 171.800159539 s: 1.301659x, or 23.175% less
complete reference time. Each run has 21 preliminary DF cycles and 14 exact
Focks (eight screened plus six unscreened); there are no guess fallbacks,
execution retries, discarded primal/response attempts or extra post-SCF Focks.
DF physical-Fock counts remain unknown rather than inferred from cycle counts.
The speedup is not obtained by reducing iteration counts or changing tolerances.

Both candidate repeats independently pass energy/density/orbital-energy gates
of `2e-9 Eh / 3e-8 / 3e-8 Eh` against the retained PySCF 2.14 exact-RHF oracle.
Maximum errors are `3.97904e-13 Eh / 1.66102e-11 / 6.98932e-11 Eh`.
Commutator/canonical density drift are `8.37919e-12 / 4.13601e-12`, below
their `1e-8` gates. The oracle density is never provided to native execution.

The same allocation also completes a 144-AO ethane single-basis DF-CCSD(T)
E+F control pair: 65.785337588 / 65.772442685 s (baseline/candidate). Both use
13 DF cycles and 12 exact Focks, with no fallback/retry; energy is identical
and maximum force difference is `1.22125e-14 Eh/Bohr` (pair gates `2e-9 Eh`
and `2e-7 Eh/Bohr`). Lambda/Z/stationarity residuals are respectively
`5.34923e-13 / 2.99180e-13 / 2.31001e-13` in both runs. This is a complete
force non-regression control, not a meaningful small-system speedup, a CBS
calculation or an independent oracle for the ethane correlated forces.

## Independent qualification

Host tests: 1940 recurrence/dispatch/contracts tests pass with nine optional
skips; another 256 traversal, source-contraction and independent Gaussian
quadrature cases pass. These are not substitutes for device qualification.

Node5 Slurm 1523 passes the compiled native materialized test (221.71 s,
including cached compilation). Serial whole-shell tests remain intact; a
separate cooperative set covers all twelve f classes and orders 3--12, signed
contractions, RHF/UHF J/K, full/SR/LR, repeated pairs, tails, fully screened and
inactive work. Exact preparation/component counters match. Filtered racecheck
on shared publication/retirement reports zero hazards, errors or warnings;
filtered memcheck on the complete cooperative class set reports zero errors.

Node5 Slurm 1524 loads the identical frozen production library. Twenty-eight
screened/bounded RHF and DF-MP2/DF-CBS E+F tests pass (80.97 s), followed by
36 preconvergence tests (23.60 s) in a separate process. Independent PySCF
spdf E+F, both spins/representations, cold/warm/displaced geometry and
reconverged correlated-energy finite differences remain the scientific gates.
Filtered memcheck/racecheck on the first twelve actual bounded Fock launches
also report zero errors/hazards. These are filtered checks, not claims that
every launch of every production kernel was instrumented.

Compiler/SCF structure, default-promotion inventory, production-path ledger,
all configured hooks and Sphinx warnings-as-errors pass. No raw receipts or
binaries are added to tracked evidence.

Raw receipts remain ignored under `.artifacts/rhf-bounded/` and the node4
`/data/jzzeng/qc-cbs-rhf-20261013/receipts/rhf-fix/` directory, with prefix
`cooperative-v3`. `benchmarks/df_hf_preconvergence.cpp` supplies complete
reference/force endpoints; `tests/python/test_direct_pair_materialized_cuda.py`
and `tests/python/test_rhf_bounded_materialized.py` reproduce the focused gates.

## Limits and revisit conditions

This is not a complete C7 CCSD(T)/CBS E+F timing. The separate slow exact-response
operator exposed by the unfinished C7 force experiment is unchanged. Do not
substitute reference timing for total CBS time, or promote convolution to
canonical response/DFT/derivative consumers using this bounded-HF evidence.
Revisit those owners, expanded Rys capabilities or other GPU architectures only
with their own source-matched endpoint, resource, work and numerical gates.
