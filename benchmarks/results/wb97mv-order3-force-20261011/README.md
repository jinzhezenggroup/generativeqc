# WB97M-V order-three and LR-force scheduling qualification

The frozen base is master `637466c624c4c4c4dd4eb7c511a2c88675744325`
plus submitted PR #2236 (`66c4e617`). It is neither bare master nor the
earlier `f6b9220` performance arm. The candidate automatically parallelizes
the order-three Hermite axes and selects 128-thread ordinary LR force CTAs.
No opt-in, molecule/basis/functional whitelist, precision change, or screening
approximation is introduced.

## Protocol and scope

- Water proxies, not OMol25-distribution samples; complete spherical def2-TZVPD
  H/O with diffuse and f shells retained.
- Exact Direct WB97M-V/RKS FP64; 48-by-16-by-32 grid per atom, Becke3;
  energy/density/screening tolerances 1e-12/1e-10/1e-12, maximum 100 iterations;
  VV10 density cutoff 1e-8 and native host/device force budgets of 4 GiB.
- Fresh owner and independently initialized density for every cold sample.
  Imports, CUDA context, and Calculator construction precede the complete clock;
  prepare, SCF, physical forces, synchronization, and host publication are inside.
  Teardown and serialization follow it. Compiler/artifact caches are preserved;
  separately retained first-use priming samples are not throughput arms.
- Matched 3- and 12-atom complete endpoints use ABBA, two samples per arm,
  without profiler/interposer instrumentation. Independent frozen full-Fock
  GPU4PySCF receipts gate physical E/F; their densities are never borrowed.
- The 96-atom result is only the four full/LR order-three source windows,
  omitting every other source. Complete 96-atom cold E+F and its independent
  physical-force acceptance remain explicitly absent/null.

## Results and qualification

`evidence.json.gz` retains source/library identities, full protocol and physical
samples, matched medians, independent numerical errors, semantic work/traffic,
raw hashes, source-only profiles, rejected force variants, and the per-class
native force census. The accepted Release library SHA-256 is
`8286424dbc7863a5af99ee7e64377057ecf74cef73f185d27999ad75e76fee62`.

Matched complete endpoint medians (seconds; two ABBA samples per arm):

| Atoms | Stage | Frozen baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| 3 | Complete prepare + SCF + E/F publication | 12.118045 | 11.977786 | -1.16% |
| 3 | Physical forces | 3.423379 | 3.332032 | -2.67% |
| 12 | Complete prepare + SCF + E/F publication | 194.953520 | 190.796692 | -2.13% |
| 12 | Physical forces | 36.039981 | 33.355487 | -7.45% |

These are observed paired results, not statistical significance claims. Each
3-atom arm performs 15 iterations/Focks; each 12-atom arm performs 21. The
maximum independent physical energy/force errors across all matched arms are
8.356e-12 / 5.438e-11. Recorded SCF AO work, force capacity/observed work, and
force traffic match exactly across corresponding arms. Empty generated,
screened, compacted, and executed force-counter maps remain empty: the receipt
does not substitute capacity bounds for missing primitive execution counts.

The Release 96-atom order-three window changes from 177.410883 to
150.254918 seconds, a 15.31% reduction. All six requested raw J/K channels
match within 5.862e-13. This is not a complete SCF or force speedup.

Gates include 81 host tests, 513 compiler modules with zero dependency errors,
four native CUDA suites, separate radial and mixed s/p/d/f derivative/orbit
gates, eight independent Libcint cases, complete low/high-order primitive-work
gates, and four axis-consumer sanitizers with zero errors/hazards.

The intrusive fixed-state native shell census matches exactly per class across
launch widths. LR admits 7,497,817 shell quartets. Its 335,167,456 AO and
767,930,390 primitive quantities are capacities inside admitted shells, not
post-component-screening executed work. Complete endpoint SCF/force work and
traffic comparisons are separate from that diagnostic.

The force gap is narrowed, not closed: the candidate's 12-atom force median is
33.355487 seconds. The independent frozen reference force clock is 11.140979
seconds (complete E/F 182.512815 seconds, 45 iterations / 46 Focks); it is not a
freshly paired reference throughput arm
or a stock-default GPU4PySCF comparison. Rejected blanket angular splitting,
f-only angular splitting, 64-thread force widths, and initial compiler-cache
misses remain distinguishable in the receipt.

## Retention and reproduction

Raw inputs, logs, native matrices, Nsight report/SQLite, compiler commands/cache
statistics, gates, and source hashes remain on node1 under
`/data/jzzeng/wb97m-order3-force-20261011/`, with an ignored local mirror under
`.artifacts/canonical-order3-force-20261011/`. The receipt binds the retained
raw archive by checksum and size. No Release, release tag, or external raw
backup is published.

The archive and per-result hash inventory exclude the compact receipt, matched
summary, and receipt-builder stdout. These are outputs of receipt construction;
including them would create checksum cycles or bind an incomplete mutable log.

Input scripts freeze master plus #2236, apply the candidate patch, and build
Release sm_120 with verified ccache and checkout-root normalization. Run every
real-GPU gate/profile/benchmark through a finite Slurm job on an allowed node,
preserving its assigned device visibility. `complete-abba.sh` reproduces the
complete timing arms; `profile-release-96.sh` reproduces only the named source
windows. `receipt.py` rejects mismatched numerical, work, or library identities.

The [Agent Note](../../../.agents/notes/implemented/performance/2026-10-11-order-three-axis-and-lr-force-scheduling.md)
preserves the diagnosis, immutable-axis publication invariant, unsuccessful
alternatives, and the remaining shared-derivative-recurrence direction.
