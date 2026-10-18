# MD-J contraction and reciprocal-reuse trials

These follow-up experiments use the accepted 96-atom/1088-AO density and
scientific settings identified in [the diagnostic record](README.md). All real
GPU work runs through finite Slurm allocations on n1/RTX 5090, preserving the
assigned device visibility. CUDA 12.9.1/GNU 13.3 Release sm_120 builds reuse
verified ccache 4.5.1. No FP32 conversion, screening relaxation, CPU production
oracle, driver change or Nsight Compute hardware-counter claim is involved.

## Clean prepared-resident J

Each trial uses one same-GPU allocation. Fresh processes run in the order
GPU4PySCF, original, experiment-on, experiment-off, experiment-on, original,
experiment-off. Each native process performs one untimed warm-up and four
uncounted/unprofiled replays. The table reports the median of all eight replays
per native mode, not a full SCF/force speedup. On/off comparisons use the same
trial binary; the original binary is a separate compatibility control.

| Trial | Slurm | Off, seconds | On, seconds | On/off change |
| --- | ---: | ---: | ---: | ---: |
| Bounded register accumulation + fixed Hermite-loop expansion | 10114 | 4.128958 | 4.121597 | -0.18% |
| Above + task-local density contraction before J atomic publication | 10115 | 4.134439 | 4.248589 | +2.76% |
| Angular-three uniform reciprocal source reuse | 10117 | 4.121193 | 3.923415 | -4.80% |

The first trial is effectively neutral at this sampling depth; the second is
slower. Their production changes and experimental selector are removed. The
retained third trial is opt-in with `GENERATIVEQC_MD_J_RECIPROCAL_HIGH=1`;
unset/0 keeps the prior oriented provider for these classes. It extends the
existing compiler-owned reciprocal consumer to `(0,3)`, `(1,3)`, `(2,3)` and
`(3,3)`, sharing a radial source across two independently admitted density
directions. Angular-four work, residual masks, normal K and resource fallbacks
are unchanged. There is no additional resident allocation or recurrence owner.

The common original library SHA-256 is
`5ca1225535852f611b5814d2872455b7c1bfbe8be31543b25e8be76069d8c314`.
Trial library SHA-256 values, in table order:

```text
46afcd39eea1e0afd0f96b93fe7464433581c2247a49a0ea0b742814a0a4f4aa
9308c24c6d69835cda0e5ecc4862dcc657040720b429813abb2a921a7e1ff6b5
57c052abef324edb4e7f630616581a9f10ea3b6afcefcdcfdb202dac19db8552
```

The source base is `b469940ac6ad9372b0d9bc3901c6bca3c363d906`; exact experimental
patches and build/cache receipts are retained in the ignored evidence bundle.
The subsequent `6d99889ed` only completes two unrelated host-fixture CUDA event
types; it does not change the measured production sources.

GPU4PySCF 1.8.1 medians are 0.712617/0.714770/0.712708 seconds in these three
allocations. Every native replay passes the same independent raw-J gates
(`atol=3e-9, rtol=3e-11`), with maximum absolute error about `2.445e-11`.
Native aggregate-error screening and GPU4PySCF's per-interaction cutoff remain
different contracts; the approximately 5.5x remaining prepared-J gap is not an
attribution of that gap to a single mechanism.

## Separate profiling and semantic work

Slurm 10116 profiles original/register/reduced-publication binaries separately
from clean timing. Residual kernel sums are respectively 2.473238, 2.461300 and
2.571298 seconds. Task-local atomic aggregation therefore does not improve the
measured residual endpoint. This does not prove atomics are free or identify a
hardware bottleneck; lower active-lane parallelism and changed instructions are
confounded with the reduction.

Slurm 10118 profiles the original and angular-three experiment and collects a
separate instrumented work census for all four binaries:

| Kernel family | Original seconds | Angular-three reuse seconds |
| --- | ---: | ---: |
| Residual selection and contraction | 2.481107 | 2.455768 |
| Uniform reciprocal | 1.019238 | 1.267694 |
| Uniform oriented | 0.645140 | 0.190210 |

The reciprocal increase includes the newly transferred angular-three work.
Uniform contraction in aggregate falls from 1.664378 to 1.457903 seconds;
do not assign the small residual timing variation to this uniform-only change.

| Executed work | Original | Angular-three reuse |
| --- | ---: | ---: |
| Uniform radial evaluations | 2,225,962,468 | 2,038,308,216 |
| Uniform density directions | 4,013,012,720 | 4,013,012,720 |
| Uniform Hermite summands | 95,323,936,432 | 95,323,936,432 |
| Residual admitted tasks | 43,920,982 | 43,920,982 |
| Residual primitive products | 242,399,046 | 242,399,046 |

All four censuses preserve per-class density directions/summands, residual
admission/products and the 526,842,030-candidate domain. Compaction order can
change the individual-test/tile-rejection partition across fresh owners; that
partition is not a semantic invariant. The rejected residual trials do not
reduce radial evaluations or integral work.

## Scientific and safety gates

- Both residual experiments pass 80 independent resident-GPU tests each.
- The retained angular-three experiment passes 78 resident-GPU tests, including
  nonsymmetric RKS/UKS densities, Cartesian/spherical AOs, zero/nonzero AO
  screening, same-pair symmetry, K preservation, bounded fallback, actual work
  conservation, parent-selector fallback and frozen replay policy.
- Slurm 10116 and 10118 each pass isolated memcheck, racecheck and synccheck;
  no errors or race warnings/hazards are reported. These small cases are not a
  complete large-system sanitizer qualification.
- Final focused host validation passes 267 tests, with 78 real-GPU cases skipped
  outside their explicit scheduled run. Compiler, SCF and CUDA ownership checks
  pass. The old angular-limit source-string assertion is replaced by executable
  independent numerical coverage rather than another source fingerprint.

## Complete clean energy/force endpoints

Slurm 10122 runs off then on in fresh processes on one assigned GPU, using the
same final library and source-matched FP64 force companion. All intrusive
counting/profiling selectors are unset. The independent reference and protocol
are the existing repeats-one projection described in the diagnostic record.
Both arms finish and pass every independent energy/force gate.

| Phase | Off complete seconds | On complete seconds | Fock builds, off/on |
| --- | ---: | ---: | ---: |
| Cold | 270.418484 | 264.922447 | 24 / 24 |
| Warm | 57.457074 | 57.194450 | 1 / 1 |
| Changed geometry | 170.816579 | 167.888167 | 13 / 13 |
| Changed-geometry warm | 58.318112 | 57.382561 | 1 / 1 |
| Sum | 557.010250 | 547.387625 | 39 / 39 |

These use the driver's `complete_seconds`, including preparation in the cold
record. Cold evaluation-only `seconds` are 261.559972/256.273540. Complete cold
improves 2.03%, and the four-phase sum improves 1.73% in this one pair. Warm
differences of this size are not repeat-qualified speedups. No Fock work is
removed to obtain the improvement.

Maximum independent energy error is `2.456e-11` Eh, and maximum force error is
`6.353e-11` Eh/bohr across both arms. The maximum off/on energy difference is
`1.364e-12` Eh. This is encouraging endpoint evidence, not a broad or repeated
default-promotion qualification. The new selector remains off by default and
PR #2600 remains draft; earlier residual-default qualification limits still
apply independently of this uniform-only optimization.

## Retention

Compact raw receipts, exact source patches, build/cache commands, clean timing
records, work censuses and sanitizer logs are retained in ignored local
`.artifacts/mdj-remote/contraction-trials-20261018.tar.xz`:

```text
bytes: 118968
members: 192
sha256: 721d7f30d65a6224070587be4d676b235349c64e0ec3ae0cdd4b12ebc122181d
```

Every retained member hash is verified against the original bytes. The sibling
`contraction-trials-manifest.json` lists member checksums. Full matrices,
timelines and binary snapshots remain in the ignored original local/remote
task directories; they are not added to Git or replaced by this compact
archive. No Release, release asset or new external storage host is created.

## Subsequent default decision

Following these measurements, the user explicitly authorized default-on
angular-three reciprocal reuse. The
[superseding decision](../../../../.agents/notes/implemented/performance/2026-10-18-md-j-high-reciprocal-default.md)
sets unset/1 to enabled and retains `GENERATIVEQC_MD_J_RECIPROCAL_HIGH=0` as
the oriented fallback. The historical tables above measure the original opt-in
binary, not a relabeled default build. Separate default-path validation retains
its own source and binary receipts. This promotion does not imply statistically
established cross-device speedups or resolve the residual-tile qualification.
The [default-path record](default-reuse.md) reports that separate pair and its
114 scheduled independent GPU tests.
