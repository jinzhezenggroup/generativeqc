# Decision: preserve whole-shell recurrence reuse in bounded GPU RHF

Status: implemented
Date: 2026-10-13

## Problem

The 434-spherical-AO QM7 C7H16/cc-pVTZ reference preparation spent essentially
all its GPU time in the generic bounded Fock consumer. The original diagnostic
completed 14 Fock kernels totaling 1312.588 GPU seconds before a 25-minute Slurm
timeout; it did **not** finish RHF. SCF convergence was not established by that
trace and MP2 never ran.

Two routing facts explain the work rather than an eigensolver/DIIS bottleneck:

- This particular `portable_cuda` build has zero generated Fock-class coverage.
  Do not describe its measurements as tuned-AOT performance.
- The optional canonical ERI source requires 55,559,675,520 bytes for this
  geometry, exceeding its 8-GiB allowance. Refusing it is correct. However, the
  bounded fallback then prepares Hermite/Boys/Coulomb recurrence separately for
  Cartesian AO components. Whole-shell reuse already exists in the compiler
  and in fixed-topology/independent J/K consumers, but was not connected here.

The bounded route is required for large reference topology. Removing its
resource guard or just increasing the quartic cache limit does not fix it.

## Decision

For uncovered full-range FP64 classes of total angular order 3--12, stream the
existing class-major shell-pair topology directly into the compiler-owned
`contract_materialized_direct_pair_fock` consumer. One CTA owns a whole shell
quartet, sharing preparation across every live component. A 64-bit grid-stride
domain avoids a topology-sized task arena and a 32-bit persistent-counter
overflow. The existing component-lane/slot bounds cover *all* s/p/d/f
compositions, including fddd and ffff tails.

Generated classes keep their current owners. Each newly consumed class is
excluded from the generic fallback, which still handles remaining classes.
The existing `GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=0` control restores
that fallback for A/B qualification. Missing primitive-pair storage, mixed
precision and incompatible recurrence modes also retain the original path.

## Invariants

- No change to exact RHF versus DF-RHF, screening tolerances, convergence
  tolerances, public basis representation, primitive coefficient order or force
  equations. No production CPU/oracle or reference-density seed is introduced.
- Borrow the existing geometry-bound primitive cache; no new global numeric
  allocation, four-index tensor or extra admission budget. Shared recurrence
  remains bounded by the compiler's existing 48-KiB static limit.
- Preserve canonical pair orientation, per-shell density/Schwarz screening,
  component-level screening, RHF/UHF scatter, and compensated reference output.
- Inactive and screened lanes participate in shared publication/retirement.
  Warm geometry updates use the existing cache rebuild and graph ownership.
- Class timers count accepted shell quartets, not primitive recurrence work or
  AO-component counts. Complete endpoint counters remain SCF/Fock actions.

The existing optional progress journal also records completed split-provider
iterations at their already-required stream fence. It adds no GPU transfer or
synchronization. Failed submission/solver/fence observations are not completed
iterations; the callback's iteration index is already one based. Fully
device-tail iteration retains its existing final completion counters.

## Qualification

The input is QM7 index 5604 (zero based), not an arbitrary molecule sharing
the formula C7H16. Promote the dataset's float32 Bohr coordinates unchanged to
float64. There are 23 atoms, 58 electrons and 434 spherical/485 Cartesian AOs.

- Dataset SHA256:
  `fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
- Geometry/basis input SHA256:
  `696b783d661883745a71f15cb529e83802773d7ace6cbaf38f3e69f2e6cc16d0`.
- Baseline source: `c67a913dfeb7297725aad3dc42fc4c727da71d92`.
- Baseline library SHA256:
  `36966c4d7338b1b7e281dc239aa7d0e6281fe108683577adf6c369dba2d7b0c9`.
- Independent PySCF 2.14.0 conventional unscreened RHF energy:
  `-274.496819172566 Eh`; its retained density and orbital spectrum are oracles,
  not native inputs.

`tests/python/test_rhf_bounded_materialized.py` checks full E+F against PySCF
with spdf shells and signed contractions, RHF/UHF, Cartesian/spherical bases, two
screening tolerances, both scheduling choices, batched warm replay and a changed
geometry. The public HF API requires positive screening; the C7 physical
reference qualification separately exercises exactly zero screening.

The clean baseline completed on node4/RTX 5090 in Slurm job 805: 2719.5223 s
including native setup and physical-reference publication, 29 SCF iterations,
29 exact Fock builds, zero extra final Fock builds and no execution retries.
The automatic provisional DF guess was refused by its independent 512-MiB
budget, so no DF iterations or externally supplied density entered this run.
Absolute errors versus the independent oracle were 5.69e-14 Eh in energy,
1.74e-11 in density and 6.88e-11 Eh in the orbital spectrum.

The portable candidate completes on the same physical RTX 5090 in job 807:
1348.6332 s, again 29 iterations/29 exact Focks, no extra final Fock and no
execution retry. Energy/density/orbital-energy errors are 5.12e-13 Eh,
2.69e-11 and 5.65e-11 Eh. The 2.0165x endpoint improvement removes redundant
generic preparation, but approximately 46.5 s per iteration remains too slow;
this is not evidence that tuned production RHF is optimized.

On node5, two interleaved ethane/cc-pVTZ (144 spherical AOs) controls take
4.00471/3.88644 s before and 3.89636/3.89780 s after. Each retains 13 native
provisional DF iterations and 12 exact Focks; the natural resident-source
route is unchanged. All four results have the same independent maximum
energy/density/orbital-spectrum errors, 6.97e-13 Eh/1.44e-11/4.18e-11 Eh.
These two samples establish no meaningful small-case speedup.

Portable numerical qualification passes 16 independent E+F/DF-CBS tests on
node5 in job 1498, including the new bounded RHF/UHF cases and the existing
DF-MP2/DF-CBS force/reuse gates. Filtered new-kernel memcheck passes with zero
errors (node2 job 2941). The full E+F racecheck reaches its 20-minute limit and
is not a pass; a focused two-iteration spdf probe passes racecheck with zero
hazards (node2 job 2943). That probe is not a converged endpoint.

Acceptance gates for the same C7 input are 2e-9 Eh in energy, 3e-8 in maximum
density error and 3e-8 Eh in maximum orbital-energy error, plus the native
physical/canonical residual checks and valid complete Fock-action counters.
The small public E+F gates are 2e-9 Eh and 2e-8 Eh/Bohr, respectively.

The pinned input can be reproduced without a reference SCF:

```sh
PYTHONPATH=python:. python -m benchmarks.df_hf_preconvergence prepare-case \
  --case qm7-c7h16-tz --qm7 /path/to/qm7.mat \
  --output .artifacts/rhf-bounded/c7-tz.input
```

Compile `benchmarks/df_hf_preconvergence.cpp` against the qualified CUDA library
and run `df-hf-preconvergence INPUT - OUTPUT auto-direct hf` inside a finite
`srun --partition=main --gres=gpu:5090:1` allocation. In this mode the unused JK
file argument supplies no data. Keep `GENERATIVEQC_PROFILE=off`, native FP64,
zero screening, E/density tolerances 1e-12/1e-11 and the exported 64-GiB numeric
budget unchanged. The input exporter reproduced the measured bytes exactly.

Raw profiles, receipts and binaries stay in ignored `.artifacts/` and the
qualification workspace rather than increasing the tracked evidence budget.
All device execution, including profilers and sanitizers, uses finite Slurm
allocations. Complete endpoint timing must be distinguished from the bounded
two-iteration, nonconverged profiling probe.

## Tuned-profile follow-up: value-only f-shell Rys

The stock sm120 manifest covers 21 s/p/d Fock classes but no f-containing Fock
class (`fpps` is force-only). With whole-shell reuse enabled, uncovered f classes
still dominate the tuned C7 trace. Extend the existing value-only component-lane
Rys decoder, not its force decoder: f Cartesian axis counts reuse the compiler's
existing axis table, and unraised second-center HRR already supports cubic
order. No new quadrature tables or recurrence equations are introduced.

Promote 22 value-only Fock recipes in the sm120 manifest: `fsss fsps fspp fsds
fsdp fsfs fpss fppp fpds fpdp fpfs fpfp fdss fdps fdpp fdds fdfs ffss ffps ffpp
ffds fffs`. Keep the existing `fpps` force owner unchanged. Eligibility retains
the 2--5-root limit, at most p on the fourth center, and one complete Cartesian
component tile per block (at most 1024 lanes). Thus `fddp` (1080 components),
`fsdd`/`dddd` (fourth-center d), and higher-root combinations still use their
previous generated or bounded owners. The change is target-profile-specific;
other architectures retain portable dispatch rather than inheriting unmeasured
sm120 schedules. Existing force derivative admission is not widened.

### Same-device complete endpoints

All C7 runs below use node4's RTX 5090
`GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`, driver 580.178.04, 575-W limit, the
same pinned input, strict tolerances, FP64 and exactly zero screening. Times
include setup through native physical-reference publication, not only kernels.

| Build / route | Slurm job | Complete RHF seconds | Iterations / exact Focks |
| --- | --- | ---: | ---: |
| Portable clean base | 805 | 2719.522267755 | 29 / 29 |
| Portable whole-shell reuse | 807 | 1348.633152803 | 29 / 29 |
| Stock tuned, old fallback ablation | 810 | 1579.959116925 | 29 / 29 |
| Stock tuned, whole-shell reuse | 809 | 859.523261330 | 29 / 29 |
| Tuned, whole-shell reuse + f-value Rys | 811 | 552.299426638 | 29 / 29 |

The tuned old-fallback control is the same stock-tuned candidate binary with
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=0`, not a separately rebuilt base.
The final endpoint is 2.8607x faster than that tuned rollback control, 1.5563x
faster than stock tuned whole-shell reuse, and 2.4419x faster than the earlier
46.5-s/iteration portable result. It averages 19.04 s/iteration including setup.
These are single complete C7 runs, not medians or a claim of optimal RHF speed.

All five runs retain 29 exact Fock actions, zero extra final Focks, no discarded
attempts/retries, no provisional DF cycles and no oracle seed. The unchanged
unscreened topology contains 96,070,591 shell quartets per exact Fock; the
two-iteration profiling census contains 192,141,182. Timers on generated primary
pages and uncovered classes now both report their actual work, rather than
mistaking empty overflow pages for zero generated work.

The final C7 errors versus the independent oracle are 5.68e-14 Eh (energy),
2.41e-11 (maximum density), and 7.69e-11 Eh (orbital energies). The commutator,
canonical-density drift, eigen residual and orthogonality error are respectively
6.02e-12, 8.33e-12, 4.09e-13 and 2.07e-13. All stated acceptance gates pass.
The tuned rollback control also passes: E/D/orbital errors 1.71e-13 Eh,
1.65e-11 and 6.89e-11 Eh.

Node4 job 812 checks the natural resident-source ethane/cc-pVTZ path in ABBA
order (stock tuned / f-Rys / f-Rys / stock tuned): 3.858450399 / 3.832204289 /
3.843332765 / 3.823447398 s. All retain 13 provisional DF iterations and 12 exact
iterations/Focks, with identical oracle errors E/D/orbital = 6.96e-13 Eh /
1.43e-11 / 4.17e-11 Eh. There is no meaningful small-case speedup or regression.

### Qualification and provenance

- Node5 job 1504 passes all 3168 independent Libcint fixed-density GPU cases:
  all 22 new classes, Cartesian/coincident/reversed pairs, RHF/UHF, J/K/HF-K/
  combined consumers, task tails 1/33/129, and ordinary/compensated reductions.
- The same job passes 24 complete E+F/post-HF tests: 16 automatic/bounded
  RHF/UHF cases with batched/warm/moved geometry, plus eight existing native
  DF-MP2/DF-CBS correctness and reuse gates. No force equations change.
- Node5 job 1506 passes filtered memcheck on all 22 new RHF kernels with
  compensated 129-task tails (zero errors), and focused `fffs` racecheck
  (zero hazards). This is not a claim of exhaustive sanitizer coverage.
- Stock tuned whole-shell reuse separately passes eight RHF/UHF E+F tests
  (job 1501) and eight DF-MP2/DF-CBS tests (job 1502). The DF cases were skipped
  in job 1501 because its opt-in flag was absent; only job 1502 is their pass.
- Stock tuned library SHA256:
  `c45693c96560586a74ab1e5dc96c7841bf75291fd2f41a873f6e2b5d77ace092`.
- Final f-Rys library SHA256:
  `44dce75cdd99d6c3df25c430fc33ce597609eb1f49c001a933cae47e8601d3a7`.
- The measured experimental manifest SHA256 is
  `88566329b10dc90324f96ae0c702a4da5fe3a46feae399a29588499ffd105698`.
  The checked-in promotion has identical parsed fields and ordering. All ten
  generated registry/shard files from the canonical manifest are byte-identical
  to the files compiled into the measured library; formatting/path differences
  are not substituted for source-matched qualification.

Builds use CUDA 12.9.1, Release sm120, eight stable AOT shards, fast-compile off,
separable compilation off, direct device link on, and verified sccache 0.16.0
launchers. One earlier shared-cache server exit required local compiler fallback;
do not label all compilation as cache hits. Frozen qualified libraries and raw
receipts remain in ignored local/remote qualification storage.

This measurement ends at the C7 RHF reference, **not** full C7 CBS E+F. It does
not convert the reference to DF-RHF or claim that large-basis C7 CCSD(T) is now
feasible. Remaining expensive classes include `fddp dddd fddd fdfp fsdd fpdd
fpps`; improving those requires separate bounded schedules/endpoint evidence,
not relaxing precision, screening, physical checks, or the memory guard.

## Subsequent reference qualification

The historical timings above precede the materialized-reference compensation
repair (`dcc773b35`) and value-only Rys registry-link repair (`89cda7cbf`). The
first preserves the compensated output plane through the whole-shell
contractor; executable host cancellation/traversal tests protect it. The second
does not advertise nonexistent force entrypoints for value-only Rys recipes;
strict-link tests cover both registry emitters. Lazy loading had hidden those
undefined symbols in the earlier Python qualification.

The [screened-reference iteration note](2026-10-13-screened-rhf-reference-iterations.md)
records subsequent source-matched tests and endpoints with both repairs. It
also supersedes the all-iterations-zero-screening policy for nonresident
reference preparation, while preserving unscreened final refinement and the
published conventional RHF reference. Do not relabel these older receipts as
measurements of the later library.

## Revisit

Profile actual generated-class gaps on tuned builds before changing their
dispatch. Further optimizations should target measured remaining classes or
source reuse without replacing exact reference semantics with density fitting.
## Hosted resource-ledger requalification after promotion

The first hosted CI for head `085aea32695c430cd4329f564220f16e4fc6c160`
compiled the canonical eight-shard release context successfully. The exact
checkout was synthetic merge `8827875eb60d328e675ac474f1728d46b537a888`
(tree `e0eea507909fea797811c105c101f0c130013ac0`).
[Job 115188395377](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38361914950/job/115188395377)
completed the shard-6 cache miss in 676.97 s with compiler exit zero. The gate
then rejected the old source/preprocessed/manifest identities and the expanded
kernel inventory. It did **not** report a resource regression or compiler
timeout.

[Artifact 11796364827](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38361914950/artifacts/11796364827)
is 43,143 bytes, ZIP SHA256
`57d60fa861b1420b1b50973efc64fcc290dc182d1c5f5a1018ee296b4f545d21`.
All 17 ZIP members pass CRC. The report-selected cache record is
`9d4360b4fda0afee9b5d13f61a45d4c20bf978eaf6f381d86d653a481320776d`;
its raw `cuobjdump.txt` reproduces all 102 reported kernel rows and its
`nvcc.log` matches the retained compile-record hash. Restored historical cache
records in the same archive are not the new measurement.

All 86 previously qualified kernels retain exactly the same REG/STACK/SHARED/
LOCAL values, including the ordinary dsss/ddss/ddps streaming consumers.
Sixteen added rows are the RHF/UHF ordinary/persistent variants of four newly
promoted value-only classes co-located in shard 6:

| Class | REG, ordinary / persistent | STACK bytes | SHARED, ordinary / persistent |
| --- | ---: | ---: | ---: |
| fdpp | 96 / 96 | 256 | 1520 / 1528 |
| ffps | 166 / 164 | 112 | 1520 / 1528 |
| fppp | 166 / 166 | 120 | 1520 / 1528 |
| fsps | 252 / 254 | 64 | 1504 / 1512 |

Each row has LOCAL zero; ptxas explicitly reports zero spill loads/stores.
Nonzero stack frames remain recorded rather than relabeled as spill-free
zero-stack kernels. These four classes are not the full 22-class promotion,
and static resource results do not establish dynamic traffic or new endpoint
performance. The independent device qualification above retains its scope.

Local regeneration matches the measured shard source SHA256
`699d162defae5ba377c018997bc765b3c097ca3d2e17c7ea75301b9ae02f7f06` and
manifest SHA256 `dd9ff7e8c9df954b5c82749b60d9331cbadb6afba32ab165889ab0e9e8152d9d`.
All 231 compiler/generator/compensated-header input blobs match the actual
CI checkout. The measured preprocessed identity, original toolchain/release
flags and exact resource rows deliberately extend the required ledger. No
resource limits, comparison rules or compile timeout are relaxed; all 408
single-field REG/STACK/SHARED/LOCAL mutation checks still reject drift.

The separate legacy integral-artifact byte-preservation test is a historical
migration receipt, not a permanent source-spelling contract. Preserve its
original JSON hashes and make replay explicit with
`GENERATIVEQC_CHECK_LEGACY_INTEGRAL_ARTIFACTS=1`. It intentionally detects this
later profile expansion. Ordinary CI still checks the retained catalog,
executable registry/ABI dispatch, capability and numerical contracts, cache
identity, and the independently measured resource ledger.
