# Live-panel XC capacity: complete Direct DFT cold qualification

GGA retains four AO jets but only one density-product jet. Sizing the optional
XC arena from the larger live panel admits 32 original point tiles instead of
8 on water-96. This report retains the original frozen qualification and a
separate composition with current master after #2676/#2743 landed.

## Current-master composition

Three rotating fresh-process rounds compare master `71504dc55`, candidate
`27a977ef4` and stock GPU4PySCF. All source/binary identities are independently
bound to their exact scientific input inventories. Endpoint settings, force
work and native Fock counts match the original protocol below.

| Complete endpoint | Current master median (s) | Candidate median (s) | GPU4PySCF median (s) | Native improvement |
| --- | ---: | ---: | ---: | ---: |
| PBE0 cold | 21.628970 | 21.310210 | 20.382071 | 1.47%, below 2% floor |
| PBE cold | 17.717633 | 17.264743 | 16.205765 | 2.56%, above 2% floor |
| PBE0 warm | 6.833278 | 6.755254 | 6.784568 | below 2% floor |
| PBE warm | 5.312277 | 5.269002 | 5.623311 | below 2% floor |

The observed current-master PBE0 gap is **0.9281 seconds / 4.55%**. Its
three-round native improvement does not clear the prespecified floor. PBE's
cold gain does clear it. The earlier 0.8400-second gap below belongs to its
own frozen cohort and is not substituted for this current-source result.

A separate five-pair PBE0 confirmation was prespecified after observing the
below-floor three-round gain, with unchanged binaries, settings and the same
floor. It alternates fresh master/candidate processes and never pools prior
samples or combines its native timings with an earlier GPU4PySCF population.
Its cold medians are **21.642982 / 21.224130 seconds**, an observed **1.94%**
improvement, also below the unchanged 2% floor. All ten records / 20 E/F calls
pass the independent gates. The 21.5321-second candidate sample is retained.
Current-source PBE0 cold speedup remains unqualified by this performance floor;
the promotion has a qualified current-source PBE cold gain and the separately
qualified original PBE0 cohort. No universal or current PBE0 speedup is claimed.

All 36 current-source static E/F calls and three original/warm/moved calls
pass their unchanged independent gates. Current-source validation repeats
147 host tests, all 34 public native force/occupied-default GPU tests, three
native point/ragged/spatial suites and both zero-error sanitizers.

The first composition attempt exposed a timestamp-stale incremental build:
restoring an older Git archive left the old candidate object in the supposed
master library. The loaded hash/identity revealed the mismatch before
acceptance. Its seven complete two-call records and interrupted zero-call arm
remain under `integration.quarantined_records`; every returned E/F vector also
passes the independent numerical gates, but the entire comparison is rejected
for source provenance. None of these timings enter a valid median. Master was
rebuilt after forcing its sole differing production translation unit, then its
scientific source digest was independently verified against the loaded binary.
The candidate's fully rebuilt bytes stayed unchanged. The original frozen
cohort was unaffected. Both build attempts and repair protocol remain retained.

## Original frozen endpoint population

The original three-round PBE0 cold median improves **2.44%**, from **21.6781 to
21.1493 seconds**. Stock GPU4PySCF measures **20.3093 seconds**, a gap of
**0.8400 seconds / 4.14%** in that cohort.

Each fresh process times constructor, preparation, SCF, full analytic forces
and host publication, followed by one frozen same-owner warm call. Imports and
prebuilt artifacts are outside timing. Every completed sample is retained,
including the first 23.1677-second baseline PBE0 sample. Profilers and pilot
runs are separate populations and are never pooled into headline medians.

The frozen protocol uses three rotating rounds, water-96/def2-SVP spherical,
Direct Standard, production grids and SCF controls, and stock GPU4PySCF DIIS.
Eight Slurm-bound CPU cores, one OMP/OpenBLAS/MKL thread, and one RTX 5090 are
used throughout. Every GPU invocation uses finite `srun`; scheduler device
visibility is preserved. The predeclared improvement floor is
`max(2%, 2*(MAD_master/median_master + MAD_candidate/median_candidate))`.
Both headline cold improvements exceed their 2% floor. This small population
is not a confidence interval or a claim of universal profitability.

| Complete endpoint | Frozen master median (s) | Candidate median (s) | GPU4PySCF median (s) | Native improvement |
| --- | ---: | ---: | ---: | ---: |
| PBE0 cold | 21.678081 | 21.149298 | 20.309319 | 2.44% |
| PBE cold | 17.738103 | 17.182746 | 16.205383 | 3.13% |
| PBE0 warm | 6.816484 | 6.768160 | 6.763901 | below 2% floor |
| PBE warm | 5.323350 | 5.260057 | 5.622063 | below 2% floor |

All native headline PBE0 cold calls use 9 Focks; PBE uses 11; warm uses 1.
GPU4PySCF reports 9/11/1 iterations, but its Fock counts are unavailable.
Iterations are not treated as equivalent cross-backend work counts.

The single-pair holdouts below are numerical and compatibility controls;
they do not establish statistically qualified speedups. The small PBE0/3 cold
regression is retained. Each row's cold/warm native Fock counts match.

| Held-out case | Master cold / warm (s) | Candidate cold / warm (s) |
| --- | ---: | ---: |
| PBE0 Direct 3 Standard | 1.8592 / 0.0788 | 1.9331 / 0.0792 |
| PBE0 Direct 12 Standard | 2.9339 / 0.3762 | 2.8927 / 0.3757 |
| PBE0 Direct 48 Standard | 9.7133 / 2.4258 | 9.4908 / 2.4112 |
| PBE Direct 3 Standard | 1.3849 / 0.0495 | 1.3379 / 0.0495 |
| B3LYP Direct 48 Standard | 10.0450 / 2.5121 | 9.5817 / 2.4746 |
| PBE0 DF 96 Standard | 31.9457 / 11.8984 | 31.3694 / 11.8498 |
| PBE0 Direct 96 Fast | 21.8592 / 6.3449 | 21.3981 / 6.3072 |
| r2SCAN Direct 96 Standard | 19.3410 / 5.9885 | 19.2879 / 5.9850 |
| r2SCAN-3c Direct 96 Standard | 42.3306 / 15.7402 | 42.2749 / 15.7607 |

The separate three-pair same-image pilot changes only the explicit cap from
32 to 128 MiB. PBE0 cold medians are 21.9182 / 21.2471 seconds, a 3.06% gain
above its 2.22% floor; warm changes remain below the floor.

## Mechanism, work and resource boundaries

The policy requests at most 32 original tiles within
`32 MiB * max(AO jets, density-product jets)`. LDA remains at 32 MiB; meta-GGA
remains at 128 MiB; GGA now receives 128 MiB. The exact planner still selects
actual capacity. Explicit byte caps remain exact. Automatic allocation
refusal retries 32 MiB, then one tile. A public resource ledger still admits
zero optional batch bytes, preserving force/fleet reservations.

Final-candidate traces compare explicit 32 MiB (`candidate-off`, meaning the
larger allowance is off) against the default. Per Fock:

| Plan/work fact | 32-MiB cap | New default |
| --- | ---: | ---: |
| Original SCF tiles | 4,213 | 4,213 |
| Admitted batch tiles | 8 | 32 |
| Retained arena bytes | 27,202,568 | 88,317,344 |
| Point-AO visits | 190,426,944 | 190,426,944 |
| Point-AO-square sum | 35,027,345,728 | 35,027,345,728 |

Across cold plus warm, compact potential and equal-spin point kernel
submissions drop from 5,270 to 1,320. Four-jet AO launches remain 46,343.
In the separate final intrusive profiles, equal-spin point kernel GPU time
is 0.3402 / 0.0908 seconds, potential kernels 0.6177 / 0.5603 seconds, and
four-jet AO kernels 1.1818 / 1.1816 seconds. These cold-plus-warm kernel sums
explain the mechanism; they are not additive attribution of clean cold time.
The final profile retains about 58.3 MiB more optional memory per owner.

Grid points, local maps, precision, functional algebra, ordered accumulation,
screening and solver settings do not change. The trace counters are host plan
facts and logical work; they add no device counter, launch or fence.

## Independent acceptance and retained failures

The bundle contains 47 complete static records / **94 E/F calls**: 18 headline,
18 holdout, 6 pilot and 5 intrusive-profile records. Every call passes the
previously audited dense independent references in
[`readme-final-20261022`](../readme-final-20261022/), using unchanged gates of
`1e-5 Eh/atom`, `1e-4 max` and `3e-5 RMS Eh/Bohr`. Maximum errors across all
94 calls are `3.044e-6`, `5.402e-5` and `2.473e-5`, respectively.
The largest paired native headline energy/force differences are
`1.365e-12 Eh` / `1.168e-9 Eh/Bohr`.

Three additional PBE0/96 cold/warm/moved calls use fresh tighter matched-grid
GPU4PySCF references after timing. Their gates remain `2.5e-6`, `2.5e-5`,
`7.5e-6`; maximum measured errors are `6.869e-13 Eh/atom`,
`9.349e-7 max` and `2.053e-7 RMS Eh/Bohr`. All **97 calls** in the original cohort pass. Including the separate current
composition and confirmation, **156 source-qualified calls** pass. The 14
returned calls from the quarantined build attempt pass numerical gates but
remain excluded from source/performance acceptance.

Validation retained in the bundle:

- 147 host resource/capability/compact/ragged/grid-inventory tests passed;
  189 policy/CUDA ownership tests passed; compiler structure has zero errors.
- Native CUDA point-batch, ragged-batch and spatial-grid suites passed.
- Five independent UKS/finite-difference/replay/failure-isolation tests passed.
- All 32 public native CUDA force tests passed, including independent RKS/UKS
  references, default grids, empty spin, minimum fleet budgets and rollback.
- Compute Sanitizer memcheck and synccheck: zero errors.

Initial failures remain visible rather than being converted to passing
measurements. Host fixture stubs first lacked the new AO-jet field, then
assumed the old capacity; corrected tests exercise actual compiled planning.
An initial GPU fixture lacked `CUDACXX`. The old public batch test required a
Python executor and an exact internal execution count, despite the already
promoted native owner; it also fails on frozen master. It now checks failed
neighbor isolation and E/F agreement with a fresh owner at moved geometry.

Two strict default-grid reference cases also fail on frozen master. The
independent PySCF helper assigned `radii_adjust` after installing coordinates,
which invalidated the arrays, rebuilt the grid and omitted Treutler-adjusted
radii. The helper now configures the partition first, installs the exact native
quadrature, and asserts its arrays remain unchanged after SCF and gradient.
This restores about `1e-13 Eh` / `1e-11 Eh/Bohr` agreement without changing
production science or weakening tolerances. Initial failures, baseline
reproductions and diagnostic logs are retained.

The first B3LYP48 holdout failed because its existing stationary AOT package
was absent and returned no E/F samples. Its complete failed record/log remains
under `failed_records` / `packaging-failure`. The matching B3LYP RKS/SPD AOT
targets were built and added to both campaign library directories after proving
identical contracts in the two source closures. The frozen original directory
was left untouched; the copied master core and candidate core hashes did not
change. Only the unmeasured holdout and remaining planned cases resumed.
No completed timing was replaced. The full continuation protocol is retained.

## Source, build and replay

The frozen master binary is from `72e0a7e5a011592428b5625acdfa6f19168645ac`;
the candidate is `48918670f59a8e10c254c822565a17796aa42f54` plus the retained
`qualified.patch`. Between those revisions, runtime/build differences affect
CLI code and CLI test registration only; relevant DFT/compiler code is unchanged.
The baseline keeps its original identity. Final test-only fixes and report
edits postdate the original measurement. Current-master composition has its
own full input inventories, libraries and exact-source attestation; only
`src/dft/cuda_ks.cpp` differs between its production source trees.

[`manifest.json`](manifest.json) pins both core libraries, the receipt archive,
reference dependencies and raw artifact inventory. `retained_files` contains
all three candidate input manifests/patches, the baseline inventory, exact
scripts/protocols, cache statistics, profile CSV reports and validation logs.
`qualified-source.tar` is the formatted source actually built and measured;
earlier prototype/final tarballs are explicitly superseded. The build uses
Release, CUDA 12.9.86, sm_120 and verified sccache 0.16.0. Generated Ninja
commands invoke sccache; local compiled-host tests use verified ccache 4.5.1.
The build's initial missing dependency prefix and wrapper-nvcc setup attempts
remain in the raw build logs. No cache was cleared or disabled.

From the repository root, with NumPy installed:

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-ao-batch-capacity-20261024/reproduce.py
PYTHONPATH=python:. python -O benchmarks/results/pbe0-ao-batch-capacity-20261024/reproduce.py
```

Replay verifies the exact population (including the failed attempt), every
independent E/F gate, binary identities, paired settings/Fock work, resource
bounds, trace capacity/unchanged AO work, moved calls and claimed cold gain
floors. Manual negative cases for missing populations, wrong binaries, bad energy,
missing moved calls and omitted failures fail closed under `-O`, including
the current-master cohort.
These historical source identities are opt-in evidence, not permanent tests
of production function spelling.

Raw source tarballs, profiler databases and large build logs remain in the
ignored campaign at `node1:/data/jzzeng/pbe0-cold-chase-20261024/`; their exact
byte counts/digests are retained. The compressed full receipts remain below
1 MiB and the complete change below the 2-MiB evidence review budget. No
release, external archive, cap increase or timing-population mixing is used.
