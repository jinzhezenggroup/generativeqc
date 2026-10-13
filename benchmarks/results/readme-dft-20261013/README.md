# HF and DFT energy and analytic forces: cold and warm

Measurements are based on master `e0ad22036fe83d9bcaed277d0be9a1dbcb453e1c`
on 2026-10-13. PBE, PBE0, B3LYP and r2SCAN at **12 and 96 atoms** were remeasured
with this change's conditional force-memory handoff; other points retain the
original master measurements. HF and DF were newly measured using the fixed build.
Each point retains its actual native source
inventory identity and library hash in [summary.json.xz](summary.json.xz).
[manifest.json](manifest.json) identifies both snapshots and the changed files.

![Cold and warm complete direct HF/DFT endpoints](dft-cold-warm.svg)

![Cold and warm complete DF HF/DFT endpoints](df-cold-warm.svg)

See the [DF timing table](df-timings.md) for the separate density-fitted population.

See the [complete timing table](timings.md) for all sizes and both engines.
Cold is one observed call per point; warm series connect the median of all five
complete endpoint repeats and show their minimum–maximum range. Crosses and †
retain differing SCF iteration/build counts. Only failed or missing endpoints
break lines; endpoint statistics never imply equal SCF work.

## Measurements and acceptance

The direct matrix has **29 of 32 native points** passing all 12 endpoint checks
(348 native observations); **30 of 32 reference points** completed.
Across accepted native observations, maximum absolute errors were **2.319e-11 Eh**
and **9.159e-11 Eh/Bohr**. All DFT reference electronic XC evaluations
reported the CUDA Libxc backend. These are complete endpoint observations;
no iteration-normalized speedup is claimed. HF and DF additions are tabulated below.

| Functional | Qualified native sizes (atoms) |
| --- | --- |
| HF | 3, 12, 48, 96 |
| PBE | 3, 12, 48, 96 |
| PBE0 | 3, 12, 48, 96 |
| B3LYP | 3, 12, 48, 96 |
| r²SCAN | 3, 12, 48, 96 |
| r²SCAN-3c | 3, 12, 48 |
| M06-2X | 3, 12, 48, 96 |
| ωB97M-V | 3, 12 |

The 3-atom M06-2X cold call took **229.691 s**, including **226.157 s**
in the recorded force-provider lookup/construction interval. This first-call
cost remains in the cold comparison. One observed cold call with retained
caches does not establish a monotonic size trend or an empty-cache launch cost.

Unsuccessful points remain explicit; their partial timings are excluded:

| Functional | Atoms | Engine | Outcome |
| --- | ---: | --- | --- |
| r²SCAN-3c | 96 | native | Time limit; last stage `cold/0` |
| ωB97M-V | 48 | reference | Time limit; last stage `moved/0` |
| ωB97M-V | 48 | native | Skipped: complete reference unavailable |
| ωB97M-V | 96 | reference | Time limit; last stage `cold/0` |
| ωB97M-V | 96 | native | Skipped: complete reference unavailable |

A converged SCF energy with missing analytic forces cannot pass the endpoint gate.
The ωB97M-V reference has a 1800-second per-point limit and a 3600-second
outer allocation limit. Its retained outcome identifies which limit stopped
each incomplete point. No native timing is inferred from an incomplete oracle.

Initial host validation: 140 tests passed across force ownership/admission, ctypes
bridges, telemetry and benchmark publication. The real-device lifetime/oracle
probe and ten native CLI force/finite-difference cases also passed. The HF/DF
publication follow-up passes 215 host tests, including endpoint gates, plotting
statistics, prepared-force fixtures and evidence-retention checks.
The final interruption/reporting follow-up passes 27 focused benchmark tests;
operator cancellation retains partial evidence without publishing its timings.

Every published native point must complete 12 independently checked endpoints:
cold, five fixed-density warm calls, changed geometry, and five moved-geometry
warm calls. All must return finite energy and host forces, converge and meet
absolute errors of **1e-8 Eh** and **1e-7 Eh/Bohr**. The comparison does not divide
wall time by iterations or assert equivalent stopping rules between engines.
The full scalar observations retain native iterations/Fock builds, reference
SCF J/K builds, residuals, errors and available resource/work diagnostics.

## Density-fitted coverage and negative probes

The separate DF matrix uses **HF, PBE, PBE0 and r²SCAN**, the same spherical
cc-pVDZ-JKFIT auxiliary snapshot in both engines, and independent DF references.
All native storage, response and memory planning use the recorded source
snapshot's public defaults.
GPU4PySCF gradients explicitly include auxiliary-basis/metric response as well
as moving-grid response for DFT. Direct and DF are different approximations
and are never gated against each other's energies.

The DF matrix has **14 of 16 native points** passing all 12 endpoint checks
(168 observations); all **16 reference points** completed. Maximum accepted
native errors are **8.027e-11 Eh** and **1.557e-10 Eh/Bohr**. HF and PBE0 pass
at all four sizes; PBE and r²SCAN pass at 3, 12 and 48 atoms.

The two 96-atom semilocal points expose a severe default streamed-DF performance
defect, tracked in [#2411](https://github.com/jinzhezenggroup/generativeqc/issues/2411).
Each eventually completed its first cold call, but its warm population remained
incomplete. These partial cold observations are diagnostic evidence, excluded
from both the plotted timings and acceptance totals:

| Method / atoms | Cold E+F (s) | Integral derivative interval (s) | DF streamed |
| --- | ---: | ---: | --- |
| PBE / 48 | 21.634 | 0.408 | false |
| PBE / 96 | 1225.048 | 627.548 | true |
| r²SCAN / 48 | 42.475 | 0.382 | false |
| r²SCAN / 96 | 1221.951 | 617.521 | true |

PBE job 8012 exhausted the original 1800-second whole-point limit during
`warm/0` (exit 124). r²SCAN job 8014 was explicitly cancelled at
2026-10-13 14:01:01 +08:00 during `warm/0`, before that limit; its outcome is
**interrupted**, not timeout. Both cold calls converged and met the numerical
gates. The observed route and phase timings establish a performance cliff;
they do not establish a deadlock or its precise repeated-work mechanism.
The PBE0/96 control uses a resident DF plan and takes 77.368 s cold, including
3.107 s in the integral derivative interval.

Supplemental probes retain the other requested functionals' actual limits;
they are not plotted as qualified DF series:

| DF method | Atoms | Native outcome |
| --- | ---: | --- |
| B3LYP | 3 | Energy converges, then force fails: `invalid KS final-state model for functional 3` |
| r²SCAN-3c | 3 | Public capability does not advertise analytic forces for this DF/basis combination |
| M06-2X | 3 | Preparation rejects the split-global-hybrid DF combination; requires direct J/K |
| ωB97M-V | 3, 12 | 3 atoms pass; 12-atom cold energy error **2.4092e-8 Eh** exceeds the **1e-8 Eh** gate |

The larger ωB97M-V DF probe was stopped after the 12-atom gate failed; 48 atoms
retain the interrupted reference outcome, and 96 atoms were not attempted.
Its 3-atom success, 12-atom failure and interrupted larger probe remain in
[df-summary.json.xz](df-summary.json.xz), alongside the other negative probes.
All eight methods remain represented in the direct comparison; unsupported
DF forces are not substituted with direct forces.

The benchmark findings have separate actionable issues with reproduction,
source identities and acceptance gates:

| Finding | Follow-up |
| --- | --- |
| PBE/r²SCAN DF streaming performance cliff | [#2411](https://github.com/jinzhezenggroup/generativeqc/issues/2411) |
| B3LYP DF final-state/force failure | [#2412](https://github.com/jinzhezenggroup/generativeqc/issues/2412) |
| ωB97M-V DF energy accuracy failure | [#2413](https://github.com/jinzhezenggroup/generativeqc/issues/2413) |
| M06-2X DF support missing | [#2414](https://github.com/jinzhezenggroup/generativeqc/issues/2414) |
| r²SCAN-3c DF force capability missing | [#2415](https://github.com/jinzhezenggroup/generativeqc/issues/2415) |
| Direct r²SCAN-3c/96 incomplete cold endpoint | [#2416](https://github.com/jinzhezenggroup/generativeqc/issues/2416) |
| GPU4PySCF ωB97M-V/48,96 incomplete reference populations | [#2417](https://github.com/jinzhezenggroup/generativeqc/issues/2417) |

These findings remain unresolved by the force-cache and plotting fixes. The
reference issue is a benchmark blocker; the available evidence does not prove
an upstream GPU4PySCF correctness bug.

## Default OOM correction

The original master campaign converged SCF but ran out of memory while creating
forces at 96 atoms for PBE, PBE0, B3LYP and r2SCAN. With the optional ERI cache
still resident, the new default runs had only **234–252 MiB** free at force entry.
The handoff released **15,120,367,225 bytes (14.08 GiB)** of optional provider
storage and retained the converged state and warm seed. All four methods now
pass all 12 cold/warm/moved checks, including the formerly failing cold calls.
The 12-atom controls completed without cache eviction; the native lifetime
probe separately verifies retention of admitted ERIs when memory is sufficient.

[force-memory.json.xz](force-memory.json.xz) retains the per-call memory receipts
and additional validation. A diagnostic pressure injection left 64 MiB free:
master failed with OOM; the fixed endpoint completed all 12 checks. Its timings
are excluded from this figure. M06-2X and ωB97M-V three-atom controls also passed.
Together these checks cover **132 fixed-build endpoints**, with maximum
errors **1.046e-11 Eh** and
**9.159e-11 Eh/Bohr**. Original failed records and
an unmodified master report remain in the ignored evidence directory; they are
not presented as measurements of the fixed build. The r2SCAN-3c and ωB97M-V
reference timeouts listed above are separate limitations.

## Scientific and timing boundaries

- Nested water clusters: 3, 12, 48 and 96 atoms, batch size one, restricted FP64
  calculations. All methods use full def2-SVP spherical shells (24 AOs per water)
  except r2SCAN-3c, whose defining def2-mTZVPP basis has 34 AOs per water.
- Both engines use the same explicit 48×16×32 per-atom moving quadrature, three
  original-Becke partition iterations, no radius adjustment and analytic grid
  response. This is a matched-grid comparison, not a claim about each package's
  default grid or chemical grid-convergence accuracy.
- The public native endpoint and GPU4PySCF both include SCF energy, analytic
  forces, synchronization and force return to the host. No native calculation
  uses a reference density or reference energy/force implementation.
- Cold includes calculator/engine construction, preparation and the first
  endpoint. Process startup, provenance collection and input snapshot loading
  are excluded. Existing compiler/artifact caches are retained; these are API
  cold starts, not empty-cache compilation or process-launch timings.
- Each warm series reuses one frozen engine-local post-cold density. The moved
  geometry displaces the first hydrogen by 0.001 Bohr along z; its own post-move
  density is frozen for the five moved replays. Densities are not progressively
  improved across measured repeats.
- Native tolerances are 1e-12 Eh for energy, 1e-10 for density and 1e-12 for
  screening, with 100 SCF iterations maximum. GPU4PySCF uses 1e-12 Eh energy,
  1e-10 orbital gradient and 1e-14 direct-integral screening tolerances. Its stock
  DIIS is unchanged. Full-density potential rebuilds suppress both incremental
  inputs, as in the existing PBE0 protocol, and their work stays timed.
- r2SCAN-3c includes D4 (`s6=1`, `s8=0`, `s9=2`, `a1=0.42`, `a2=5.65`,
  `ga=2`, `gc=1`) and gCP. Its reference combines GPU4PySCF electronic calculations
  with CPU dftd4 4.3.0 and the independent CPU gCP qualification provider. Both
  corrections and their geometry-dependent setup are evaluated inside **every**
  reference endpoint. The native endpoint uses CUDA D4 and native CPU gCP; its
  corrections also remain inside every timed endpoint. The figure labels the
  reference CPU corrections.
- WB97M-V includes self-consistent VV10 and the shared molecular density-screening
  policy. Direct and DF use separate, matching independent oracles. No mixed
  precision, experimental AO selection or preliminary-density override is enabled.

## Hardware and provenance

Runs use Slurm on `n1` (`node1`), `main`, one NVIDIA GeForce RTX 5090 per
allocation, eight CPU threads and finite job/point timeouts. Independent method
groups may use separate GPUs on the same node. Slurm's device visibility is
preserved. Device state snapshots were taken **before** measurement; reduction
preserves and corrects the shared helper's legacy sampling-point label.

Runtime: Python 3.13.9, GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0, cuTENSOR
2.2.0 and CUDA 12.9. Native compilation uses GCC 11.4, NVCC 12.9.86, the Release
sm_120 preset and verified ccache 4.5.1; generated Ninja commands invoke the
launcher. Retained cache statistics are shared-cache snapshots, not an isolated
per-build hit-rate claim. Runtime JIT uses verified sccache 0.16.0 where needed;
the execution host's default C++ compiler is GCC 13.3, as recorded in the raw
environment, distinct from the core/package build compiler. The host has two
AMD EPYC 9354 processors; each Slurm allocation requests eight CPU threads.

The remote source is a plain synchronized snapshot, so its raw Git fields are
null. The manifest records the base revision and both native source inventories and
loaded-library hashes; per-file hashes bind the actual benchmark and Python
consumers. The eight remeasured DFT points and all new HF/DF points use the fixed build;
the remaining direct DFT points preserve their original master provenance.
The exact earlier source/build campaigns are retained losslessly in
[source-campaigns.json.xz](source-campaigns.json.xz).

## Reproduction

Install the normal native build dependencies plus GPU4PySCF, CuPy, PySCF, dftd4,
Matplotlib and a supported compiler cache. Use the master base plus this change
to reproduce the fixed behavior. The mixed historical matrix records each
original snapshot separately. Build with the production preset and explicit
verified launchers, retaining the shared compiler cache:

```bash
ccache --version
export CCACHE_BASEDIR="$PWD"
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v ccache)" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v ccache)" \
  -DGENERATIVEQC_BUILD_TESTS=OFF
cmake --build --preset cuda-release-sm120 -j 6
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export README_BENCHMARK_OUTPUT=.artifacts/readme-dft
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=04:00:00 \
  bash benchmarks/run_readme_dft_benchmarks.sh both
PYTHONPATH=.:python python -m tools.render_readme_dft \
  --raw .artifacts/readme-dft --output benchmarks/results/readme-dft-20261013 \
  --source-commit "$(git rev-parse HEAD)"
```

`README_BENCHMARK_METHODS`, `README_BENCHMARK_SIZES` and
`README_BENCHMARK_POINT_TIMEOUT` select a bounded subset; defaults are the eight
documented methods, four sizes, five repeats and 1800 seconds per engine/size.
For the separate DF campaign, set `README_BENCHMARK_DENSITY_FITTING=cuda` and
use a distinct `README_BENCHMARK_OUTPUT`. `README_BENCHMARK_AUXILIARY_BASIS`
selects the explicit auxiliary snapshot; the default is cc-pVDZ-JKFIT. Select
`README_BENCHMARK_METHODS="hf pbe pbe0 r2scan"` for the plotted DF population,
and pass `--density-fitting --methods hf pbe pbe0 r2scan` to the reducer.
Unsupported combinations remain explicit outcomes.
The runner also accepts `reference` and `native` separately, preserving the
independent reference JSON. Use a fresh output directory for each campaign.

Re-render the committed reduction without a GPU:

```bash
PYTHONPATH=.:python python -m tools.render_readme_dft \
  --summary benchmarks/results/readme-dft-20261013/summary.json.xz \
  --output .artifacts/readme-figure
```

The compact archive retains original-file hashes and independent cold/moved
reference vectors. Large native arrays, build products and trial logs remain in
ignored local artifact storage. Hashes establish integrity, not authentication
or a new numerical qualification of a different source revision.

## Connected medians and the GPU4PySCF replay audit

The previous renderer deliberately broke a line when SCF iterations/builds
varied between repeats, although all those endpoints completed. The updated
figure connects medians of all accepted endpoints and keeps full ranges and
work counts. It neither discards slower observations nor normalizes by iterations.

A separate [GPU4PySCF 1.8.1 audit](reference-replay-audit.json) reproduces
PBE/48 warm iteration counts **4, 5, 4, 4, 4**. The frozen input density remains
byte-identical through every replay; repeated potentials differ by up to
**3.553e-15**. The stock solver creates fresh DIIS history per call and uses
a strict **1e-12 Eh** energy test alongside its orbital-gradient test. Recorded
energy changes near that cutoff can add an iteration. Every replay converged.
These observations do not establish a wrong-energy/force bug in GPU4PySCF.
Audit timings are excluded from both figures; no reference tolerance was relaxed.

## Evidence storage

The XZ archives are ordinary JSON after decompression (`lzma.open(path, "rt")`).
They preserve every scalar observation, diagnostic field and independent
reference vector. Original gzip bytes are recoverable from the existing Git
commit recorded in the manifest. The separately reviewed upstream density
fixture storage repair remains intact. No evidence was published as a Release
or removed from Git history.

Later upstream integrations on this PR do not relabel these frozen measurements
as timings of a newer source or initialization policy. Use the recorded source
inventories and consumer hashes when reproducing this population.
