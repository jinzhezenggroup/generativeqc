# Occupied XC: bounded proposals with strict matrix publication

Qualification spans November 4–5, 2026 on Slurm node1/main RTX 5090. The matrix
route remains the default; exact `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` enables
the initializer. This is neither a default promotion nor an all-functional
speedup claim.

## Contract and source

Admitted provisional features skip original-D verification and certificate
arithmetic. Finite features, canonical-frame ownership and a cheaper single
scratch panel guard initialization. The bounded stage then permanently hands
off to the original stored-D target, resets the energy epoch, and retains the
original physical/canonical publication and force gates. Unsupported inputs
retain the matrix route. Exact low-level callers keep two-panel certificates
and bitwise stored-D verification.

Final scientific/runtime source: `07ec901f833c584518a6cc8481efef5649ba90ab`.
Native source identity:
`5c9e1790313172a051d7b5dff755ecfee2275d3b9da09fe57cf803c0fde74223`.
Library SHA-256:
`4a9971dfd533b31cd8a4e5857bab04bb44dc67f76f25c93b6fc79d75f3d6569d`.
Release/C++20/O3/sm_120, strict XC `--fmad=false`, CUDA 12.9.1 and verified
sccache 0.16.0; existing caches retained. Shared compiler-cache statistics are
provenance, not exclusively attributable compile counts.

## Retained populations

- `endpoints-final.json.xz`: ten alternating matrix/proposal pairs for RKS/UKS
  cold/warm/moved/moved-warm E+F, 160 complete observations. Cold includes owner
  construction/preparation in a shared process/context, not process startup.
  Independent original/displaced PySCF 2.14.0/Libxc 7.0.0 default-grid oracles
  and paired comparisons retain **2e-9 Eh / 1e-7 Eh/Bohr** gates. Production CPU
  scientific derivative fallbacks are forbidden.
- `qualification.json.xz`: three alternating fresh-process pairs on the
  original **96-atom Direct Standard r2SCAN-3c** workload. Each process owns
  complete cold E+F and one frozen warm E+F, with native solver/grid defaults.
  The existing audited GPU4PySCF 1.8.1 grid-7/8 oracle, grid stability and
  matched CPU/GPU three-atom spot checks are reused without rewriting source,
  package versions or scientific inputs. Original independent gates remain
  **1e-5 Eh/atom, 1e-4 maximum and 3e-5 RMS Eh/Bohr**. Paired matrix/proposal
  vectors additionally pass **2e-9 Eh / 1e-7 Eh/Bohr**.
- Every complete energy, force vector, duration, iteration/Fock count and native
  work diagnostic remains retained. Timing excludes intrusive tracing/profiling.
  Scaling uses the original `max(2%, 2*(relative MADs summed))` gain floor;
  a below-floor change is not a speedup.
- Rejected cohorts stay separate: `rejected-reset-history.json.xz` retains all
  48 scientifically passing but slower reset-DIIS endpoints; its exact source
  tree is reconstructible with `rejected-source.patch.xz` on #2971 merge
  `c776b6cc12855b68195e4be27d0604b1b524690e`. The restored tree hash is
  `48cd81d567916ee8be8b2f9840383a1871a0c916`.
  `two-panel-small-endpoints.json.xz` retains the later 160 below-floor small
  observations. The capsule separately retains the full six-process
  `6918c4584b85a5571dc69cbab35fde77fade09d7` scaling regression and its profile;
  runtime-null certificate arithmetic is not relabelled as recovered speed.

## Measured result

| 96-atom complete E+F | Matrix median (s) | Proposal median (s) | Result | Iterations/Focks, both |
| --- | ---: | ---: | --- | --- |
| Cold | 30.500666 | 28.909603 | **5.22% less time, 1.055x**, above 2% floor | 10/10 |
| Frozen warm | 11.139000 | 11.232632 | 0.84% slower, below 2% floor | 1/1 |

Imports and CUDA-context setup precede the cold timer; constructor, preparation,
SCF, complete native D4/gCP, analytic forces and host export are included. No
warm speedup is claimed. Small-system changes are about 0–2% and are not a
general speedup claim. All 172 final timed endpoints pass their stated gates.

The separate same-image Nsight/DF trace records seven occupied generations
2–8, followed permanently by matrix work. There are 231 factor and 132 matrix
density-product launches versus 363 matrix launches. Combined density-product
device time is **5.216700 → 3.674313 seconds** across cold+warm; feature time is
0.138856 → 0.096979 seconds. These intrusive samples do not enter clean medians.
They are neither hardware FLOP measurements nor an additive decomposition of
the complete endpoint, and do not imply that potential/force work is accelerated.

**398 focused host/compiled tests** pass. **236 emitted GPU cases** pass normally
and under each of memcheck/synccheck/racecheck, zero errors/hazards (Slurm 15891).
**12 native tests** pass normally (15894) and under memcheck (15895), zero errors.
They cover exact/provisional work avoidance, high-rank one-panel admission,
compact scratch refusal, StrictAudit exclusion, controls and geometry lifetimes.
236 optional GPU skips in the host run are not device qualification. Earlier
missing-cache/SONAME/artifact-copy and diagnostic-serialization setup failures
remain explicitly identified in the capsule, not passing performance samples.

## Reproduction and limits

From the repository root, no GPU or native library is needed for replay:

```bash
PYTHONPATH=.:python python benchmarks/results/occupied-xc-audited-proposals-20261104/replay.py
PYTHONPATH=.:python python -O benchmarks/results/occupied-xc-audited-proposals-20261104/replay.py
```

The reader recomputes independent/stability/paired gates, completeness, work,
timing medians and variability floors; corrupted energy/force/source/population
controls must fail. The small receipt also has a stdlib-only reader:

```bash
python benchmarks/occupied_xc_proposals.py --replay benchmarks/results/occupied-xc-audited-proposals-20261104/endpoints-final.json.xz
```

Use `benchmarks/occupied_xc_proposals.py --help` for small-case execution.
For scaling, run `python -m benchmarks.readme_defaults standard --method
r2scan-3c --atoms 96 --warm-repeats 1 --inputs inputs --source-revision SHA
--output .artifacts/occupied-xc/r0-0.json` with selectors in orders `0,1`,
`1,0`, `0,1`, each in a fresh process. Select the exact library and set
`GENERATIVEQC_PROFILE=off`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`PYTHONPATH=.:python:tests/python`. All real-GPU execution must use finite Slurm,
for example `srun --input=none --partition=main --nodelist=node1
--gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:25:00`; preserve its visibility.
Qualification uses `GENERATIVEQC_TEST_OCCUPIED_XC=1` and
`CCACHE=/home/jzzeng/.local/bin/sccache` for cached host probes. Supply the
native SONAME symlink alongside the library for linked C++ plan tests.

Static final-library cuobjdump qualifies changed and fallback launch legality,
not measured occupancy, spill traffic or allocator peak. Complete native force
endpoints and independent oracles qualify only the stated geometries/methods.
No universal XC sensitivity bound, trajectory accuracy claim or claim that the
remaining integral/force work is accelerated follows from these measurements.
