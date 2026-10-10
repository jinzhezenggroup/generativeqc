# Reciprocal MD-J complete PBE0 cold qualification

The candidate shares one canonical native `fill_coulomb` source across both
independently screened uniform MD-J density directions for unordered pair-angular
classes 00, 01, 02, 11, 12 and 22. Higher angular classes, partially screened
residual work, independent generated K, XC and force consumers are unchanged.
`GENERATIVEQC_MD_J_RECIPROCAL=0` retains the incumbent oriented contraction;
unset or `1` selects the reciprocal schedule. The resident owner freezes this
choice at construction. The existing 128-MiB MD admission and normal-J fallback
remain authoritative, including the public native-resource-ledger fallback.

## Complete endpoint, not a kernel claim

The protocol is exact-direct PBE0-RKS energy plus complete host forces for
96 atoms / 32 waters, 768 spherical def2-SVP AOs, FP64, 2,359,296 unpruned
48x16x32 moving-grid points and three Becke iterations. Native E/D tolerances
are 1e-12/1e-10 and screening is 1e-12. Independent GPU4PySCF uses the same
basis/grid, stock DIIS and explicit full-density rebuilding with screening
1e-14. No reference initial density, preliminary solve, tolerance change or
incremental/ordered-DIIS opt-in is used.

Cold uses fresh processes, densities and owners with persistent disk caches
retained. It includes preparation through synchronized first public energy and
host forces; imports, context, Calculator construction, serialization and
post-result teardown are excluded. `initial_density_used` in native diagnostics
is true for the ordinary public native initialization; it is not evidence of an
oracle seed. Warm/fallback flags and the recorded launchers distinguish that
initialization from an imported converged density.

Slurm **7045** uses one library on node1's RTX 5090, eight CPUs and an immutable
18-minute launcher. The measured source is master
`82c166caca181fc0df51b1ff9378b972e3b1eba8` plus this patch. Integration HEAD
`8eaaa66b3113d17bf2c492423aea0ec43ba6db4f` adds only CI/diagnostic tooling;
native, compiler, CMake and benchmark sources are unchanged between these two
master revisions. Source hashes and the library receipt identify measured bytes,
not merely the master label.

| Arm | Complete cold samples (s) | Median (s) | Fock builds |
| --- | --- | ---: | --- |
| Incumbent, `0` | 94.582284, 98.818512, 94.944027 | 94.944027 | 17, 18, 17 |
| Reciprocal, unset | 107.504216, 89.159602, 88.905666 | 89.159602 | 22, 17, 17 |

The complete cold median improves **6.09%**, exceeding
`max(2%, 2 * (baseline relative MAD + candidate relative MAD)) = 2%`.
All histories, including the candidate's slower 22-Fock trajectory, remain
retained. Do not normalize timings by Focks, discard extra-iteration branches
or interpret this small cohort as a universal/tail-latency guarantee.

Independent errors are recomputed from the retained complete vectors against
the frozen reference, not accepted solely from the original `gate` flags.
Every endpoint passes 1e-8 Eh energy and 1e-7 Eh/bohr per-entry force gates and
physical-residual RMS <=1e-10. `summary.json` preserves individual errors,
histories/work counts and acceptance thresholds.

## Broad non-regression

Slurm **7033** uses the same mathematical patch on master `15bc69700`, before
the final-source rebuild. For each size it retains three independent processes
per arm and every cold, warm, changed-geometry and changed-geometry-warm result.
This is supporting scope coverage, not a substitute for the final-source cold
population above. Unset ordered DIIS remains serial after the master updates.

| Atoms | Cold gain | Warm gain | Moved gain | Moved-warm gain |
| ---: | ---: | ---: | ---: | ---: |
| 3 | +0.05% | +0.02% | -0.18% | -1.69% |
| 48 | +3.89% | +1.57% | +4.24% | +1.65% |
| 96 | +4.63% | -0.74% | +4.24% | +0.14% |

Every scientific gate passes. All regressions remain inside the same
2%-floor/relative-MAD envelope; subthreshold changes are not advertised as wins.
The 96-atom cold and moved gains exceed their respective 3.42% and 4.01% gates.

## Executed work and independent kernel gates

Slurm **7024** freezes one native density with SHA-256
`a611e324fbbd69e37a5fa2528cecc0138af982ac0edca8218f5acef47c7fbfd3`.
Resident J event medians are 1.618226/1.290978 s, a diagnostic 20.22% improvement.
These are not complete endpoint timings.

- Uniform radial evaluations: 1,926,600,288 -> 1,001,504,816 (**48.02% fewer**).
- Density directions remain 1,926,600,288; Hermite summands remain 23,506,259,200.
- Residual tested candidates remain 209,520,685, admitted shell tasks 12,903,112,
  and primitive radial sources 93,656,520.
- `GENERATIVEQC_MD_J_WORK_COUNTS=1` charges 1200 diagnostic bytes inside the
  existing allowance and adds a fence/download; it is absent in clean timings.

The test-only resident bridge invokes `enqueue_prepared_cuda_fock`, not the
retained public host compatibility evaluator. Slurm 7045 passes all 16 tests,
covering independent libcint raw J, nonsymmetric RKS/UKS densities, spherical/
Cartesian representation, screening, changed/zero density, unchanged K,
bounded fallback and owner-frozen policy. Memcheck reports zero errors;
racecheck reports zero errors, warnings or hazards. Post-integration host
resource/compiler/ownership tests pass 126 focused and 1337 additional tests.

## Retention and reproduction

`samples/` retains every accepted population's original JSON bytes as
deterministic gzip, including physical forces and complete SCF histories.
`receipts/` retains immutable launchers, sanitizer logs, Slurm receipts and
source/library hashes. `retention.json` binds compressed and original hashes.
No binary, profiler archive, Release, release asset or tag is published.

Configure the CUDA sm120 build with ccache as both C++ and CUDA launchers and
build the native library plus PBE0 stationary manifests. Run the archived
launchers with their environment/source roots adapted to the checkout through
finite `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1
--cpus-per-task=8 --time=00:18:00`. Preserve Slurm's device visibility and never
schedule the maintenance node. The repository GPU test can be reproduced with
`GENERATIVEQC_TEST_MD_J_NORMAL=1 python -m pytest -q
tests/python/test_md_j_normal_cuda.py` inside that allocation; rerun under
memcheck and racecheck in separate finite allocations when needed.

The rejected angular-zero/one-only candidate and failed harness attempts remain
in ignored local evidence `.artifacts/pbe0-mdj-reuse-20261010/` and the original
n1 evidence owner `/data/jzzeng/pbe0-mdj-source-reuse-20261010-b74815dba/`.
They are not relabeled passes or mixed into the promoted population. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-10-md-j-reciprocal-source-reuse.md).
