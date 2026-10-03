# Matched ωB97M-V energy and analytic-force endpoints

The opt-in joint SCF/force active-AO integration has lower complete warm latency
than its paired GPU4PySCF reference at 24 and 48 atoms. This is a measured
candidate snapshot, not a claim about the master default or either engine's
default quadrature. The 96-atom comparison remains pending.

| Atoms / spherical AOs | Native dense warm | Native joint warm | Joint-paired reference warm | Joint / reference |
| --- | ---: | ---: | ---: | ---: |
| 24 / 192 | 27.110692 s | 26.330918 s | 27.397887 s | 0.961057 |
| 48 / 384 | 106.139498 s | 97.112717 s | 103.042334 s | 0.942455 |

All three warm samples enter each median. The dense variants have their own
interleaved reference samples; those medians are 27.402939 and 103.253732 s.
Native joint 48-atom samples are 97.129844, 97.112717 and 97.014143 s; reference
samples are 103.055296, 103.042334 and 103.035995 s. Every warm/priming result
converges in one iteration, using its own engine's frozen post-cold density.
There is no cross-engine density injection or iteration-normalized timing.

## Cold startup and work

Cold includes calculator/engine construction, synchronized preparation and the
first complete execution. Imports and separate library probes are outside this
boundary; this does not assert an empty compiler/filesystem cache.

| Atoms | Native dense complete cold | Native joint complete cold | Joint-paired reference complete cold |
| --- | ---: | ---: | ---: |
| 24 | 249.833558 s | 240.398506 s | 116.268821 s |
| 48 | 1113.882289 s | 962.211204 s | 474.695895 s |

Cold remains substantially slower than reference. Native cold submits 18/21 XC
evaluations at 24/48 atoms; warm submits one. SCF discovery is preparation work,
not a cost added again to warm calls. At 48 atoms it takes 0.649430 s, retains
4608 tiles (436 empty), has an active-AO sum of 621388 and reports 14194184
numeric peak host bytes. SCF point/AO-square work is 14.5202% of dense; force
work is 18.7396%. These counters describe contraction work, not a speedup model.
The force map has no warm rediscovery. Dense-disabled force cache counters
are absent in the original report and are not replaced with invented zeros.

## Protocol and numerical gates

- Water-cluster prefixes from `benchmarks.readme_hf_scaling.scaling_cases`,
  WB97M-V/RKS, complete spherical def2-SVP. Both engines use the same unpruned
  48 radial × 16 polar × 32 azimuthal grid per atom: 589824/1179648 points.
  Semilocal and VV10 grids are the same full grid, with equal-radius Becke
  partitioning and analytic grid/weight response. Shared VV10 density threshold
  is 1e-8. This is not a comparison of the engines' respective default grids.
- Native energy/density/screening controls are 1e-11/1e-9/1e-12, maximum 180
  iterations. GPU4PySCF uses 1e-11 energy, 1e-8 orbital-gradient and 1e-14 direct
  screening tolerances, with full-density Fock builds. Different stopping
  criteria are retained, not presented as identical internal SCF algorithms.
- Explicit native SCF `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`; force caller uses
  `active_ao_cutoff=1e-16`, `active_ao_cache_bytes=64<<20`. Dense control disables
  both maps. SCF tiles remain 256 points; the composite force planner chooses
  its budgeted tiles. The fixed 64 MiB SCF host-map cap is experimental and
  does not establish public constrained-host-budget support.
- Slurm n1 jobs 5577/5576 run each dense/joint comparison sequentially in one
  RTX 5090 allocation, preserving scheduler visibility. CUDA 12.9.1, CuPy
  13.6.0, NumPy 2.4.6, PySCF 2.14.0 and GPU4PySCF 1.8.1; eight OpenMP,
  OpenBLAS and MKL threads. Raw reports retain device and toolchain metadata.
- Cold, priming and all three warm pairs must each pass 1e-8 Eh / 1e-7
  Eh/Bohr, finite-value, shape and convergence checks. All 20 native/reference
  pairs across both sizes and controls pass. Maximum joint errors are below
  1.001e-11 Eh / 6.307e-10 Eh/Bohr. Every reference XC component reports
  `on_gpu=true`. No CPU XC fallback is used to explain the comparison.
- Independent complete WB97M-V qualification includes changed geometry,
  reconverged displaced energies and stale-snapshot isolation. Slurm 5575
  passes seven cases with joint maps and seven with force-cache allowance zero;
  each group verifies 66 successful native calls and 467 XC submissions.
  These are numerical qualification, not timed large changed-geometry results.

## Source, retained records and reproduction

Measured source is [0132d7584](https://github.com/jinzhezenggroup/generativeqc/commit/0132d75844c13d90bd04cb250e3aafae12d2a3dd),
an integration based on master d442177a6. Its 1351 build-manifest inputs match
the built source c06859af9. It includes the pending SCF/force AO stack and
qualified force improvements; it is broader than standalone PRs #1778/#1780.
The native/library identity and source archive verification are recorded in
[manifest.json](manifest.json), independently of the archived runner's unavailable
Git probe. Library SHA-256 is
`b82e468613c5c90e076aa10082b2389c8dda74335f625a7b77641c1c085ef3f9`.
Default screening is not promoted by this documentation.

The four `.json.gz` files preserve every original report byte, including all
energy/force arrays, timings, convergence, preparation, XC backend and AO work.
The manifest records stored and decoded SHA-256 digests and sizes. Verify them
without a GPU:

```bash
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 24
python benchmarks/results/wb97mv-active-ao-20261003/verify.py 48
```

Build the measured checkout in Release/sm_120 with CUDA 12.9.1 and explicit
CXX/CUDA `ccache` launchers after verifying `ccache --version`. Retain source
identity, compiler commands and pre/post cache statistics. No binary is hosted
in this evidence directory. The replay patch reconstructs the exact three
measurement scripts from the measured checkout's `benchmarks/readme_wb97mv.py`:

```bash
export ACTIVE_AO_ROOT=/data/jzzeng/wb97mv-replay
mkdir -p "$ACTIVE_AO_ROOT"
cp benchmarks/readme_wb97mv.py "$ACTIVE_AO_ROOT/complete-cold-benchmark.py"
patch -p1 -d "$ACTIVE_AO_ROOT" < /path/to/this/evidence/reproduce.patch
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export ACTIVE_AO_MODE=sparse GENERATIVEQC_CUDA_KS_ACTIVE_AO=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=01:00:00 \
  python "$ACTIVE_AO_ROOT/matched-probe.py" --atoms 48 --repeats 3 \
  --grid 48 16 32 --output "$ACTIVE_AO_ROOT/matched48-sparse.json"
```

Use `ACTIVE_AO_MODE=dense GENERATIVEQC_CUDA_KS_ACTIVE_AO=0` for the dense
control, and 24 for the smaller case. Supply the recorded CUDA runtime and
GPU4PySCF environment through the cluster's normal module/library setup.
Preserve Slurm's device visibility. The replay patch's source and reconstructed
script digests are retained; cold preparation is timed explicitly, and all
diagnostic hooks run after endpoint timers stop. Full build/scheduler/profiler
debug material stays under ignored `.artifacts/scf-active-ao/` in the measured
worktree; these retained reports and scripts are sufficient to audit the table.
