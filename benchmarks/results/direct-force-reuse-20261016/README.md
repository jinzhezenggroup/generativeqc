# Exact Direct force recurrence/contraction reuse

This is the **historical frozen-composition qualification**, not a measurement
of subsequent master. Baseline `6d8192be8` combines master `9a92cad27`,
#2531 `01ed327f4` and #2532 `fb37f6d1c`. Candidate deployment `028249106`
contains the generated order-six preweighted-root and order-seven Hermite-axis
convolution changes. Subsequent test/evidence commits preserve native inputs.
Both changes are automatic within their admitted consumers on PR #2541.
No PR is merged by this campaign.

#2532 subsequently merged; PR #2541 now includes master snapshot `5348b9e66`
through `4bdffe3ce` and targets master. Fresh source-matched requalification is
in progress. The original full narrative and all original evidence remain
recoverable at immutable commit `6369de44fe1f30759b3d07c8f267f4a02d0d70dd`,
at this same repository path. This shortening removes duplicated prose only;
every retained primary/reference array and other evidence member is unchanged.

## Complete E+F

RTX 5090, CUDA 12.9.1, official Release/sm_120, strict FP64, portable=0,
fast_compile=0. Primary: canonical def2-mTZVPP, 544/1088 spherical AOs,
unchanged moving 48×16×32 Becke grid/atom, screening 1e-12 and original SCF
convergence. Controls: def2-SVP. Finite Slurm allocations preserve assigned
visibility. Every population retains cold + five warm + moved + five moved-warm
calls and the original 1800-second process limit. All 96 calls complete.

Times are seconds; warm entries are five-call medians, not normalized per build.

| Case | Phase | Baseline | Candidate |
| --- | --- | ---: | ---: |
| r²SCAN48 | cold | 76.637382 | 70.498551 |
| | warm | 21.557655 | 15.285856 |
| | moved | 60.232175 | 51.184662 |
| | moved-warm | 21.566031 | 15.357800 |
| r²SCAN96 | cold | 411.130928 | 386.092838 |
| | warm | 90.751786 | 65.555410 |
| | moved | 260.226115 | 234.750226 |
| | moved-warm | 90.919707 | 65.669 |
| PBE0 48 | cold/warm | 23.704/4.596 | 23.538/4.303 |
| | moved/moved-warm | 22.537/4.612 | 22.274/4.321 |
| HF48 | cold/warm | 10.901/0.742 | 10.923/0.743 |
| | moved/moved-warm | 7.198/0.742 | 7.196/0.744 |

96 warm/moved-warm improve 27.76%/27.77%; candidate remains ~2.47× historical
GPU4PySCF 26.59 s, not a fresh competitor measurement. Maximum independent
errors are 3.86535e-11 Eh and 6.01639e-11 Eh/Bohr, below 1e-8/1e-7 gates.

Actual cold/moved iterations/builds: r²SCAN48 19/19 both, moved 14/14→13/13;
r²SCAN96 24/24 and 13/13 both; PBE0 17/18 and 12/12 both; HF21/22 and16/17
both. Every warm call is1/1. The changed 48-atom moved trajectory is retained,
not normalized away. HF uses returned valid native physical-build counters.

## Separate diagnostic profile

| 96-atom device group | Baseline s | Candidate s |
| --- | ---: | ---: |
| Generated low J′ | 4.626643 | 4.626063 |
| Order six | 21.754343 | 17.441551 |
| Order seven | 28.034387 | 7.305988 |
| Order eight | 5.772215 | 5.771981 |

Exclusive J′ dispatch/wait:60.196200→35.154146 s. Intrusive complete diagnostic:
95.189178→70.208878 s, **not clean population timing**. Profile JSON reconciles
mutually exclusive SCF J/K, XC/AO/grid, other force/Becke, preparation,
validation and synchronization wall owners. Device times are not added to wall.

Executed work is unchanged: order-six primitive products2,400,504,912,
components1,121,296,224; order-seven primitive/Coulomb preparations1,250,544,
bra/ket preparations2,606,140/3,147,016 and contractions1,937,135,584;
order-eight primitive/Coulomb preparations95,976. Unexposed counters stay null.
Entered source algebra over28 unweighted spatial boxes has products16573→15984
and additions2300→2956. One response/all648 dddp components has products
194184→62244, additions36288→27504, coefficient reads127008→13608 and root
reads36288→18432. These are not machine instructions/FLOPs or endpoint totals.
Nsight Compute fails ERR_NVGPUCTRPERM. Order-six registers/CTA remain255/128;
order-seven registers222→214, CTA256, static/dynamic shared416/42984 bytes.
Achieved occupancy and full process peak memory are not measured.

## Validation and replay

318 focused host checks, six actual-linked independent Libcint GPU cases
(RKS/UKS, Cartesian/spherical, separate/combined derivatives), additional AD/
symmetry checks and zero-error memcheck/synccheck qualify the frozen source.
Compiler structure:541 modules/0 errors. Overlapping suites are not summed.
An accurate cross-root DAG regresses its probe~0.428→0.443 ms despite fewer
registers; its patch is retained, not enabled. Prior negative scheduling/page
experiments are not restored. This is scoped numerical acceptance and endpoint
evidence, not formal globally interleaved performance promotion.

`numeric-arrays.npy.xz` retains **every primary E/F and independent geometry
oracle losslessly** in standard NumPy/XZ. `numeric-index.json.xz` retains shapes,
offsets, protocols, per-call timings/counts/gates and reference metadata.
`python verify-arrays.py` recomputes all96 gates; `--restore-references`
reconstructs the four original oracle files. Decompress `def2-svp.json.gz`.
Other JSON/XZ records retain ledgers, provenance, work, resources and validation.
Detailed native histories, inventories and build/Nsight logs remain hash-bound
under ignored `.artifacts/direct-reuse/` and remote
`/data/jzzeng/gqc-direct-reuse-20261016` on n1/n4/n5. No Release/tag/external
archive is created and no storage cap is increased.

`build.sh` takes `SOURCE_ROOT` and uses verified compiler caching. Place each
source export under `ARM/source`, built libraries/manifests under `ARM/runtime`,
and references/basis under the campaign root. `run-gpu.sh.gz` accepts
`GQC_DIRECT_REUSE_ROOT`; included Python wrappers document cluster glibc/CUDA
paths. Adapt paths, not scientific conditions. Never override device visibility.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=28G --time=00:35:00 \
  bash run-gpu.sh candidate r2scan-3c 96 endpoint
```

Use `profile` for intrusive capture, `tests` for Libcint checks, and `counted`
for HF work counters. The n1/n5 oracle-python shim contains only dftd4/cffi/
pycparser, not shadow production GQC/PySCF work. Next: compiler-owned primitive
preparation/derivative contraction reuse, not CTA/page tuning.
