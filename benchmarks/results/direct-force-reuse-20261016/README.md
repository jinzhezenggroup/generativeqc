# Source-matched exact Direct force arithmetic reuse

Clean exports: incumbent `4bdffe3ce` (PR #2541: preweighted order-six roots plus
order-seven Hermite convolution) and response `00fced59f` (stacked constant-
Jacobian factoring). #2531/#2532 are merged in their baseline. The complete
preceding master `5348b9e66` versus incumbent qualification is retained at
immutable commit `4eb9fd859a69a31fa21c5fbbe9b13f7d36a9d90a`, same path. Subsequent master
movement is not relabeled as measured. Each native inventory contains1753
inputs; exact loaded identities, image hashes and environments are retained.

This incremental two-arm packet supersedes its predecessor **at this path**;
the original complete bundle remains at immutable Git commit
`6369de44fe1f30759b3d07c8f267f4a02d0d70dd`, same path. No old measurement is
reclassified as current-source evidence, no unrelated history is removed,
and no storage budget/Release/tag/external archive is created or changed.

## Complete E+F

RTX5090, CUDA12.9.1, official Release/sm_120, strict FP64, portable=0,
fast_compile=0. Canonical def2-mTZVPP,544/1088 spherical AOs, unchanged moving
48×16×32 Becke grid peratom, original screening1e-12/SCF convergence.
HF/PBE0 controls use def2-SVP. Every GPU operation uses finite Slurm allocations
and assigned visibility. Primary96 runs on the same n4 device; primary48 and
diagnostic profiles use separate matched n1 allocations. No concurrent jobs
share an assigned GPU. Every population retains cold, five warm, moved and
five moved-warm calls under the unchanged1800-second process timeout.

Seconds; warm columns are five-call medians. No iteration normalization:

| Case | Phase | Incumbent s | Response s |
| --- | --- | ---: | ---: |
| r2scan-3c-48 | cold | 69.934962 | 69.224673 |
| r2scan-3c-48 | warm | 15.226590 | 14.368431 |
| r2scan-3c-48 | moved | 50.849639 | 49.897404 |
| r2scan-3c-48 | moved-warm | 15.253462 | 14.339476 |
| r2scan-3c-96 | cold | 386.116851 | 382.483718 |
| r2scan-3c-96 | warm | 65.477918 | 62.204506 |
| r2scan-3c-96 | moved | 234.961981 | 231.331466 |
| r2scan-3c-96 | moved-warm | 65.707564 | 62.139573 |
| pbe0-48 | cold | 23.372648 | 24.088161 |
| pbe0-48 | warm | 4.258826 | 4.236261 |
| pbe0-48 | moved | 22.088725 | 22.191909 |
| pbe0-48 | moved-warm | 4.257219 | 4.217512 |
| hf-48 | cold | 10.937647 | 11.003730 |
| hf-48 | warm | 0.741202 | 0.745075 |
| hf-48 | moved | 7.204627 | 7.229003 |
| hf-48 | moved-warm | 0.742078 | 0.745135 |

response/incumbent96 warm/moved-warm reductions: 5.00%/5.43%.
Response remains 2.34× historical GPU4PySCF26.59s; no fresh competitor measurement.

Actual cold/moved iteration/build counts (all warm calls are1/1):

| Case | Incumbent | Response |
| --- | --- | --- |
| r2scan-3c-48 | 19/19; 13/13 | 19/19; 13/13 |
| r2scan-3c-96 | 24/24; 13/13 | 24/24; 13/13 |
| pbe0-48 | 17/18; 12/12 | 17/18; 12/12 |
| hf-48 | 21/22; 16/17 | 21/22; 16/17 |

All 96 primary calls complete and pass independent gates1e-8Eh/1e-7Eh/Bohr. Maximum errors: 3.8653525e-11Eh and 6.0142558e-11Eh/Bohr. Every trajectory/count remains explicit, including any moved convergence difference.

## Diagnostic profile and executed work

Separate intrusive warm captures, not substituted for clean endpoint timing:

| Device group | Incumbent s | Response s |
| --- | ---: | ---: |
| direct_j | 8.353921 | 8.342424 |
| cuda_ks_xc_shared | 4.550966 | 4.538433 |
| direct_force_generated_low | 4.678237 | 4.668891 |
| direct_force_order6_spatial | 17.549077 | 13.914502 |
| direct_force_order7_cooperative | 7.375439 | 7.356338 |
| direct_force_order8_materialized | 5.824592 | 5.810460 |

`profile-summary.json.xz` reconciles mutually exclusive host owners and separate
device windows: SCF J/K, XC/AO/grid, order6/7/8 J′, other force/Becke response,
final snapshot/decode/validation, preparation and synchronization. Device
kernel times are never added to their parent host waits. Resource rows retain
registers, launch bounds and shared-memory limits; no achieved occupancy,
hardware instruction/FLOP count or full process peak-memory claim is made.

incumbent entered order6/7/8 primitive products: 2,400,504,912/1,250,544/95,976.
response entered order6/7/8 primitive products: 2,400,504,912/1,250,544/95,976.

The additional response optimization sums two pair-coefficient responses and
one root response before projecting to nuclear centers; nine bounded sums
replace nine center sums, with no extra storage/barrier. Other orders and the
nonspatial fallback retain their original arithmetic. Exact J/K, RKS/UKS,
representations, screening, signs, symmetry and fourth-center translation are
preserved. Separate instrumented primitive launches count constant-Jacobian
products75,497,472→1,179,648 (full),68,633,856→1,072,404 (partial), and
2,359,296→36,864 (singleton), for both4/2 and3/3 pair partitions. These are
entered **source operation** counts, not machine instructions, whole-kernel
FLOPs or endpoint totals. Counted launches are excluded from timing samples.

## Validation and rejected trials

Base synchronization:588 focused host tests. Response branch:890 in one
focused suite, including300 independent correlated-Gaussian raised/lowered
quadrature checks and common-center full/LR finite differences. Overlapping
suites are not summed. Actual-linked independent Libcint checks cover both
spins, Cartesian/spherical and separate/combined J′/K′. Response probe memcheck
and synccheck report zero errors. Compiler structure:541 modules/0 errors.

Warp-local Boys/prefactor reuse cuts instrumented radial preparations32-fold
but regresses4/2 median1.218603→1.239000ms and3/3 1.064220→1.067742ms.
Subgroup bra-term sharing gives1.206765→1.218532ms and1.065542→1.063032ms:
no consistent useful gain. Both have zero derivative differences under full,
partial/singleton masks; neither is enabled. Their AO layouts differ, so do
not compare across trials. Prior rejected scalar-DAG/cooperative-six/page
experiments remain rejected. No unsupported hardware-stall diagnosis is made;
historical Nsight Compute reports ERR_NVGPUCTRPERM.

## Retention and replay

`primitive-trials.json.xz` retains the selected generator/probe UTF-8 sources,
raw timing/count/resource/sanitizer receipts and replay instructions as ordinary
lossless JSON text, with per-source SHA256. Decompress it and write the `sources`
entries to their relative filenames to replay; no custom numeric codec is used.

Every primary E/F array and both independent geometry oracles are retained
losslessly in standard `numeric-arrays.npy.xz`, with protocols, offsets,
per-call timings/counts/gates and oracle metadata in `numeric-index.json.xz`.
`python verify-arrays.py` recomputes every gate; `--restore-references`
reconstructs independent oracle inputs. Other JSON uses ordinary XZ. Detailed
native histories, source inventories, build/cache/probe/Nsight records remain
hash-bound under ignored `.artifacts/direct-reuse-next/` and remote
`/data/jzzeng/gqc-direct-next-20261016` on n1/n4; original frozen receipts remain
under `.artifacts/direct-reuse/`. This is scoped numerical acceptance and
matched endpoint evidence, not formal global interleaved performance promotion.

Build each exact source export with `SOURCE_ROOT=... bash build.sh`, using
verified ccache4.5.1 (existing shared cache retained; CXX/CUDA launchers and
checkout-root CCACHE_BASEDIR). Place source under `ARM/source`, libraries and
stationary manifests under `ARM/runtime`; restore references under the root,
decompress `def2-svp.json.gz`, and adapt the cluster Python wrappers documented
in the immutable frozen bundle. The HF wrapper retains already-returned valid
strict/mixed/post-SCF physical-build counts after timing, not estimates.

```bash
GQC_DIRECT_REUSE_ROOT=/path/to/campaign srun --partition=main \
  --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=28G \
  --time=00:35:00 bash run-gpu.sh incumbent r2scan-3c 96 endpoint
```

Use `profile` for diagnostics, `tests` for Libcint gates and `counted` for HF.
Remove any local `defer-profiles` scheduling marker to run profiles on that
node. Both optimizations are automatic in their admitted branch paths; no PR
has been merged by this agent. Remaining work: generated axis-factorized
coefficient/response contraction and primitive preparation reuse, not CTA/page
tuning. Historical GPU4PySCF26.59s is a target, not a newly measured baseline.
