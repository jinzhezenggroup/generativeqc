# Bounded temporary-file ERI spill qualification

The qualified automatic order for an admitted cold strict **long-range** source
is GPU, pageable RAM, a local temporary file, then exact Direct recomputation.
Full-range RAM remains automatic for admitted sources, but full-range disk
replay is explicit-only: the complete NVMe result below is slower than Direct.
No faster incremental/generated consumer is displaced by storage admission.

## Source-matched complete endpoints

The measured implementation is commit
`4d65aec4600ea864c80c12c6b0b3b00ec3e40394`. The packet binds its source archive,
native inventory, library and benchmark driver. All timed arms use that same
binary; separate profiled primers are excluded from the timing comparison.
The final policy commit `b01be68a0ffb5cce2d5c48690e95652ab9751593` restricts
automatic spill to LR. Its only native-source change is the KS policy owner;
the tested file provider, recurrence and contraction are unchanged. The final
policy has separate GPU endpoint validation, without relabeling old timings.

Release CUDA 12.9.1 sm120, scalar CPU linear algebra, six allocated CPUs,
96 GiB job RAM and one BLAS/OpenMP thread. GPU work uses finite Slurm allocations
with assigned visibility. The driver includes constructor/preparation, SCF,
analytic forces and host export; its declared validation-GEMM force-response
route is enabled throughout. Historical OpenBLAS results are not pooled here.

### 108-atom WB97M-V/def2-SVP spherical DF

RTX 5090 on n4, Slurm job 1341, 90-minute limit, node-local `/tmp` on ext4 and
a KIOXIA EXCERIA G2 NVMe. GPU ceiling is 16 GiB with a common explicit bucket
mask; the actual admitted GPU values occupy 9,359,989,648 bytes. RAM-only allows
32 GiB; the mixed arm limits RAM backing to 2 GiB. Disk allows at most 64 GiB.
Fresh processes run GPU, RAM, disk, then mixed. Each has 28 SCF iterations and
29 cold Focks; frozen-density warm has one Fock.

| Backing for omitted GPU values | Cold E+F (s) | Warm E+F (s) |
| --- | ---: | ---: |
| Direct recomputation | 681.1930 | 55.9931 |
| RAM | 363.2495 | 56.0020 |
| Temporary NVMe file | 493.5743 | 55.9656 |
| RAM limited to 2 GiB, then NVMe | 462.3187 | 55.9813 |

Disk-only reduces cold time by 27.54%; mixed backing reduces it by 32.13%.
RAM remains preferable when it fits. These are single screening observations
per arm, not a confidence interval or a universal disk crossover model.

Each cached action visits 2,156,803,082 canonical values. GPU covers
1,169,998,706; disk-only retains the other 986,804,376 values (7,894,435,008
bytes). The mixed arm retains 1,598,317,568 value bytes in RAM and
6,296,117,440 on disk, using 1,874,486,872 bytes of combined RAM/device staging
and host values. Both cover every omitted bucket and perform zero omitted
recurrences. H2D remains 7,894,435,008 bytes/action in both streamed arms.

The disk primer records about 221.03 GB of process physical reads versus
221.04 GB of logical ERI reads across 28 cached SCF actions. This exercised
physical storage, not only the OS page cache. Process counters include other
input/library traffic and are not per-file block-device attribution. Traced
action durations include file I/O, staging, waits and contraction; they are
not pure disk or PCIe bandwidth. Profiling adds synchronization and is excluded
from formal endpoint timing. A separate small automatic-source control admits
GPU values and reports zero disk bytes.

### Negative full-range control

RTX 5090 on n5, job 1857 with a 30-minute limit, the same local NVMe model and
binary. PBE0/def2-SVP spherical, exact full-range J/K, strict FP64, default
grids, incremental Fock disabled in both arms, forced existing source,
2 GiB GPU ceiling and common bucket mask `268435008`.

| Omitted full-range values | Cold E+F (s) | Warm E+F (s) |
| --- | ---: | ---: |
| Direct recomputation | 94.7835 | 24.9705 |
| Temporary NVMe file | 143.8260 | 51.4234 |

Both take nine SCF iterations and ten cold Focks; forced warm takes two Focks
and rebuilds the source. Disk is 51.74% slower cold. Its 12,873,240,736 cached
bytes/action do avoid recurrence, but movement costs more. The final default
therefore retains full-range Direct when RAM is insufficient. Explicit disk
controls preserve this experiment; RAM and the faster natural incremental
path remain independent. The [RAM cohort](../host-full-eri-20261025/README.md)
retains the natural default's approximately 21.82 s cold control.

## Accuracy, failure and lifetime evidence

All 20 complete E+F benchmark calls, including separate primers and the small
control, pass independent PySCF/GPU4PySCF gates: E/atom 1e-5 Eh, force max
1e-4 and RMS 3e-5 Eh/bohr. Dense-grid stability, CPU/GPU spot agreement and
original oracle producer identities are rechecked offline.

On n2's RTX PRO 6000 Blackwell, the measured implementation passes 23 GPU
cases: 12 independent full/LR disk/tiered signed-density cases; eight LR
default/refusal/lifetime cases; three independent PBE0 RKS/UKS and B3LYP cases
across nine policies. These endpoint cases contain 172 complete cold/warm/moved
calls. Two Compute Sanitizer runs report zero errors and leaked bytes. The
final default restriction passes all 11 endpoint cases again. This device's
test durations are not pooled with RTX 5090 performance.

Compiled local-file tests check physical reservation, bounds, actual truncated
reads, immediate unlink and descriptor retirement. Native GPU probes truncate
the real file, require replay to fail without certifying a partial result, and
independently check a complete explicit Direct retry. Fifteen final file,
capacity and trace tests pass, alongside the earlier structure/default checks.
Default LR spill needs no disk opt-in when RAM is exhausted. Full-range unset
disk now checks the retained Direct fallback.

Earlier n2 compiler/link setup failures and a missing KS-to-owner disk argument
were caught before qualification. The original n4 attempt was stopped when
its trace showed zero disk consumption. Those failures, source identities and
the aborted receipt are retained separately and supply no performance result.

## Reproduction and storage contract

```bash
PYTHONPATH=python:. python benchmarks/results/disk-eri-20261025/reproduce.py
```

`measurements.json.xz` retains original complete receipts, both source/build
bindings, every completed ERI source action, selected preparation/retirement
events, executed scripts, physical-I/O observations, JUnit/sanitizer output and
failed-attempt dispositions. Raw large journals and binaries remain ignored.

The file is private, immediately unlinked and really preallocated. Default LR
capacity is bounded by 64 GiB and one quarter of free local space; tmpfs/ramfs
and common network filesystems are refused. Staging still fits complete
numeric/device and available-job-RAM bounds. Setup I/O refusal preserves
GPU/RAM/Direct; replay I/O failure fails the action. Geometry/precision ownership
retires the descriptor. Flush/page-eviction calls are OS hints, not a physical
I/O guarantee. No persistent, shared, or cross-geometry cache is created.
