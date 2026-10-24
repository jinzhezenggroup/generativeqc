# Automatic occupied-XC on current master

Frozen master: `72e0a7e5a011592428b5625acdfa6f19168645ac`.
RTX 5090 on node1, 2026-10-24. The candidate requests the existing authenticated
occupied-XC route when its selector is unset. The same frame, rank, precision,
method, lifetime and fallback checks remain; explicit `0` restores matrix
contraction. No kernel mathematics, allocation or scientific setting changes.

Three rotating fresh-process rounds compare separate pristine-master and
candidate libraries with stock GPU4PySCF 1.8.1. Cold includes construction,
preparation, SCF, analytic forces and host export; one frozen warm call follows.
Imports and prior AOT compilation are outside the timer. All 54 calls remain.
These are default-grid endpoint measurements, not a globally optimized ranking.

| Direct Standard / 96 atoms | Master cold (s) | Candidate cold (s) | GPU4PySCF cold (s) | Candidate gap |
| --- | ---: | ---: | ---: | ---: |
| PBE | 17.831442 | 17.987506 | 16.179199 | +11.18% |
| PBE0 | 21.719859 | 21.790777 | 20.328221 | +7.19% |
| r2SCAN-3c | 42.376693 | 40.730322 | 32.448150 | +25.52% |

r2SCAN-3c cold improves **3.89%**, exceeding its predeclared 2% variability
floor. PBE/PBE0 changes are below the floor, consistent with their unchanged
one-jet density route. Native cold SCF/Fock counts remain 11/9/10; warm counts
remain one. GPU4PySCF cycles are 11/9/11; equivalent Fock counts are not exported.
The native semilocal grids contain 1,078,336 points. Each engine keeps its
original grids and solver settings, recorded in every receipt.

Warm medians (master/candidate/GPU4PySCF seconds) are 5.325965/5.339341/5.627152,
6.836680/6.847178/6.820197 and 15.820725/15.811506/14.966111. All native changes
are below the same floor; no warm speedup is claimed. The floor is
`max(2%, 2 * (MAD_master / median_master + MAD_candidate / median_candidate))`.
No observation, including the first 7.058909-second PBE0 candidate warm call,
is removed or replaced.

## Mechanism and resource work

A separate intrusive 3c cold-plus-warm trace reports eight accepted occupied
frames (generations 2–9), with the first cold, held final confirmation and warm
using matrix products. There are 2,112 factor-specialized density launches and
792 matrix launches versus 2,904 matrix launches in the rollback. The combined
density-product device time drops 4.415185 → 2.748385 seconds; total XC intervals
are 11.948984 → 10.254781 seconds. These traces never enter clean timing medians.

AO-potential contraction remains about 4.12 device seconds, and the two leading
integral-force kernel families total about 12.73 seconds across cold+warm.
These overlapping/profiled intervals are not an additive explanation of the
inter-engine endpoint gap. Per-tile dense versus factor projection changes
`4*p*a*a` to `4*p*a*min(a,rank)` on accepted frames; there is no measured hardware
FLOP counter or per-tile factor census in these receipts.

Force work is unchanged: 4,213 maps, 11,732,295,680 discovery jet values,
327,106,432 selected point-AO visits and 111,889,789,952 AO-square point work.
Cold retained maps use 10,224,728 bytes; warm performs no map discovery.
The 3c force's additional device/host bounds remain 480,269,448/263,387,808 bytes
within 536,870,912/268,435,456-byte allowances. The small native WB force receipt
reports bounds but no corresponding ceiling; the rechecker keeps that missing
field explicit rather than claiming a resource gate for it.

## Qualification and limitations

All **166 retained static E/F calls** and **three original/warm/moved calls**
pass independent gates. Static limits are 1e-5 Eh/atom, 1e-4 maximum force and
3e-5 RMS force in Eh/Bohr, using the audited dense references, grid-stability
and CPU/GPU spots in `../readme-final-20261022/`. Maxima over the whole retained
population are 3.0435e-6, 5.6422e-5 and 2.5239e-5 respectively. The moved reference
is a new tighter full-density GPU4PySCF solve at atom 1 z +0.02 Bohr, after
native timing. All three calls pass 2.5e-6/2.5e-5/7.5e-6 limits; the moved max
force error is 1.5990e-6 Eh/Bohr. This is matched-grid displaced validation,
not a new displaced-grid convergence study.

Held-out pairs cover r2SCAN Direct3/96, r2SCAN-3c Direct48 and DF96, M06-2X
Direct48, PBE0 and 3c Fast96, and WB97M-V Direct3. These single pairs establish
numerical/regression coverage, not repeated-population speedup claims.

Small r2SCAN warm timing remains about 0.083 seconds. A separate three-pair
follow-up initially flags a 3.81% cold slowdown (1.422194 → 1.476425 s, 2.85%
floor). All observations remain. A five-pair same-image policy ablation spans
1.40–1.74 seconds and does not reproduce a policy regression. A new five-pair
comparison of both original images gives 1.492828 → 1.460484 seconds cold,
below its 4.04% gain floor, and 0.082877 → 0.082904 seconds warm. These three
cohorts are separately predeclared and never pooled. No small-system cold
speedup or universal nonregression guarantee is claimed.

433 host tests, 18 independent GPU feature/owner tests, three RKS/UKS force and
reconverged finite-difference tests, and two new public default/rollback/geometry
tests pass. The latter exercise 24 complete SCF solves against independent CPU
references. Memcheck, synccheck and racecheck each pass eight owner cases with
zero errors/hazards. The first public-policy assertion incorrectly required all
warm UKS iterations to lack a frame; it is retained alongside the corrected
check of the first held-density iteration and the successful rerun.

## Provenance and reproduction

Main Slurm allocation 13840 assigns GPU visibility `1`, eight bound physical
CPU cores, and one OMP/OpenBLAS/MKL thread. Other jobs use other GPUs on the
shared host. All GPU tests/profiling use finite Slurm and preserve visibility.
CUDA is 12.9.1, Python 3.13.9, GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0.
Verified sccache 0.16.0 wraps native/JIT builds; host checks use ccache 4.5.1.
Cache statistics are shared-server totals, not a private build hit-rate claim.

Master library SHA256:
`cd831a0c57ce8468bfc1326ce9c44639e8079076bc747e9f96feac3d676ea348`.
Candidate library SHA256:
`498e00b0784020f4bffe6dee6d697315390b69ff2055f4c799d03639a52b90ae`.
All measured native/runtime/compiler source bytes match the submitted change;
notes, evidence and the supplemental policy test were added after compilation.
`receipts.json.xz` retains complete numerical arrays, all populations, work,
source manifests, original hashes, scripts, protocols, validation logs and
selected trace records. The lossless raw archive additionally retains full
source tarballs, build commands/cache logs, profiles and failure outputs at
`node1:/data/jzzeng/direct-dft-occupied-default-20261024/` and ignored local
artifacts. Frozen libraries remain at that root and the referenced master root.
No Release, release asset or tag is created.

Recompute every gate and statistic, including under optimized Python:

```bash
PYTHONPATH=python:. python benchmarks/results/occupied-xc-default-20261024/reproduce.py
PYTHONPATH=python:. python -O benchmarks/results/occupied-xc-default-20261024/reproduce.py
```

The new [decision note](../../../.agents/notes/implemented/performance/2026-10-24-automatic-occupied-xc.md)
records the fallback/lifetime policy. The
[rejected follow-ups](../../../.agents/notes/rejected/2026-10-24-direct-dft-cold-followups.md)
retain the losing AO-chunk schedule and an older candidate; none of their
measurements is pooled into this promotion.
