# Full-range RAM ERI reuse qualification

This extends the bounded FP64 host owner to full-range J(total)/K(spin), with
separate `GENERATIVEQC_KS_HOST_*` controls. RAM backing defaults automatically
only for an admitted cold strict source. Existing incremental/generated
consumer admission stays authoritative.

## Complete consumer evidence

PBE0/def2-SVP spherical, strict FP64, exact primary J/K, default grids, complete
cold preparation + SCF + analytic forces + host export. Both paired cohorts use
the same library (`2ca675a2c82a513d36f9a8fa553b8a3b9858bad27da2d3e0613ae16cd1dd6950`),
CUDA 12.9.1 Release sm120, scalar CPU linear algebra, RTX 5090 on n5, six CPUs,
96 GiB RAM, and one BLAS/OpenMP thread. GPU ceiling is 2 GiB with the identical
explicit GPU bucket mask `268435008`; RAM ceiling is 32 GiB. The declared
validation-GEMM force-response route is enabled in all arms.

The source-eligible cohort explicitly disables incremental Fock in both arms
and forces the existing resident source. Job 1855 has a finite 45-minute limit.
Fresh processes run GPU, host, host, GPU; separate profiled primers are excluded.

| 96 atoms | GPU partial cold (s) | GPU + RAM cold (s) |
| --- | ---: | ---: |
| First observation | 94.8484 | 39.1250 |
| Second observation | 94.8885 | 39.0807 |
| Median | 94.8685 | 39.1029 |

The reduction is 58.78%, at identical nine SCF iterations and ten cold Fock
builds. Every cached action covers 1,873,221,716 canonical values: 264,066,624
on GPU and 1,609,155,092 in RAM. H2D is 12,873,240,736 bytes/action and omitted
recurrences are zero. The profile contains nine cached cold SCF actions, one
strict Direct action and two cached warm actions. Forced warm calls rebuild
sources and take about 23.22 s with RAM versus 24.96 s with GPU-only; this is
not the automatic warm policy.

**The natural default stays faster.** Job 1854 leaves incremental Fock enabled:
96-atom default cold is 21.8126/21.8191 s, and warm is 6.8490/6.8568 s, with
nine cold Focks and one warm Fock. Forced GPU and RAM arms have the same cold
path (21.78–21.85 s); their source is only consumed on warm replay. A valid
progress journal exposes this distinction. Those cold times do not measure
RAM reuse. At 48 atoms, natural default cold/warm is 9.5022/2.4426 s; forcing
residency gives 9.8298/5.6043 s. Do not broaden default source admission merely
because a RAM cache fits. The extension helps an existing source-eligible
schedule and does not replace the faster default incremental schedule.

These are two-pair screening observations, not a universal crossover model.
Public geometries and complete work/settings remain in all raw receipts.

## Source, numerical and lifecycle validation

The natural cohort measures commit `61ff39eab2dab2c8e465ceaac3a2381a2b424ee9`;
the controlled cohort measures `07e25bade33c782152ea0dbc441230b97efe5bdf`.
They differ only by a benchmark option that explicitly records incremental
disablement. Both source archives reproduce the same native identity and
library hash. Their producer revisions are retained independently.

All 28 complete E+F calls (including separate primers) pass independent
PySCF/GPU4PySCF gates: E/atom 1e-5 Eh, force max 1e-4 and RMS 3e-5 Eh/bohr,
with dense-grid stability and CPU/GPU spot checks. Nine scheduled GPU tests
cover four raw signed/zero RKS/UKS source cases, three complete PBE0 RKS/UKS
and B3LYP endpoint cases, and two incumbent phase regressions. Each endpoint
case checks seven policies and cold/warm/moved/moved-warm against independent
PySCF (84 complete calls). Compute Sanitizer reports zero errors/leaked bytes.

Earlier qualification attempts exposed missing packaged force artifacts, a
PySCF UKS DIIS convergence problem (resolved by independent Newton), an obsolete
optional-admission assertion also failing on the parent library, and dynamic
progress-name lifetime. These failed attempts remain in ignored artifacts;
they are not counted as successful endpoint evidence. The final library owns
progress names and the retained journals parse correctly.

```bash
PYTHONPATH=python:. python benchmarks/results/host-full-eri-20261025/reproduce.py
```

The compressed packet retains raw endpoint receipts, archive/build bindings,
all completed source-action counters, selected preparation/retirement events,
executed scripts, compiler-cache statistics and JUnit/sanitizer results. Raw
large journals, source archives and binaries remain in ignored artifacts.
