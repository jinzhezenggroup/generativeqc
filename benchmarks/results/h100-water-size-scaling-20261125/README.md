# Cold HF and DFT size scaling on H100

This probe for [#3170](https://github.com/jinzhezenggroup/generativeqc/issues/3170)
uses GenerativeQC revision `22f72d0692a760c2892cd7791022584d9ae1c3e5` and
GPU4PySCF 1.9.0. It measures complete cold energy and analytic forces on
96, 144, 192, 240 and 300 atoms.

[summary.csv](summary.csv) reports medians, sampled memory and matched
repeats. [gates.json](gates.json) contains the sample-level oracle errors.

The sweep recorded 80 prewarms, 168 complete formal endpoints and 60
paired repeats. GenerativeQC completed 20 of 40 prewarms, with 17 timeouts
and three SCF nonconvergences. GPU4PySCF completed 36 of 40, with two
allocation failures and two illegal CUDA addresses. Every arm that
completed its prewarm completed all three formal repeats.

The ratios below divide the GPU4PySCF median by the GenerativeQC median.
Each ratio represents three matched repeats. A dash marks a case with an
incomplete arm. Every paired case above 96 atoms favored GPU4PySCF.

| Method | Route | 96 atoms | 144 atoms | 192 atoms | 240 atoms | 300 atoms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HF | Direct | 0.760× | 0.371× | 0.072× | 0.044× | — |
| HF | DF | 1.830× | 0.172× | — | — | — |
| PBE | Direct | 1.106× | 0.114× | 0.064× | — | — |
| PBE | DF | 1.814× | 0.118× | 0.065× | — | — |
| r²SCAN | Direct | 1.044× | 0.128× | 0.067× | — | — |
| r²SCAN | DF | 1.663× | 0.131× | 0.070× | — | — |
| ωB97M-V | Direct | 0.370× | — | — | — | — |
| ωB97M-V | DF | 1.668× | — | — | — | — |

At 96 atoms, both engines passed all independent checks on every repeat of
HF Direct, PBE Direct and PBE DF. GenerativeQC and GPU4PySCF median times
were 10.937 and 8.310 s for HF Direct, 9.567 and 10.585 s for PBE Direct,
and 5.059 and 9.176 s for PBE DF. PBE grid levels 4 and 5 met the stability
limits. Both HF DF arms exceeded the energy gate on every repeat, with
maximum error `2.075e-5 Eh/atom`. CPU references for 96-atom r²SCAN and
ωB97M-V and 300-atom HF and PBE timed out. Independent references were
available only for HF and PBE at 96 atoms. Across the completed pairs above
96 atoms, the largest cross-engine difference was `7.15e-12 Eh/atom` in
energy and `1.51e-6 Eh/bohr` in any force component.

## Molecular inputs

The five geometries contain 32, 48, 64, 80 and 100 intact SPC water molecules.
They are nested cuts from a liquid-water simulation snapshot distributed by
[ErgoSCF](https://www.ergoscf.org/xyz/h2o.php). The selector takes the oxygen
closest to the centroid of the source cluster and sorts whole water molecules
by oxygen distance from that center. It translates the selected center to
the origin and converts coordinates to bohr. Oxygen radii are 6.17, 6.96, 7.71,
8.37 and 8.89 Å respectively. The snapshots represent liquid-derived hydrogen
bond networks at fixed geometries. They share a parent snapshot.

[geometries.json.xz](geometries.json.xz) records archive and member hashes, the selected
molecule indices and every coordinate. [geometry.py](geometry.py) reconstructs the
selection. Both engines use spherical def2-SVP, charge zero and spin zero,
with the four HF, PBE, r²SCAN and ωB97M-V families in both Direct and DF modes.

## Endpoint and accuracy protocol

[endpoint.py](endpoint.py) calls the existing paper driver with one cold endpoint
per process. The timer starts before calculator construction and ends after both
energy and analytic forces have reached the host. Each engine uses its stock
convergence and initial guess, semilocal quadrature and applicable VV10
quadrature. The GPU4PySCF DFT gradient includes moving-grid response. The DF
auxiliary basis tables are frozen from `paper-hfdft-cold3-20261120`. The long-range
exchange treatment in the stock ωB97M-V DF routes differs between engines. The
DF rows report the complete stock routes and are screened against the common
strict Direct reference.

[acquire.py](acquire.py) schedules a cold prewarm for each arm and case, then
three fresh-process cold observations with alternating engine order. The prewarm
populates artifact caches and remains outside the timing summary. The process
timeout is 600 s, including startup and result collection. If an arm fails its
prewarm, its formal observations remain marked unattempted and the other engine
can continue. NVML is polled at approximately 50 ms intervals through each child
process. Peak memory is the largest observed whole-device value including the CUDA
context. The stored NVML baseline records device memory immediately before the
child. A sampled maximum can miss shorter allocations.
Across completed formal observations, the highest sampled peaks were
53.23 GiB for GenerativeQC (96-atom r²SCAN DF) and 68.01 GiB for GPU4PySCF
(240-atom ωB97M-V DF).

The independent oracle uses CPU PySCF 2.14.0 with stricter Direct SCF, moving-grid
forces, and semilocal levels 4 and 5 where applicable. The relative allocation
of oracle work is documented in [run_cpu.sh](run_cpu.sh). The energy
absolute-error limit is `1e-5 Eh/atom`, maximum Cartesian component force
error is `1e-4 Eh/bohr`, and force RMS error is `3e-5 Eh/bohr`.
For DFT, the difference between CPU levels 4 and 5 must fit within one quarter
of each limit before using level 5 as the reference. HF has one tight Direct
reference with no grid. Oracle coverage and gate results are recorded per sample.
[cpu-raw.json.xz](cpu-raw.json.xz) includes the 1,200 s oracle exits.
The 40-core PBE extension encountered the qz user quota and was cancelled
before an allocated Slurm step. A job with a 20-core allocation and 18
threads completed both 96-atom PBE levels within a single 6,000 s process
limit. [cpu-pbe-long.json.xz](cpu-pbe-long.json.xz) records that run.

## Source and reproduction

[source.json](source.json) pins the Git tree, source archive, runtime source
identity, paper driver and frozen-input digests. The GenerativeQC library is
built for SM90 with compiler caching and paper-style stationary force AOT
profiles for the measured DFT families. The GPU4PySCF and CuPy 13.4.1 environment
matches the paper reference environment. [build.py](build.py) verifies the
export, compiler launcher and loaded binary identity. [build-raw.json.xz](build-raw.json.xz)
contains the build commands and receipts, including the AOT and library audit.

Both timing lanes used NVIDIA H100 80 GB HBM3 devices with driver `595.58.03`,
81,559 MiB reported memory and a 700 W power limit. The 96–192 and 240–300
atom groups ran on two distinct physical cards. Every paired comparison ran
on the same physical card. The two lanes used the identical hashed native
library. [resources.json](resources.json) records allocation, effective
priority, submitted images, terminal status and estimated cost.

Use Python 3.12 or later to reconstruct the stage in a fresh directory.
Download the asset linked in [geometries.json.xz](geometries.json.xz) to
`water_clusters_b_005_to_100.tar.gz` before running the command.

```bash
results=benchmarks/results/h100-water-size-scaling-20261125
python "$results/prepare.py" --repository . \
  --ergo-archive water_clusters_b_005_to_100.tar.gz --root .artifacts/issue-3170-replay
python "$results/replay.py"
```

To reacquire on qz, stage the reconstructed directory in the paper's H100
environment. Use its standard Python interpreter to run `build.py` with the
stage root as its argument, then run `acquire.py --root <stage> --sizes 96 144 192`.
For sizes 240 and 300, prepare a second root with the same staged sources and
inputs. Once the first build is complete, link the second root's `build/` and
`build-receipts/` to the first root's directories on shared storage, as the
recorded run did. Both directories can also be copied together. The
stationary AOT `.so` libraries and their `.json` manifests must be available
beside the main `libgenerativeqc.so`. Run `acquire.py --root <second-stage>
--sizes 240 300` with the second allocation. Set `GQC_QZ_JOB_NAME` and
`GQC_QZ_PROJECT` to the allocated job and project for each lane. The generated
`paper-config.json` records the interpreter and environment paths from that
qz installation. Other installations need equivalent dependencies at their
own paths. Run `run_cpu.sh` under an allocated Slurm step with CPU PySCF 2.14.0.

[replay.py](replay.py) checks the full planned H100 population and raw-file
hashes and regenerates the timing and accuracy tables.
[master-drivers.json.xz](master-drivers.json.xz) holds the
exact scripts used for acquisition. The maintained drivers include subsequent
review fixes to input verification, cleanup and interruption handling. The
speed ratio is the GPU4PySCF median divided by the GenerativeQC median over
jointly completed repeats of the same method, route, geometry and device.
`summary.csv` also reports each arm's median and peak sampled memory across
its own completed formal observations, including rows with only one available
arm. Raw records retain each endpoint's full force vector, elapsed time and
process log. [failures.csv](failures.csv) reports failed and unattempted rows
with the last stage marker, process elapsed time and sampled peak
where available.

## Completion and memory at 240–300 atoms

At 240 atoms, HF Direct was the only GenerativeQC route that completed.
The median was 386.65 s versus 17.04 s for GPU4PySCF, with three fresh-process
pairs. At 300 atoms, six GenerativeQC routes timed out and PBE DF and r²SCAN DF
reported SCF nonconvergence. All eight stopped within the combined native SCF
and analytic-force call. The 300-atom native HF Direct timeout sampled a peak
of 13.17 GiB. The highest sampled native peak at that size was 30.69 GiB
in the ωB97M-V DF timeout. These establish the 600 s protocol's completion
boundary on this geometry series. The device memory ceiling for GenerativeQC
remains undetermined.

At 300 atoms, GPU4PySCF completed HF Direct, PBE Direct and DF, and r²SCAN
Direct and DF, with three observations each and medians of 23.44–43.94 s.
HF DF and ωB97M-V DF stopped with explicit CUDA allocation errors during
analytic forces, with sampled peaks of 79.51 and 79.54 GiB, respectively.
ωB97M-V Direct reported an illegal CUDA address in the VV10 force stack
at both 240 and 300 atoms. The sampled peaks for these attempts were 34.27
and 36.97 GiB, respectively.

## Direct path diagnostics

All three unprofiled 240-atom HF Direct pairs had 34 native SCF iterations
and 11 GPU4PySCF iterations. The formal 96-atom runs had 12 and 11,
respectively. The native cold PBE and r²SCAN Direct receipts record seven
and six incremental J/K builds per repeat at 96 atoms. At 144 atoms
both record zero, with automatic incremental J/K inactive.

The source-matched [native component trace](trace-hf.json.xz) recorded 12 native
SCF iterations at 96 atoms and 33 at 240 atoms. The 96-atom diagnostic used the
Hückel guess. At 240 atoms the initial-guess work receipt records
`budget_skipped`. This is a change in the observed initialization route.

The [Nsight census](trace-nsys.json.xz) covers two additional diagnostic
GenerativeQC endpoints. For HF Direct at 240 atoms, the union of CUDA kernel
intervals was 370.777 s. Generated RHF Fock kernels occupied 289.263 s and the
bounded force-wave kernel occupied 62.925 s. At 96 atoms for ωB97M-V Direct,
the kernel union was 94.718 s. The three canonical J/K kernels accounted for
43.279 s, generated Fock kernels for 13.385 s, and the two named Direct
force kernels for 20.376 s. The comparison uses the unprofiled process timings.

The original QDSTRM hashes were recorded before import. The first importer run
altered the two QDSTRM streams in place. The post-import streams, Nsight reports
and SQLite databases are retained outside Git, and their identities and
import/export output are archived with the kernel census. The maintained
converter works from a temporary copy.

For the 144-atom semilocal [trace](trace-semilocal.json.xz), the demangled
stationary XC force-geometry kernel ran 1,580 times per endpoint. Its summed
duration was 104.010 s for PBE Direct, 104.039 s for PBE DF, and 106.376 s for
r²SCAN Direct. The corresponding device kernel unions were 124.794, 112.200
and 126.618 s. The lone quadrature-setup kernel with the same short name was
resolved separately through demangled symbols. Each QDSTRM input was preserved
and the converter imported a scratch copy.

## Earlier revision

The first diagnostic Job ran `d154546d2`. Its GPU4PySCF instrumentation replaced
class bindings and the reference stack failed an `isinstance` check. The second
Job corrected the instrumentation and recorded six complete 96-atom pairs
before the source changed to `22f72d069`. The earlier CPU Job finished the
96-atom HF reference before it was stopped. Earlier GPU and CPU receipts are
kept alongside [source-legacy.json](source-legacy.json) and are excluded from
the reported speed ratios and gates for `22f72d069`.
