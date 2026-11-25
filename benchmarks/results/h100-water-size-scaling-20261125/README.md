# Cold HF and DFT size scaling on H100

This probe for [#3170](https://github.com/jinzhezenggroup/generativeqc/issues/3170)
uses GenerativeQC revision `22f72d0692a760c2892cd7791022584d9ae1c3e5` and
GPU4PySCF 1.9.0. It measures complete cold energy and analytic forces on
96, 144, 192, 240 and 300 atoms.

The master acquisition is in progress. This checkpoint contains 25 complete
formal endpoints and 12 pairs. [summary.csv](summary.csv) reports the observed
pairs and [gates.json](gates.json) contains the sample-level oracle errors.
The first 96-atom HF Direct pair reached 10.937 s and 8.240 s, respectively,
with both endpoints passing the CPU PySCF gates. The first 240-atom HF Direct
pair reached 386.520 s and 18.201 s, respectively, and recorded 34 and 11 SCF
iterations. These are single observations on their respective H100s.
The larger-size sweep and repeat acquisition continue.

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

The independent oracle uses CPU PySCF 2.14.0 with stricter Direct SCF, moving-grid
forces, and semilocal levels 4 and 5 where applicable. The relative allocation
of oracle work is documented in [run_cpu.sh](run_cpu.sh). The energy
absolute-error limit is `1e-5 Eh/atom`, maximum Cartesian component force
error is `1e-4 Eh/bohr`, and force RMS error is `3e-5 Eh/bohr`.
For DFT, the difference between CPU levels 4 and 5 must fit within one quarter
of each limit before using level 5 as the reference. HF has one tight Direct
reference with no grid. Oracle coverage and gate results are recorded per sample.

## Source and reproduction

[source.json](source.json) pins the Git tree, source archive, runtime source
identity, paper driver and frozen-input digests. The GenerativeQC library is
built for SM90 with compiler caching and paper-style stationary force AOT
profiles for the measured DFT families. The GPU4PySCF and CuPy 13.4.1 environment
matches the paper reference environment. [build.py](build.py) verifies the
export, compiler launcher and loaded binary identity before collecting results.

Use Python 3.12 or later to reconstruct the stage in a fresh directory.
Download the asset linked in [geometries.json.xz](geometries.json.xz) to
`water_clusters_b_005_to_100.tar.gz` before running the command.

```bash
results=benchmarks/results/h100-water-size-scaling-20261125
python "$results/prepare.py" --repository . \
  --ergo-archive water_clusters_b_005_to_100.tar.gz --root .artifacts/issue-3170-replay
python "$results/replay.py"
```

[replay.py](replay.py) checks the retained bytes and regenerates the timing and
accuracy tables. [master-drivers.json.xz](master-drivers.json.xz) holds the
exact scripts used for acquisition. The maintained drivers include subsequent
review fixes to input verification, cleanup and interruption handling. The speed ratio is the GPU4PySCF median divided by the
GenerativeQC median over jointly completed repeats of the same method, route,
geometry and device. Reported ratios characterize these samples and fixed
geometries.

## Direct path diagnostics

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

## Earlier revision

The first diagnostic Job ran `d154546d2`. Its GPU4PySCF instrumentation replaced
class bindings and the reference stack failed an `isinstance` check. The second
Job corrected the instrumentation and recorded six complete 96-atom pairs
before the source changed to `22f72d069`. The earlier CPU Job finished the
96-atom HF reference before it was stopped. Earlier GPU and CPU receipts are
kept alongside [source-legacy.json](source-legacy.json) and are excluded from
the reported speed ratios and gates for `22f72d069`.
