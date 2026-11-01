# Portable integration: three-device endpoint evidence

This is a **default-grid DF96 snapshot**, not a universal GPU speed guarantee.
All seven README methods lead GPU4PySCF in the measured cold and warm populations
on RTX 5090 and RTX PRO 6000 Blackwell. A100 leads in all seven warm populations,
but five cold populations remain slower. The integrated image combines native
OpenBLAS discovery and bounded packed projection; lazy range warm energy is off.

Each cell below is **cold / warm**, computed as
`100 * (median(native) / median(GPU4PySCF) - 1)`. Positive means native is slower.

| Method | A100 | RTX 5090 | RTX PRO 6000 Blackwell |
| --- | --- | --- | --- |
| PBE | -5.59% / -5.01% | -9.57% / -9.69% | -18.39% / -8.98% |
| PBE0 | +19.12% / -5.23% | -8.45% / -42.04% | -13.18% / -41.68% |
| B3LYP | +16.10% / -3.98% | -12.83% / -41.33% | -17.02% / -41.05% |
| r2SCAN | -11.08% / -20.52% | -17.15% / -29.88% | -22.18% / -29.66% |
| r2SCAN-3c | +27.66% / -1.83% | -0.81% / -31.23% | -4.31% / -29.98% |
| M06-2X | +10.55% / -17.82% | -13.10% / -48.13% | -16.54% / -47.82% |
| WB97M-V | +54.52% / -21.05% | -38.15% / -54.06% | -36.68% / -63.18% |

A separate RTX 5090 r2SCAN-3c repeat gives -1.13% cold and -31.33% warm.
The small cold advantage is an observed result, not a robust universal margin.
A100 WB97M-V medians are 137.017784 / 88.675083 seconds cold and
23.610583 / 29.907136 seconds warm (native / GPU4PySCF).

## Protocol and provenance

Cold includes calculator construction, preparation, SCF, analytic forces and
host export. Each method has native and GPU4PySCF conditioning (cold plus one
warm), then native/reference/reference/native fresh owners, each with cold plus
three frozen engine-local warm calls. Conditioning is included in accuracy gates
and excluded from the timing medians. Both engines retain their own convergence
rules and initial guesses; iteration counts are not equivalent Fock counts.

Stock semilocal/VV10 grids, spherical orbital bases, method corrections and the
common independent README reference gates are retained. Native uses explicit
SCF/force tiles 256/1024, eight scheduled CPU slots and single-threaded math.
GPU4PySCF retains its production optimizations and moving-grid force response.
This qualifies these sampled static geometries only. Earlier projection/lifecycle
suites cover their separately documented moved/larger/resource domains.

Allocations are LSF 224192/224193/224194 on gpu22 (A100), Slurm 15044 on n1 and
1412 on n4 (5090), and Slurm 3842 on n2 (PRO 6000). The independent 5090 repeat
is Slurm 15045. Each comparison remains within its own allocation and physical
GPU; none uses a ratio assembled from different jobs. Raw scheduler visibility,
packages, device locators, work, settings and all observations are retained.

The measured native source identity is
`eb69035ae9a9e3586d63c95dc1b5ab1667ddc40d252cbc460637f5ee935e17f4`,
verified byte-for-byte against the native inventory at
`5adef0250199a6efbe0bc41156266c6c4b702cd7`. CPU provider CMake is separately
verified against `90c9af4c1d52523d53906449b2e0576397a52073` because that file
is outside the native inventory. Original dirty-source receipts are unchanged.
The main binary hashes are `874f660b7ba1a9c6844da5c95a1b3ab4d3616625ef363662050574891f8b62d2`
(A100) and `f2057442c40d3cacfcdaed39d29d7498fcd9ddc871dca87a4e4e0249881a78ef`
(Blackwell). These are **pre-capability-getter-fix images**, not measurements of
later PR heads. Compiler/provider and binary identity cannot be replaced by the
native source hash alone.

## Tuning evidence and rejected changes

SCF point tiles 256/512/1024 are explicitly selected public options, not changed
defaults. Conditioning plus forward/reverse populations retain cold and three
warm calls on each GPU (LSF 224196; Slurm 15046 and 3843). On A100 B3LYP,
1024 reduces cold/warm time by 8.48%/2.28% relative to 256. The corresponding
cold reductions on 5090/PRO 6000 are 2.04%/1.88%. r2SCAN-3c at 1024 reduces
cold time but increases warm time by 0.96%/0.84%/0.94% on A100/5090/PRO 6000.
This rejects an unconditional larger tile. These ablations are not new GPU4PySCF
rankings and cannot be multiplied into the table above.

The A100 compiled matrix-tile 16-to-32 experiment (224187) regresses B3LYP,
r2SCAN and r2SCAN-3c cold time by 5.18%, 8.53% and 7.23%; warm time also
regresses. Both sides use their original same-allocation provider/build image,
separate from the integrated ranking image. Computation and nine GPU tests
completed; a disallowed upload filename caused the LSF exit. A separate scheduled
CPU recovery preserved the complete original artifacts and computation exit code.

Full-target WB97M-V SCF with forced range residency (224200) reduces iterations
28 to 16 but raises SCF VV10 pair sweeps 4 to 16. Complete cold time rises
35.06%; warm is essentially unchanged (+0.01%). Two controls change together;
this is not an isolated solver or residency attribution. Both controls and
all successful numerical gates remain visible. Do not select it using the
iteration count alone.

Fixed-density XC probes use two 96-atom geometries, tiles 128/256/512 and six
observations per route. Indexed SYR2K is slower than generated execution; an
experimental FP64 GEMM plus symmetric scatter is faster in the 256/512 component
cases. Independent scalar, CPU local-map, capture and resource gates pass.
The prototype also selects default cuBLAS math for its typed FP64 call and
retains old rank2k diagnostic metadata: its source override and original/candidate
headers are archived. It is **not a production lowering identity or an SCF/force
endpoint speedup**. Full-endpoint qualification is separate.

Failed scheduler scripts 224184/224185 retained successful native conditioning
but never ran their GPU4PySCF arms because of an old-Bash empty-array error.
Provider build 224197 lacked its deployed fixture. All three nonzero exits and
available receipts remain in this bundle and are excluded from rankings.

## Replay

```bash
PYTHONPATH=python:. python benchmarks/results/portable-integration-20261101/report.py
```

The reader authenticates 522 original files, restores every FP64 word without
rounding, and rechecks **212 complete populations / 704 energy-and-force calls**,
including conditioning from unsuccessful allocations. It checks independent
energy/full-force/grid-stability/CPU-GPU spot gates, source/binary and allocation
identity, Fock timelines, all-repeat paired errors and complete timing counts.
The output includes raw timing populations and semantic work, not just medians.
The 24 component observations have their own explicitly narrower gates.
`python -O` retains all checks. No external archive or Release is required.
