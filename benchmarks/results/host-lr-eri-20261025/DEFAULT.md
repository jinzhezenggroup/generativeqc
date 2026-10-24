# Automatic bounded RAM tier qualification

This is a separate cohort from the earlier [opt-in screen](README.md).
It measures the integrated automatic policy at commit
`3af978574ab80422a8aeb4409ed4575b8000fa6c`, with library SHA-256
`f18d040e90c395535015ba43d2bb1d758433b29f9c6c7cd8a2aa3cf33401b1c9`.
The source archive was checked against all 12,923 committed regular files;
the retained 1,965-file native inventory reproduces the library's source identity.

Unset host capacity uses at most 32 GiB and one quarter of available machine
and cgroup-ancestor headroom. GPU buckets retain priority. Explicit zero
disables RAM backing. This cohort contains no disk spill.

## Complete endpoints

WB97M-V/def2-SVP spherical DF, strict FP64, default grids, complete preparation,
SCF, analytic forces and host export. Same binary, RTX 5090 on n4, Slurm job
1339, six allocated CPUs, 96 GiB RAM, one BLAS/OpenMP thread, finite 90-minute
allocation. The scalar CPU linear algebra build is unchanged between arms;
historical OpenBLAS results must not be pooled with this cohort.
The driver's declared validation-GEMM force-response route remains enabled.

| Atoms | GPU partial cold (s) | Automatic GPU + RAM cold (s) | Reduction |
| --- | ---: | ---: | ---: |
| 12 | 8.4424 | 8.4410 | within noise |
| 96 | 321.8565 | 254.0091 | 21.08% |
| 108 | 681.4694, 681.7200 | 361.1968, 361.3764 | 46.99% of medians |

The 108 order is GPU, auto, auto, GPU, in fresh processes. The smaller sizes
have one GPU/auto pair each. These samples do not establish a universal
crossover or a confidence interval. Frozen-density warm medians are 56.0260 s
and 56.0409 s at 108 atoms; the one-step warm call does not rebuild the cache.
Both arms take 28 SCF iterations and 29 cold Fock builds at 96/108 atoms;
12 atoms takes 30 iterations and 31 Focks. Every warm call takes one Fock.

The separate CUDA-event profile retains 28 cached SCF actions plus two strict
Direct actions (final physical work and warm replay). Each cached 108-atom
action visits 2,156,803,082 canonical values: 1,169,998,706 on GPU and
986,804,376 in RAM. It transfers 7,894,435,008 bytes and performs zero omitted
recurrences. Host replay event durations include staging and waits; they are
not standalone PCIe bandwidth measurements. The earlier n5 96-atom screen
admitted no host values; this node's new result does not overwrite that evidence.

## Accuracy, diagnostics and provenance

All 18 complete E+F calls (eight formal cold/warm pairs plus a separate primer)
pass the existing independent PySCF/GPU4PySCF gates: energy per atom 1e-5 Eh,
force maximum 1e-4 and RMS 3e-5 Eh/bohr. Dense-grid stability, small CPU/GPU
agreement and audited reference-reuse identities are rechecked offline.
Slurm job 1338 passes 53 GPU cases: four independent matrix cases, nine
default/refusal/lifetime endpoint cases and 40 existing range regressions.
The previous screen retains sanitizer and overlap evidence. New CPU checks
cover automatic capacity and progress-scope name lifetime.

The executed benchmark script accidentally supplied `3af978574` instead of the
full revision. Original receipts and their hashes remain unchanged in
`default.json.xz`. `source-binding.json` records the verified archive, full
commit, native inventory and library hash. The verifier explicitly expands
only that proven abbreviation in a qualification input view, and reports both
the raw and derived receipt hashes. It does not relabel observations as a newer
implementation or change independent reference producers.

The original progress journal has invalid UTF-8 from borrowed dynamic scope
names. Its hash and failed disposition are retained; none of its counters or
timings is used. The independently owned CUDA-event ledger is valid. Commit
`96e4358f8` fixes name ownership, with an executable lifetime regression. Later
full-range/disk diagnostics exercise the corrected journal separately.

Run from the repository root:

```bash
PYTHONPATH=python:. python benchmarks/results/host-lr-eri-20261025/default_reproduce.py
```

The packet retains exact endpoint receipts, per-action CUDA work, compiler
cache statistics, executed scripts, source binding and JUnit evidence. Raw
large journals and binaries remain in ignored local artifacts. The later
diagnostic-only fix is not claimed as the measured source.
