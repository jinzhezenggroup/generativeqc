# Complete auxiliary metric value symmetry

Complete source panels evaluate each unordered public auxiliary pair once and
publish both orientations. Partial/offset panels and coordinate derivatives keep
their original path. Dense memory, FP64 math, basis transforms, metric cutoffs,
SCF criteria and grids are unchanged.

Three alternating fresh-process rounds on one finite Slurm allocation retain
18 populations / 54 complete Standard energy-and-force calls. Cold includes
construction, preparation, SCF, forces and host export. Each cold density seeds
two frozen warm calls. The independent GPU4 checks use the original and a displaced
geometry, including warm replay and a freshly rebuilt displaced owner.

| Method | Atoms | Baseline cold (s) | Candidate cold (s) | Baseline warm (s) | Candidate warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBE | 48 | 5.949 | 5.949 | 1.779 | 1.776 |
| PBE | 96 | 15.749 | 15.545 | 5.698 | 5.703 |
| r2SCAN-3c | 96 | 33.271 | 32.506 | 10.298 | 10.299 |

The capsule retains all samples and ranges. PBE48 and warm differences are noise
at this scale; the useful observed gain is cold preparation at larger auxiliary
sizes. This targeted comparison does not update the README cohort or establish
GPU4 parity. Default quadratures are preserved; the sampled independent checks
do not replace the broader grid-converged README reference campaign.

Validation includes 80 native raw M/A/J/K comparisons and 16 derivative cases
against independently normalized libcint/NumPy results: Cartesian, spherical,
mixed representations, signed/long contractions, auxiliary g, deficient metrics,
batch-specific shell order, all three mappings and ragged panels. Another 50 GPU
tests cover public original/moved/replayed endpoints and bounded sources. A
compute-sanitizer memcheck run of the ragged auxiliary-g batch reports zero errors.
All 12 large original/moved checks pass energy `1e-5 Eh/atom`, maximum force
`1e-4 Eh/Bohr` and RMS force `3e-5 Eh/Bohr` gates. All paired Standard samples
preserve settings, iterations and Fock counts within `1e-8` energy/force deltas.

`source.json.xz` preserves a public reconstruction base, the exact control patch,
the single candidate source patch and archive/source hashes. No dirty experiment
is relabeled as a clean Git revision. `execution.json.xz` retains Slurm allocations,
commands, scripts, verified sccache 0.16.0, compiler commands/cache statistics and
core/AOT hashes. The hardware is an RTX 5090 on node1, CUDA 12.9.1, Release sm120.

Replay every retained gate and the full timing ranges with:

```bash
python benchmarks/results/df-metric-values-20261025/reproduce.py
```

Re-execution requires the same fixtures and matched core/AOT identities, a verified
compiler cache, and a finite `srun` GPU allocation. The source/native test scripts
in `execution.json.xz` specify the executed commands. No CCSD(T) or root README
measurement is replaced here.
