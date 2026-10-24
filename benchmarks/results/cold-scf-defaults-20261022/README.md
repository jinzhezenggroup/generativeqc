# Targeted cold-SCF default qualification

These are targeted measurements of the new solver policy, **not a refreshed
README campaign**. XC/VV10 grids, complete force response, auxiliary inputs and
observable gates are unchanged. Only cold and three frozen warm calls are timed;
CBS and the nonlocal solver policy are unchanged.

## Iterations at 96 atoms

| Method | Previous README Direct | New Direct | New DF | Existing stock GPU4PySCF Direct / DF |
| --- | ---: | ---: | ---: | ---: |
| HF | 14 | 11 | 11 | 10 / 10 |
| PBE | 14 | 11 | 10 | 11 / 11 |
| PBE0 | 14 | 9 | 9 | 9 / 9 |
| B3LYP | 14 | 9 | 9 | 10 / 10 |
| r2SCAN | 14 | 9 | 9 | 10 / 10 |
| r2SCAN-3c | 15 | 10 | 10 | 11 / 11 |
| M06-2X | 14 | 10 | 10 | 10 / 10 |

The historical columns retain their original source `1178a03c7f45074cc705a7accd676f0ffc1b6d2a`;
GPU4PySCF was not rerun. Native initial-guess work and actual final Fock builds
remain in the receipts, separate from iterations. Direct HF executes 13 Fock
builds for 11 iterations. All local KS warm calls take one iteration; HF DF
retains its existing two-iteration warm path. Complete observed times and work
are in [summary.json](summary.json), without a claim of universal speed leadership.

Four Fast spot checks (HF Direct, PBE Direct/DF, PBE0 DF) each execute exactly
one solve per call. PBE Direct cold falls from 23.983 s Standard to 19.712 s Fast;
PBE DF from 42.078 s to 39.359 s. HF and PBE0 DF use their Standard fallback;
small elapsed-time differences are not significant speedups.

## Numerical acceptance

All 20 independent same-model cases pass: Direct/DF water across seven methods,
plus Direct HF/PBE0/M06-2X ammonia and methane. Original and displaced geometries
each execute a cold and frozen warm endpoint, for 80 native energy/force calls.
The independent GPU4PySCF oracle solves tightly on the same default grid and
fitted operator, with complete grid response and no shared initial density.
The worst errors are `2.17e-11 Eh/atom`, maximum force `1.31e-6 Eh/Bohr`, and RMS
force `7.19e-7 Eh/Bohr`; the solver gates are `2.5e-6`, `2.5e-5`, and `7.5e-6`.

The separate dense-reference common gates remain `1e-5 Eh/atom`, `1e-4 Eh/Bohr`
maximum force, and `3e-5 Eh/Bohr` RMS force. Of 23 targeted timing populations,
19 pass and **four remain failed and unranked**: HF DF at 3/96 atoms (fitting
energy error), and M06-2X Direct/DF at 3 atoms (default-grid force error).
Qualification covers every cold and warm sample; failed values are retained.
These independent model-error failures are not cured by stricter SCF.

All 32 strict-control/UKS compatibility tests pass. Native CUDA boundary tests
also verify physical residual limits, strict overrides, nonfinite inputs and
inactive neighbors in iteration and force-validation kernels. The first run with new
implicit defaults exceeded the old `1e-6` force gate in 12 tests; that log is
retained. The strict tests now explicitly request their original tolerances,
while separate new-default tests apply the declared solver budget. No numerical
coverage or common dense-reference gate was removed.

## Source and reproduction

The measurements used **base plus the archived reconstruction patch**, not the
clean base `afa276414b22c813f6d007c62abdf58158de5a35` alone. All frozen source inputs
were verified; every unchanged file was checked against that base, and every
changed/new file against the reconstruction. The compressed evidence contains
that patch, changed-file hashes, the full input-manifest digest, 25 library and
manifest hashes, compiler-cache statistics, launchers, test sources and logs.
The core SHA-256 is `954ebda55f03369c1abde72a4d1f20baecad9107312afb55d3b3de6ed66f4e16`.
Full build logs and the full source manifest remain under the recorded ignored
n1 artifact directory. Compilation used verified sccache 0.16.0. Every real GPU
execution used a finite Slurm allocation on n1 RTX 5090.

```bash
PYTHONPATH=python:. python benchmarks/results/cold-scf-defaults-20261022/reproduce.py
```

This checks retained numerical receipts without a GPU and does not redraw or
replace README figures. See the [decision note](../../../.agents/notes/implemented/numerics/2026-10-22-standard-cold-scf-convergence.md)
for final-state rejection handling, compatibility boundaries and rationale.

Four historical JSON records were losslessly minified to keep the existing
aggregate evidence size limit. [Storage receipts](storage-compaction.json) retain
before/after hashes; all parsed values and their order are unchanged.
