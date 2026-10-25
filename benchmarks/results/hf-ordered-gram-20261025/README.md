# Restricted HF ordered Gram reuse

Large restricted CUDA HF reuses the existing scalar-order FP64 DIIS Gram cache.
The guard is at least 512 AOs and history capacity two; smaller/unrestricted
owners retain serial defaults. Explicit `GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM=0`
rolls back. Resource admission, plan rebuilding and execution share the guard.
The recurrence, dot/extrapolation arithmetic order, tolerances and Fock work
remain unchanged; no reference density enters a production call.

## Complete endpoint results

Three alternating fresh-process rounds retain **24 populations / 96 complete
energy-and-force calls** on RTX 5090. Cold includes construction, preparation,
SCF, forces and host export. Every population retains three frozen-density warm
calls. See [all medians and ranges](timings.md); no completed result is hidden.

| Profile | Atoms | Control cold (s) | Candidate cold (s) | Change |
| --- | ---: | ---: | ---: | ---: |
| Standard | 48 | 6.194 | 6.095 | -1.60% |
| Fast | 48 | 6.106 | 6.132 | +0.42% |
| Standard | 96 | 19.366 | 18.471 | -4.62% |
| Fast | 96 | 19.413 | 18.493 | -4.74% |

HF48 is below admission and measures ordinary variation. HF96 retains eleven
cold Focks and one warm Fock, with warm near 2.93 s. Its approximately 18.5 s
cold remains slower than the preceding README GPU4 observation of 12.839 s;
that older observation is context, not an interleaved comparison in this cohort.
These scoped measurements do not replace the complete README figure cohort.

## Correctness and work

All paired samples retain iterations, Fock counts and scientific settings.
The reproducer enforces 1e-8 Eh paired energy and 1e-9 Eh/Bohr maximum force
change. All 48 candidate samples pass independent retained GPU4 references
at unchanged common gates (1e-5 Eh/atom, max force 1e-4, RMS force 3e-5 Eh/Bohr).
Fresh independent original/moved references additionally qualify **24 checks**
at 66/96/120 atoms, both profiles, including E+F → rollback → moved E → moved E+F.
Their maximum observed force error is below 4.83e-7 Eh/Bohr.

Native tests independently check ordered dot values, bitwise extrapolation,
ring resets and inactive slots at 512/513 AOs and one/two spins. The host
selector matrix covers size/history/spin boundaries and explicit/invalid
controls. The main related GPU run passes 49 cases; six initially gated
correlated lifecycle cases pass on rerun. An intermediate rerun passed two
and skipped four because `threadpoolctl` was missing; installation and final
rerun receipts are retained. This changes no CCSD(T) benchmark result.

## Provenance and reproduction

Control: `7b2ab52fdb37ebdd8d149664530e77f9b8cc8cc0`.
Candidate: `fba34e9192edf0ea0d5baf6e3cd1f0b9ab14dc5d`.
Both include the preceding response-capacity change and then-current master.
All 12,796 files in each immutable measurement tree match its Git archive.
The independent source/binary bindings, CUDA 12.9.1/GCC 13.3 Release sm120,
verified sccache 0.16.0, cache statistics and compiler rules are retained.
Each GPU job uses finite Slurm allocation on node1 with assigned visibility,
four CPU cores and one OMP/OpenBLAS/MKL thread. Full scripts and allocations
are in `jobs.json.xz`; source archives/binaries remain in ignored node1 artifacts.
Later changes add tests, documentation and DIIS-control provenance only.

```bash
PYTHONPATH=python:. python benchmarks/results/hf-ordered-gram-20261025/reproduce.py \
  --output .artifacts/hf-ordered-gram-replay
```

This rechecks all paired and independent gates from lossless receipts and
regenerates the table. Adapt the retained installation paths for GPU reexecution,
always inside finite `srun`. Retained GPU4 references preserve their original
producer identities; they share geometry, orbital basis and observable contract.

`negative-trials.json.xz` keeps all four separate HF96 source-cache experiments:
ordinary cold 19.318 s versus 97.316/64.554/83.060 s for three partial ERI masks.
They are rejected experiments, not headline samples or new defaults.
