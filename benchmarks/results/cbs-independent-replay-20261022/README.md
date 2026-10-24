# Independent Lambda replay: measured pilot

Complete energy plus analytic forces; all-electron QM7, cc-pVDZ/cc-pVTZ
MP2-assisted CBS, DZ CCSD(T) correction, RI-JK/shared JKFIT, Fast, 32 GiB budget.
Each pair uses one Slurm-allocated RTX 5090 on n1; different GPUs are not pooled.

| Cohort | Baseline (s) | Candidate (s) | Time reduction |
| --- | ---: | ---: | ---: |
| C4 | 67.352 | 67.326 | unchanged |
| C6, candidate first | 491.030 | 468.282 | 4.63% |
| C6, baseline first | 488.278 | 466.059 | 4.55% |
| C7 | 1098.699 | 1034.299 | 5.86% |

Separate C6 traces show replay 51.261 -> 29.037 s, Q batches 742 -> 47 and kernels
41,729 -> 4,454. Semantic summands are essentially unchanged. Operator Q32 is
preserved; replay uses Q16. Capacities and solver work match. Paired energies are
identical; maximum force difference is 1.762e-13 Eh/Bohr. Clocks are unlocked.

772 related tests, nine independent full-DF tests and 18 memchecked new cases
pass. Six water calls pass fresh PySCF full-vector differences. This algebraic
fixture does not establish independent full-vector C6/C7 or CBS-limit accuracy.

**Build scope:** two Lambda objects replaced over the qualified
[CBS-memory binary](../cbs-fast-memory-20261021/README.md), not a clean whole-master
build. GCC 11.4/CUDA 12.9.1, Release/sm120, cross-size arena ON, ccache 4.5.1.

`evidence.json.xz` retains raw timings, water errors/tolerances, capacities,
semantic work and identities. Full vectors, oracles, commands and logs remain
in `.artifacts/cbs-replay/trial-evidence.json.xz`; its hash is retained. This is
a compact summary, not raw-vector replay or broad performance promotion.
The [decision note](../../../.agents/notes/implemented/performance/2026-10-22-df-lambda-independent-replay.md)
records details. Use the command in `publication.json` separately with each DSO,
through finite Slurm allocations preserving assigned device visibility. Timing
includes calculator construction, both basis groups and returned forces; startup,
basis snapshots, serialization and oracle work are excluded.
