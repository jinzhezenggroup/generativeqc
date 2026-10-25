# DF source and force scheduling qualification

Run `python benchmarks/results/df-source-force-20261025/reproduce.py` from any
checkout to authenticate the capsule, replay all 129 complete energy/force
checks and recompute cold/warm medians and ranges. No GPU or NumPy is required.

Three rotated rounds compare base `6d508c262`, the retained candidate patch and
GPU4PySCF on one Slurm RTX 5090 allocation. Each population uses a fresh process,
one complete cold energy-plus-analytic-force call and two frozen warm calls.
The native scientific settings and target iteration/Fock work are unchanged.

| Case | Control cold (s) | Candidate cold (s) | GPU4 cold (s) | Control warm (s) | Candidate warm (s) | GPU4 warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PBE48 | 5.7998 | 5.3467 | 6.7275 | 1.7751 | 1.3292 | 1.4456 |
| PBE96 | 15.3923 | 14.5239 | 12.5753 | 5.6940 | 4.8601 | 4.5277 |
| r2SCAN-3c96 | 32.4400 | 30.1772 | 25.7387 | 10.3259 | 8.7214 | 12.4609 |

The candidate compacts exactly zero current-density Hermite sources and selects
larger bounded force tiles. Large ordinary all-electron DF force caps increase
from 256/512 MiB host/device to 512 MiB/1 GiB; these are additional force
reservations. Public DFT DF exact-byte resource planning remains unsupported.
There is no tolerance screening, relaxed convergence or reduced DF value budget.

`paired.json.gz` retains all timed populations, `moved.json.gz` the independent
original/moved references and native checks, and `ablations.json.gz` the separate
source-only and capacity comparisons. Shared-process qualification-reference
timings are not comparison populations. `provenance.json` and the compressed
patch bind the final production source; native, Python and final test snapshot
relationships are explicit. Executed scripts, compiler-cache receipts, binary
hashes, tests, allocations and the earlier stale-AOT refusal remain retained.

Qualification is limited to the measured water workloads. PBE96 and r2SCAN96
remain slower cold than GPU4PySCF. Root README figures are unchanged. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-25-df-source-and-force-work.md)
for resource, numerical, response and graph-lifetime boundaries.
