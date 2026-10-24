# WB bounded reuse: complete cold endpoints

Measured source: `c7eec7ed780e2a72b449edf995c8041ec30806f7`.
Water96, ωB97M-V/def2-SVP, spherical orbitals, unchanged stock grids and
convergence/precision gates, full energy plus analytic forces including grid
response. Native DF uses its declared fitted primary exchange plus exact
long-range correction; Direct keeps exact J/K. No reference density enters either
production path. RTX 5090 on n1, CUDA 12.9.1, release sm120, one numerical host
thread, finite Slurm jobs and verified sccache 0.16.0.

Five alternating native/GPU4 fresh-process pairs per mode retain cold and three
warm endpoints. Each process contributes one median warm observation. Timings
include construction, preparation, SCF, forces and host export. All 56 native
calls (40 paired, 16 cold/warm/moved/moved-warm) and all 40 paired GPU4 calls pass
the independently qualified stationary reference or exact moved-geometry control.
Moved controls establish same-geometry equivalence, not a new dense-grid
convergence study. Standard and Fast moved trajectories are both retained.

| Mode | Phase | GPU4 median ± MAD (s) | Native median ± MAD (s) | Lower latency |
| --- | --- | ---: | ---: | ---: |
| DF | cold | 389.601134 ± 0.161403 | 342.944186 ± 0.275319 | 11.976% |
| Direct | cold | 379.687684 ± 0.136042 | 281.252087 ± 0.431503 | 25.925% |
| DF | warm | 134.400591 ± 0.054874 | 45.870282 ± 0.015981 | 65.870% |
| Direct | warm | 122.258098 ± 0.009703 | 29.885357 ± 0.092340 | 75.556% |

Every difference exceeds `max(2%, 2*(baseline relative MAD + candidate relative
MAD))`. These are default-grid endpoint observations on this workload, not a
universal accuracy-qualified ranking. They do not combine timings from earlier
factor/cache pilots or establish gains for other functionals.

The integrated path restores only proved held-density factors, admits bounded
range reuse with a fitted primary, bounds through-d LR force call graphs and
requires a complete occupied U projection before admitting resident packed B.
Explicit cache controls, generic angular fallback and actual-memory refusal stay
available. Endpoint receipts retain iteration counts, Fock work and resources.

Validation includes 312 host checks (one optional skip), 24 Direct/high-angular
GPU checks, 131 range/selector checks, independent CPU fitted-WB composition,
cold/warm factor proofs, complete Fock composition, actual-memory headroom/replay
and zero Compute Sanitizer errors. An additional 68 host lifecycle checks compile
the real held-factor restoration method after fixing an incomplete test harness.
That test-only repair does not change the measured production source.

The 6,001-input source manifest is reconstructed against the retained raw-J
inventory and authenticated by SHA-256. The initial login-shell verification
failed because `python` was unavailable. A corrected `python3` pass authenticated
all inputs **during compilation and before device validation**; the original
verification receipt explicitly retains this limitation. Binary hashes, compiler
cache statistics, full validation logs, runners and unmodified E/F vectors are
losslessly stored in `receipts.json.xz`. Seven historical dft-grid-160 records are
also losslessly compacted with exact-byte restoration; storage caps are unchanged.

Reproduce all numerical gates and statistics without a GPU:

```bash
PYTHONPATH=python:. python benchmarks/results/wb-range-reuse-20261023/reproduce.py
```
