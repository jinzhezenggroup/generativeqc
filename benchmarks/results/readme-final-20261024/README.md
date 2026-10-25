# Default-grid README comparison (2026-10-25)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **192 populations / 768 complete energy-and-force calls** use frozen source `92674610bac506d76b762bc30594e72aae0a1baa`. Eight methods, Direct/DF and 3/12/48/96 atoms are remeasured for native Standard/Fast and one stock GPU4PySCF SCF profile. Each population contains one cold and three frozen-density warm calls. No old timing enters these curves. CCSD(T)/CBS is unchanged.

## Observed comparisons

Every complete performance point is visible, including points that fail the independent accuracy gates. A win is a lower observed complete endpoint latency; accuracy is retained separately and does not filter the denominator. Cold has one observation, and warm reports the median and full range of three calls in one process. Small differences are not significance claims.

| Mode | Phase | Standard wins / measured pairs | Fast wins / measured pairs |
| --- | --- | ---: | ---: |
| direct | cold | 27 / 32 | 29 / 32 |
| direct | warm | 30 / 32 | 30 / 32 |
| df | cold | 18 / 32 | 18 / 32 |
| df | warm | 24 / 32 | 26 / 32 |

Standard and Fast outperform GPU4PySCF in **54/64** and **56/64** measured warm cases.
For 96-atom Direct ωB97M-V, warm speedups are **4.15× (Standard)** and **4.27× (Fast)**.

The observations do not establish universal leadership over GPU4PySCF. The timing and work tables retain slower points, every exposed Fast solve, and the actual per-call controls.

## Accuracy and validation

| Profile | Attempted | Complete | Qualified | Failed accuracy | Unverified |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 64 | 64 | 58 | 6 | 0 |
| fast | 64 | 64 | 58 | 6 | 0 |
| gpu4pyscf | 64 | 64 | 58 | 6 | 0 |

Common gates remain energy **1e-5 Eh/atom**, maximum force **1e-4 Eh/Bohr**, and RMS force **3e-5 Eh/Bohr**. Every cold/warm sample is independently checked; reference-grid stability and CPU/GPU spots use quarter gates. DF fitting error is included. `qualification.json.xz` retains all failures and their reasons. No grid, auxiliary basis or numerical threshold is changed to turn a failed point into a pass.

Source-matched native and Python regressions retain complete endpoint comparisons, changed geometries, resource fallbacks, memory/race checks and full provider synccheck. Complete device-launch Graph synccheck remains instrumented-unqualified: a standalone unconditional-barrier reproduction fails on both sanitizer versions while ordinary launch/Graph controls, output checks, memcheck and racecheck pass. The speculative production barrier changes were withdrawn. Original failing logs, the minimal reproduction, supported-launch checks and the explicit assessment remain in `regressions.json.xz` and `provenance.json.xz`; instrumented runs are excluded from timing samples.

## Source and reproduction

Core SHA-256: `ea29370dcc36673a236dc9a82d72c2f87647b2820e0e88485b16c5b1f8de07ae`. All **12,786** archived source inputs were verified. Core and all ten ordinary/SPD stationary AOT library/manifest pairs come from the frozen source. Release/sm120, CUDA 12.9.1, GCC 13.3, verified sccache and compiler commands are retained with before/after cache statistics. Eight finite Slurm workers preserve device visibility. Raw source archives, binaries and logs remain in ignored local/n1 artifacts; compact records retain source/binary/Slurm identities, observables, work counts and controls.

Forty independent oracle receipts retain their original producer revisions. Reuse is limited to the audited identical scientific inputs and reference implementation; production calculations never consume oracle densities. The source selection, execution scripts, both failed and passing qualification cohorts, and input-reuse audit are retained in the publication.

```bash
PYTHONPATH=python:. python benchmarks/results/readme-final-20261024/reproduce.py \
  --output .artifacts/readme-final-replay
```

This recomputes every numerical gate, both original-layout 2×4 figures, and the full timing/work tables without running a GPU calculation.
