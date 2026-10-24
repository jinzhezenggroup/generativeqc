# Default-grid README comparison (2026-10-22)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **192 populations / 768 complete energy-and-force calls** use clean source `1178a03c7f45074cc705a7accd676f0ffc1b6d2a` on master `7522994e2`, which was current when frozen. Eight methods, Direct/DF and 3/12/48/96 atoms are remeasured for native Standard/Fast and one stock GPU4PySCF SCF profile. Only cold and three frozen-density warm calls are timed. CBS is unchanged.

## Observed comparisons

Times include the complete endpoint on n1 RTX 5090. A win is a lower observed latency among independently qualified pairs; small differences are not significance claims. Cold has one observation, and warm reports the median and range of three calls within one process. Slower valid points remain visible.

| Mode | Phase | Standard wins / qualified pairs | Fast wins / qualified pairs |
| --- | --- | ---: | ---: |
| direct | cold | 18 / 31 | 24 / 31 |
| direct | warm | 26 / 31 | 27 / 31 |
| df | cold | 12 / 27 | 12 / 27 |
| df | warm | 18 / 27 | 18 / 27 |

Standard and Fast are faster in **44/58** and **45/58** qualified warm comparisons, respectively. Direct ωB97M-V/96 warm speedups are **3.89× Standard** and **4.14× Fast**. The results do not show universal leadership over GPU4PySCF.

## What changed and what still costs time

The streamed native DF Hermite owner removes repeated raw three-center materialization and directly contracts charge, J and force response. Default-grid DF Fast admits separately qualified XC proposals while keeping J and metric work strict. Final-state-aware Standard stopping avoids an extra conservative sweep when the unchanged physical/canonical proof already passes.

At r2SCAN-3c DF/96, the previous complete Standard cold/median-warm result was **1648.702 / 173.968 s**; this campaign measures **69.223 / 15.857 s**. Fast changes from **1633.010 / 172.075 s** to **49.237 / 14.974 s**. Historical values are explicitly before/after implementation observations, not same-source paired timings. Current GPU4PySCF is **26.210 / 12.768 s**, so a remaining gap is still visible.

Native DF PBE/r2SCAN/3c cold iterations are 14/14/15 for Standard and 9/9/10 for Fast, with one iteration on every warm replay. All **256 Fast calls execute exactly one solve**; no extra precision audit or density-restoration solve enters them. Standard keeps its stricter stopping contract. Direct PBE0/96 Fast and GPU4PySCF both take nine cold iterations, yet complete times are 29.272 / 20.321 s: preparation and per-step/force costs still matter. Delta-D density stopping was already enabled; incremental J/K is a separate operator-reuse setting.

Five qualified Fast/Standard phase comparisons are more than 5% slower: three-atom DF B3LYP cold/warm, DF PBE cold, DF PBE0 warm, and 12-atom Direct HF cold. B3LYP/PBE0 at three atoms use the strict Standard fallback; the two B3LYP warm ranges overlap. A separate same-GPU instrumented B3LYP check finds virtually equal native batch/force work, with additional small Fast host-policy overhead. HF/12 cold and PBE/3 cold each have one observation. These points remain unchanged in the figures; diagnostic profiles never replace official timings.

## Accuracy and validation

| Profile | Attempted | Complete | Qualified | Failed accuracy | Unverified |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 64 | 64 | 58 | 6 | 0 |
| fast | 64 | 64 | 58 | 6 | 0 |
| gpu4pyscf | 64 | 64 | 58 | 6 | 0 |

Common gates remain energy **1e-5 Eh/atom**, maximum force **1e-4 Eh/Bohr**, and RMS force **3e-5 Eh/Bohr**. Every cold/warm sample is independently checked; reference grid stability and CPU/GPU spots use quarter gates. All four HF-DF sizes fail the direct-reference fitting-error gate, and M06-2X/3 fails the default-grid force gate in both modes, for all three profiles. These 18 populations remain retained and unranked. No grid, auxiliary basis or threshold is changed to remove a failure.

The frozen final binary additionally passes all **24 default/rollback geometry tests / 96 native E/F calls**: PBE spherical/Cartesian orbital and r2SCAN spherical models, matching spherical auxiliaries, native/tighter controls, original/displaced geometry and frozen replay. Native calls exclude CPU derivatives. Earlier source-bound qualification retains 178 independent angular tests, 16 integrated metric/force tests, clean memcheck/racecheck, and combined Fast-path memcheck. Host selection/lifetime/precision checks pass 525 tests; the benchmark driver/qualifier passes 47. The separate geometry job initially stayed pending with Slurm `InvalidAccount`; it was canceled before execution and successfully resubmitted after the full timing campaign.

## Source and reproduction

Core SHA-256: `05bbb70e37fc8836216fca394b2d84e5c6676a6f08820e7937073cc449f63f14`. All **12,199** archived source inputs were verified before execution. Core and all ten ordinary/SPD stationary AOT library/manifest pairs come from that source. Release/sm120, CUDA 12.9.1, GCC 13.3, FastCompile OFF and verified sccache 0.16.0 are retained with compiler commands and before/after cache statistics. Runtime source, binaries and full raw logs remain in ignored local/n1 artifacts; compact receipts retain exact source/binary/Slurm identities, observables and controls.

Forty original independent oracle receipts retain their producer revisions and audited reuse contracts. The [previous campaign](../readme-repaired-20261021/README.md) remains recoverable from its verified published Git snapshot. No old timing is mixed into the current curves.

```bash
PYTHONPATH=python:. python benchmarks/results/readme-final-20261022/reproduce.py \
  --output .artifacts/readme-final-replay
```

This recomputes every numerical gate, both 2×4 figures and the full timing/work tables without running a GPU calculation.
