# DF force admission and refreshed HF/DFT endpoints (2026-10-25)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [HF/DF48 changes](changes.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **192 populations / 768 complete energy-and-force calls** were remeasured
at frozen source `2ed9b394a2889b3a373dc91ebbaaed03931056eb`: eight methods,
Direct/DF, 3/12/48/96 atoms, Standard/Fast and stock GPU4PySCF. Each fresh
process contains one cold and three frozen-density warm calls. The original
2×4 figure layout is unchanged; CCSD(T)/CBS is unchanged. No old timing enters
the updated curves. The HF improvements from the
[preceding cold-work repair](../hf-cold-work-20261025/README.md) are included.

## Cause and scoped control

The fitted native AO-jet bitmask producer inherited the older sampled producer's
173,946,175,488 dense point×AO² work threshold. Default-grid r2SCAN-3c at
48 atoms has 159,559,221,248 units of that work and therefore remained dense,
while 96 atoms used the existing indexed path. The independently qualified
`2**35` crossover admits the cheaper native producer without atom/method/product
special cases. The AO-jet cutoff stays `1e-16`; all grid points, force derivatives,
DF response, occupancy guards and resource/lifetime fallbacks remain present.

The clean control is source `fe1e306c1` (its full SHA is in every receipt),
with the same native/AOT binaries as the candidate. Only fitted force admission
changes. These controls are separate from the complete figure cohort and from
intrusive profiles; cold has one observation and warm is the median of three.

| Arm | Method, DF 48 | Control cold (s) | Candidate cold (s) | Control warm (s) | Candidate warm (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| fast | b3lyp | 7.722 | 7.386 | 3.109 | 2.641 |
| fast | pbe | 5.714 | 5.674 | 2.029 | 1.608 |
| fast | r2scan-3c | 11.521 | 10.120 | 5.317 | 3.628 |
| standard | b3lyp | 7.758 | 7.537 | 3.099 | 2.677 |
| standard | pbe | 6.449 | 6.202 | 2.314 | 1.850 |
| standard | r2scan-3c | 13.733 | 11.001 | 7.320 | 4.200 |

All 24 control/candidate comparisons have identical energies and Fock counts;
maximum force change is below **1.8e-13 Eh/Bohr**. These are latency observations
on separately allocated cards, not statistical significance claims. One-repeat
same-binary ablations, including the unprofitable 12/24-atom extensions and
36/42-atom neighbors, remain in `ablations.json.xz`; they do not enter the plots.

Actual selected point×AO² work falls from 79,503,556,608 to 19,268,241,696 for
PBE/B3LYP, and from 159,559,221,248 to 39,370,637,056 for r2SCAN-3c. These are
semantic work counts, not measured hardware FLOPs. `profiles.json.xz` retains
the component regions and producer/resource counters. Per-shell kernel detail
is outside this admission change and remains in ignored raw traces, identified
by their hashes. FP32 geometry has no intrusive profiling ABI: the original
refusal is retained, and successful Fast records use unprofiled counters where
that consumer is selected. Profiling times never enter the headline curves.

## Observed comparisons

| Mode | Phase | Standard wins / measured pairs | Fast wins / measured pairs |
| --- | --- | ---: | ---: |
| direct | cold | 28 / 32 | 30 / 32 |
| direct | warm | 30 / 32 | 31 / 32 |
| df | cold | 24 / 32 | 25 / 32 |
| df | warm | 27 / 32 | 30 / 32 |

Standard and Fast outperform GPU4PySCF in **57/64** and **61/64** measured warm
cases. Every completed performance point is shown, including accuracy failures.
Slower points remain visible; this does not establish universal leadership.

## Accuracy and regression validation

Each profile has **58 qualified / 6 failed / 0 unverified** populations, exactly
the prior README outcome. All 768 samples are checked against the retained
independent references: energy ≤1e-5 Eh/atom, maximum force ≤1e-4 Eh/Bohr and
RMS force ≤3e-5 Eh/Bohr, with quarter-gate reference-grid stability and CPU/GPU
spots. No auxiliary basis, grid or tolerance changed to make a failed point pass.
Original reference source identities and the audited reuse manifest are intact;
production endpoints do not consume reference densities.

**263 related tests pass**, including independent matched-grid LDA/PBE/PBE0/
r2SCAN RKS/UKS energy and all-force-component oracles, cold/replay/moved/replay,
zero-map-budget and occupancy-declined dense fallbacks, and AO derivatives of
orders 0–3. The initial 5 failures were missing LDA/UKS packaged test artifacts;
258 tests passed then, and the missing 5 passed after building their matching
AOT libraries. Original failure records and compiler-cache statistics remain in
`regressions.json.xz`. The benchmark library bundle was not modified.

All **18 moved-geometry/property checks** pass against independent GPU4PySCF,
covering Standard/Fast PBE, B3LYP and r2SCAN-3c, including energy-only-to-force
replay. The separate matched-grid tests retain the tighter 1e-8 Eh / 3e-7 Eh/Bohr
oracle gates. README qualification remains limited to its sampled static inputs.
The 52 publication/protocol regressions and compiler/default-policy audits pass.

## Provenance and reproduction

The frozen candidate archive's **12,733 files** match both the measurement
checkout and actual incremental-build inputs. CUDA 12.9.1, sm120 Release, GCC
13.3, verified sccache 0.16.0, actual launcher commands, before/after cache
statistics, core/AOT hashes and finite Slurm allocations are retained in
`provenance.json.xz`. Seven independent benchmark workers preserve Slurm GPU
visibility and use one math thread; the controls and additional device tests
also use finite allocations. `jobs.json.xz` retains exact execution scripts,
commands, worker assignments and outcomes. Raw source archives, binaries and
profiler logs remain in ignored local/n1 artifacts.

```bash
PYTHONPATH=python:. python benchmarks/results/df-dft48-work-20261025/reproduce.py \
  --output .artifacts/df-dft48-replay
```

This regenerates both original-layout figures, all independent numerical gates,
timing/work tables, control comparisons and scoped HF/DF48 changes. The latter
uses the prior retained README cohort without relabeling its source or timings.

## Remaining work

Standard r2SCAN-3c DF48 still slightly trails the measured GPU4 endpoint, while
Fast is faster in this cohort. Some 96-atom cold endpoints, including HF, also
remain slower. Intermediate 36/42-atom r2SCAN-3c diagnostic endpoints expose a
separate DF integral-response cost (about 4.51/8.13 s versus 1.12 s at 48 atoms);
this work-policy repair does not resolve that response-planning discontinuity.
Those observations are retained for a focused follow-up.
