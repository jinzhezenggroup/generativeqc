# Explicit spatial bisection: water96 endpoints

Measured source: `f7aa8c87087608b2f07d63eb5cfe564312566c44`.
`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` selects longest-axis balanced bisection
with cuts aligned to the actual tile size and original-index ordering inside
leaves. It preserves every point/weight tuple. Unset/1 remains center grouping;
0 remains canonical order. Force ownership, nonlocal and fleet/resource fallback
are unchanged. This is an opt-in schedule, not a default promotion.

RTX 5090 on n1, CUDA 12.9.1, release sm120, one numerical host thread, finite
Slurm jobs, verified sccache 0.16.0. All 6,000 source inputs were authenticated
before compilation. Grid, basis, screening, convergence and precision controls
are unchanged. DF uses charge splits 32 in both variants. No occupied-XC,
one-electron schedule or later WB integration is included in this population.

Five alternating fresh-process selector 1/2 pairs per method/mode retain complete
cold energy/analytic-force endpoints and three frozen warm repeats. Each process
contributes its median warm time. All 272 native calls pass: 64 pilot, 160 paired,
32 cold/warm/moved/moved-warm and 16 PBE0/B3LYP compatibility calls. Pilots also
cover r2SCAN and M06-2X, Standard/Fast and both modes. Moved controls include fresh
PBE Direct GPU4 and exact-geometry previously retained GPU4 references; these
establish equivalence, not a new moved dense-grid convergence claim.

| Method | Mode | Center cold median ± MAD (s) | Bisection cold median ± MAD (s) | Gain |
| --- | --- | ---: | ---: | ---: |
| PBE | DF | 20.492138 ± 0.009666 | 19.692105 ± 0.040788 | 3.904% |
| PBE | Direct | 21.837323 ± 0.004424 | 20.952818 ± 0.011562 | 4.050% |
| r2SCAN-3c | DF | 43.473649 ± 0.023935 | 40.186811 ± 0.037872 | 7.561% |
| r2SCAN-3c | Direct | 49.284960 ± 0.025636 | 46.013717 ± 0.053441 | 6.637% |

All four cold gains exceed `max(2%, 2*(baseline relative MAD + candidate relative
MAD))`. PBE warm changes (1.49% DF, 1.71% Direct) fail that floor and are not
claimed as gains. r2SCAN-3c warm improves 2.763% DF and 2.168% Direct, passing it.
All remain separate from GPU4 timing populations and combined-candidate timings.

Native tests cover exact permutations, zero-tile refusal, independent CPU E/V,
changed-density replay, tail tiles, canonical force view and resource refusal.
Memcheck, synccheck and racecheck report zero errors/hazards. Complete Fock
composition passes. Binary hashes, main and supplemental test builds, cache
statistics, runners, complete E/F arrays and semantic work counts are retained
in `receipts.json.xz`; its manifest delta reconstructs and authenticates the exact
source inventory. The nine historical issue377-379-df endpoints are losslessly
compacted with exact-byte restoration to stay within unchanged storage caps.

```bash
PYTHONPATH=python:. python benchmarks/results/xc-spatial-bisection-20261023/reproduce.py
```
