# PBE0 48-atom cold follow-up: interleaved fresh-process controls

This packet addresses the original single-population cold observation
23.372648→24.088161s (+3.06%) for #2549 relative to #2541. That historical
measurement is retained, not replaced or discarded. The full qualified
48/96-atom r²SCAN-3c, PBE0 and HF packet remains at immutable commit
`631f0b003884094f758a349e4a0418f6b54eec6a`, same publication path; the earlier
master/incumbent packet remains at `4eb9fd859a69a31fa21c5fbbe9b13f7d36a9d90a`.
Only this task's own current publication is superseded; storage budgets and
unrelated evidence are unchanged. No Release, tag or external archive is made.

## Protocol and scope

There is **no production/compiler change** in this follow-up. Sources remain
incumbent `4bdffe3ceaa57e7a201f76a5687ad2d1933e6f1e` and response
`00fced59f8d083839e4bc0761f91a1e94e8ccc0b`, with the exact previously qualified
Release/sm_120 libraries and native identities verified for every population.

All primary runs share one scheduler-assigned RTX5090 on n1, within a finite
25-minute Slurm allocation. Ten independent Python processes run in the
predeclared order **A B B A B A A B A B**, five per source. Every process retains
the unchanged cold + five warm + moved + five moved-warm E+F sequence, original
1800-second process timeout, def2-SVP basis, moving48×16×32 grid per atom,
strict FP64, screening/convergence, reference inputs and actual build counts.
No process is prewarmed, no cache is cleared/disabled and no sample is trimmed.
Cold includes preparation exactly as in the original benchmark.

The previous3.06% difference was a valid single observation, not proof of a
repeatable source-induced regression. This follow-up tests reproducibility;
it does not claim an additional runtime optimization or universal performance
promotion. GPU clocks/temperature/power were recorded separately without
changing clock/power settings. Full raw native histories and monitoring remain
hash-bound in the ignored campaign directory.

## Complete endpoints

| Endpoint | Incumbent median s | Response median s | Change |
| --- | ---: | ---: | ---: |
| cold | 23.425425 | 23.371673 | -0.229% |
| warm | 4.274105 | 4.214359 | -1.398% |
| moved | 22.087280 | 22.112491 | +0.114% |
| moved-warm | 4.278988 | 4.225750 | -1.244% |

Warm entries are medians of the five per-process five-call medians. All25 individual calls per source/phase remain retained; positive changes are regressions.

| Pair | Incumbent cold s | Response cold s | Change |
| --- | ---: | ---: | ---: |
| 1 | 23.141314 | 23.150135 | +0.038% |
| 2 | 23.432668 | 23.371673 | -0.260% |
| 3 | 24.132459 | 23.399186 | -3.039% |
| 4 | 23.420661 | 23.371244 | -0.211% |
| 5 | 23.425425 | 23.390243 | -0.150% |

No repeatable3.06% source regression is observed. The cold median difference is effectively flat; it is not claimed as a statistically significant optimization. All120 primary calls complete and independently pass original-reference gates. Maximum errors are 7.503331290e-12Eh /2.492095419e-11Eh/Bohr. Every process retains cold17/18 and moved12/12 SCF iterations/physical Fock builds; all warm calls1/1.

The original cold gap is0.715512s. Its exclusive force `python_packing` owner changes0.067221→0.720458s:0.653236s, about91.3% of the total gap. That owner includes host task preparation and resident AO-map setup; the receipt does not isolate a CPU, GC, cache, driver or scheduler cause. Its five new cold samples return to0.065–0.152s on response, and0.067–0.080s on incumbent. The original sample is retained, but is not established as a stable order-six arithmetic regression.

## Diagnostic accounting

Separate intrusive cold/first-warm captures use the same assigned GPU UUID for
both sources. They are not substituted for clean endpoint timings. Published
profiles retain mutually exclusive host owners, separate device windows and
kernel device sums. Never add kernel sums to host waits or parent scopes.

The original warm-only SQLite reducer rejects an all-call capture with two
public roots; the raw incumbent capture was complete and retained. The selected
reducer works on temporary copies, removing only the other public root marker,
without changing any kernel/API event or selected call's nested scopes. Both
original SQLite hashes and exact selected root bounds are retained. The
response capture completes in a subsequent finite Slurm allocation, with the
same GPU UUID checked before reusing the incumbent capture. This diagnostic
reduction issue does not interrupt any primary population.

| Cold diagnostic device window | Incumbent s | Response s |
| --- | ---: | ---: |
| cuda_ks_exact_jk_shared | 3.841285205 | 3.845416478 |
| cuda_ks_xc_shared | 6.662236798 | 6.632718059 |
| direct_force_low_and_order6 | 0.889814033 | 0.837128539 |
| direct_force_order7_cooperative | 0.072384245 | 0.072169014 |
| direct_force_order8_materialized | 0.179144627 | 0.179311312 |

All four diagnostic calls also pass independent E/F gates; host/device reconciliation errors are below1e-7s. The generated-low/order-six device window improves, rather than showing the historical cold spike. These intrusive captures are supporting attribution only. Forty-four focused profile/benchmark host tests pass.

## Independent checks and replay

Every primary and diagnostic energy/force array and both independent geometry
oracles are retained losslessly in standard NumPy/XZ. Per-call timings, counts,
gates and metadata are in `numeric-index.json.xz`; per-process summaries and
all five cold pairs are in `endpoint-summary.json.xz`. Run
`python verify-arrays.py` to independently recompute every published gate;
`--restore-references` reconstructs the original oracle JSON inputs.

`reproduction.json.xz` retains the exact executed orchestration, independent
analysis and per-call profile selector as ordinary UTF-8 sources, plus selected
raw Slurm/cache/test/diagnostic receipts and hashes of full native inputs.
The executed first runner contains the documented warm-only reducer mismatch;
for an uninterrupted replay, replace its final `summarize_dft_profile` invocation
with `python summarize-calls.py <profile-directory>`. The primary commands and
GPU captures remain unchanged. The follow-up `profile.sh` reuses an existing
incumbent capture only on the identical assigned UUID.

Restore the original sources/build environment using `build.sh` and the
immutable predecessor packet. On n1 the campaign root is
`/data/jzzeng/gqc-direct-next-20261016`; adapt that root and the cluster Python
wrapper paths for another installation. Extract the `sources` entries of
`reproduction.json.xz`, then run the exact finite Slurm argv in `publication.json`.
Every real-GPU replay must preserve Slurm-assigned `CUDA_VISIBLE_DEVICES`.

No method-name workaround, screening/tolerance change, force approximation or
default-path change is introduced. The original qualified r²SCAN improvements
remain at the immutable predecessor; both implementation PRs remain unmerged.
