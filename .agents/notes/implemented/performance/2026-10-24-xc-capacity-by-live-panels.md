# Decision: size optional XC batches by the widest retained jet panel

Status: implemented
Date: 2026-10-24

## Problem

GGA retains four AO jets but contracts only one density-product jet. The previous
32-MiB-per-density-product-jet allowance therefore gave it one quarter of the
optional capacity of meta-GGA, although its retained AO panels have the same
width. Water-96 PBE0 has 4,213 original SCF point tiles; the 32-MiB allowance
admits eight at a time, causing 527 batch submissions per Fock. Preparation,
full analytic forces and export remain substantial independent costs.

## Decision and invariants

Size the ordinary optional allowance by `max(AO jets, density-product jets)`:
32 MiB when both are one, 128 MiB when either is four. This is a panel-shape
rule, independent of method name, molecule and device. Keep the existing
compiler planner, 32-tile request, local maps, ordered sums and all scientific
settings. Optional allocation rejection first retries 32 MiB, then retains
one-tile execution. Explicit byte caps remain exact. Public resource ledgers
continue to reserve no optional batch arena; unused ledger capacity cannot
fund this path or change later force/fleet reservations.

Opt-in native traces report selected batch tiles/bytes separately from point-AO
visits and AO-square work. Those are plan facts and logical work, not measured
hardware operations; they add no CUDA counter, launch or fence. Larger batches
change the retained arena and launch supply, not the original AO/point domain.

## Alternatives and qualification

The same-image three-pair pilot changes only the existing explicit byte cap.
PBE0 cold medians are 21.9182 / 21.2471 seconds, both nine SCF/Fock builds;
warm changes remain below the acceptance floor. All calls pass the previously
audited independent dense-grid energy/force gates. The separate intrusive
profile reduces compact point/contraction submissions from 5,270 to 1,320
across cold plus warm, with unchanged 46,343 four-jet AO launches. These are
mechanism measurements, not clean endpoint timing.

Dense-factor materialization was considered, but the current PBE0 profile
spends only about 0.54 GPU seconds in all ten density contractions, limiting
its endpoint value. Repeating force AO-map discovery in chunks was already
rejected in the previous campaign because its host-time reduction mostly
moves waits rather than eliminating endpoint work. Neither is bundled here.

The original diagnostic baseline is master `72e0a7e5a`; the branch base is
`48918670f`. The intervening native/build changes affect CLI implementation
and CLI test registration only. No relevant DFT runtime/compiler code differs.
The original baseline binary and source identities are retained rather than
relabeled as the newer checkout.

## Complete qualification

Three rotating fresh-process rounds retain the full constructor/preparation/SCF/
analytic-force/host-export endpoint. PBE0 cold medians improve from 21.6781 to
21.1493 seconds (2.44%); PBE improves from 17.7381 to 17.1827 seconds (3.13%).
Both exceed the predeclared 2% floor. PBE0 remains 0.8400 seconds / 4.14% behind
stock GPU4PySCF. Native cold Focks remain 9 and 11, warm 1; warm improvements
remain below the floor. Keep the first 23.1677-second PBE0 baseline sample.

All 94 static and three moved E/F calls pass the unchanged independent gates.
Nine held-out configurations retain small-size, DF, Fast, B3LYP and meta-GGA
controls; a single small PBE0/3 cold regression does not establish broad
profitability. Host resource/fallback tests, native point/ragged/spatial-grid
suites, independent force/finite-difference/lifecycle tests, all 32 public
native force tests and memcheck/synccheck pass. Final trace facts retain the
same 190,426,944 point-AO visits and 35,027,345,728 AO-square sum per Fock;
selected arena bytes rise 27,202,568 -> 88,317,344, within their 32/128-MiB caps.

Qualification also exposed two pre-existing test assumptions. A native force
owner need not create a Python executor; test public failed-neighbor isolation
against a fresh same-geometry owner instead of exact executor calls. PySCF's
`radii_adjust` setter invalidates installed coordinates/weights: configure the
native Treutler partition first, then install the exact quadrature and assert
it survives SCF/gradient. Both failures were reproduced on frozen master; no
production numerical change or weaker gate was needed. Keep these diagnoses
to avoid mistaking them for future capacity regressions.

## Current-master composition and archive timestamps

The earlier source-matched cohort remains a separate historical measurement.
Before opening the PR, #2676 and #2743 had landed, so the branch was rebased on
`71504dc55` and a separate paired cohort was required for the composed default.

The first composition baseline was invalid: extracting an older Git archive
into an incremental build preserved older source mtimes, so Ninja retained the
previous candidate object and source identity. Binary SHA/source-identity checks
caught it before acceptance. Stop the owned job and quarantine its seven
complete records and interrupted empty arm, including all returned vectors.
Do not pool them with either valid cohort, even when their independent E/F
checks pass. The final candidate had already rebuilt all inputs; exact source
inventories show only `cuda_ks.cpp` differs from current master. Restore master
bytes, force that translation unit to rebuild, and verify the loaded identity
against a fresh digest of the exact scientific source inventory before rerunning.
The valid current-master three rounds give PBE0 21.6290 -> 21.3102 seconds
(1.47%, below the unchanged 2% floor), with a 0.9281-second gap to GPU4PySCF.
A separate prespecified five-pair confirmation gives 21.6430 -> 21.2241 seconds
(1.94%, also below the floor). Keep both inconclusive current PBE0 outcomes;
current PBE cold improves 2.56% above its 2% floor. All 156 source-qualified
E/F calls and all 14 numerical returns from the quarantined attempt pass their
independent scientific gates. Current validation adds 34 GPU force/occupied
tests, 147 host cases and zero-error memcheck/synccheck.

Compiler caches need not be cleared: source restoration timestamps, rather than
incorrect cache contents, caused the reuse. Prefer isolated build trees or
explicit invalidation when restoring backdated source archives.

## Revisit when

Use other devices or resource shapes to refine optional admission. More arena
capacity costs about 58.3 MiB per owner on the measured case. Keep the exact
byte override, 32-MiB retry and single-tile fallback. A faster force/setup
consumer may become the next endpoint bottleneck; launch reductions alone do
not establish an improvement when those costs dominate.

## References

- [Earlier density-jet capacity policy](2026-10-15-xc-point-capacity-by-work-shape.md)
- [Complete retained campaign](../../../../benchmarks/results/pbe0-ao-batch-capacity-20261024/README.md)
- [Current batching contract](../../../../docs/developer/xc_native_cuda.md)
