# Decision: retain six bounded MP2 materializations after matched alternatives

Status: implemented
Date: 2026-11-06

## Problem

Issue #1626's six MP2/oracle findings needed actual final decisions, not another
scanner inventory. The three historical RCCSD(T) projections are already gone.
The earlier [second-stage compaction](2026-11-06-compact-factorized-mp2-second-stage.md)
reduces occupied-support storage/work but by itself does not decide all sites.

## Decision

Retain the following materializations under their current resource and reuse
contracts. These are measured choices, not mathematical claims that no future
fusion can work. No new mandatory source-spelling/hash review gate is installed.

| Site | Final disposition | Measured reason |
| --- | --- | --- |
| Dense first shell transform | Required reusable storage in the explicit dense path | Lazy first evaluation passes physics, but the matched dense-oracle endpoint changes water warm E+F only 0.045583→0.045398 s; the component repeats first work fivefold. No justified general promotion. |
| Factorized first shell transform | Retain occupied-support compact storage | Lazy evaluation lowers the N=5/O=2 shell peak 3,360→2,448 bytes, but repeats all populated first contributions fivefold. Its matched water cold complete endpoint is 0.550464→0.550833 s, despite a small warm difference. |
| Second shell transform | Retain reusable storage, compact third-MO occupied support in factorized production | Dense lazy-second evaluation lowers peak 6,528→4,728 bytes while second accumulations rise 3,125→15,625. A sparse/factorized case can do less extra work; that is not a general work bound. Matched water cold E+F regresses 0.551643→0.553961 s. |
| Third shell transform | Retain reusable storage across fourth-shell consumers | Lazy-third lowers dense/factorized peaks 6,528/3,360→5,448/2,280 bytes, but third work grows fivefold in the populated component; matched water cold E+F regresses 0.550623→0.553906 s. |
| Shell-local cotangent | Retain bounded contiguous callback materialization | The callback borrows a contiguous shell tensor for its invocation. Reusing a maximum-capacity local vector cuts component allocations 125→47 but does not remove the tensor, its work or simultaneous peak. Matched water cold/warm improvements are small (0.548524/0.153259→0.547091/0.152119 s); retain the simpler per-quartet lifetime without a qualified broad win. |
| Dense orbital RHS / Lagrangian two-electron output | Retain only the explicit pre-allocation-budgeted validation oracle | The dense complete endpoint is faster on these tiny systems (water warm 0.157246→0.045584 s), but requires full rank-four ERIs/cotangents and is not the work/storage-bounded streamed production contract. Keep overflow/capacity rejection before allocation; never silently route production through the CPU/reference oracle. |

## Rejected alternatives

The experimental helper itself alters loop organization. Production versus that
helper retaining *all* stages changes water warm timing 0.157382→0.153746 s.
Therefore comparing every alternative only to production would falsely credit
common helper work as a materialization win. Each alternative instead has its
own matched ABBA retention control; dense-first has a dense-helper control too.

An initial vector-reuse implementation grew capacity incrementally. Its old and
new allocations overlapped and violated the existing numeric staging bound in
case 2. The measured alternative reserves a checked maximum before borrowing
the callback view; both the rejection and corrected generated source are kept.
Do not restore incremental growth without charging overlapping capacity.

## Invariants

- The generic compiler materialization request remains advisory; MP2 ownership,
  algebra, precision and representation decisions stay with MP2.
- No reassociation, precision relaxation, changed production force equation,
  mandatory source fingerprint, or extra CPU oracle dependency.
- Preserve dense-oracle overflow/pre-allocation gates, factorized compact support,
  independent physical derivatives, callback lifetime and bounded staging.
- Timings cover preparation+cold E+F and complete warm E+F, not isolated kernels.

## Evidence

[Raw source/commands/results](../../../../benchmarks/results/mp2-six-site-20261106/README.md)
retain six variants across 384 direct native component invocations and 288
physical libcint derivative comparisons, plus nine relinked libraries and 64
fresh-process, CPU-pinned, single-threaded ABBA histories. All 384 timed and 54
separately memory-instrumented complete E+F calls pass independent PySCF/libcint:
maximum energy error 4.94e-10 < 1e-9 Eh; full force error 2.33e-8 < 2e-6 Eh/Bohr.
H2/LiH/water use the exact native STO-3G primitives, not differently rounded
PySCF basis tables. All representations, sparse populations and high-angular
component fixtures remain covered.

Tracked process-heap high-water is 55,057,817 bytes for every matched variant,
dominated by imports; it establishes no endpoint-memory improvement. Exact
simultaneous shell allocation peaks and executed accumulation counts are a
separate, more sensitive component measurement. Neither is described as RSS,
GPU memory or a general complete-endpoint speedup.

The retained Memray files use aggregated high-water allocation records, not a
chronological event log, with native C stack capture disabled. Full native-stack
pilot streams remain in ignored local artifacts; their different profiler
configuration is not mixed with the final matched summaries.

## Revisit when

A consumer accepts non-contiguous/lazy cotangents without repeated source work;
a larger representative workload shows a reproducible complete-endpoint win;
or a bounded production algorithm avoids the dense oracle's rank-four storage.
Retain matched helper controls and independent scientific/resource gates.
