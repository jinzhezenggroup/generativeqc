# Decision: retain scalar DIIS order while parallelizing transport

Status: implemented
Date: 2026-10-23

## Problem

After the public-AO Schwarz fix, Direct DFT still spent substantial time in a
single-warp DIIS update. A PBE0/96 Nsight trace on the first PR's binary measured
0.853269 seconds across 12 updates. This includes vector history insertion,
repeated old-old residual dots, the small solve and coefficient extrapolation.
Kernel timing alone is not the complete E/F acceptance criterion.

## Decision

For KS owners with at least 512 AOs and history capacity at least two, admit
one separately charged physical-slot Gram cache. A CTA per pending entry stages
contiguous residual operands in 256-element tiles. One lane continues the
compiler-generated increasing-element FP64 accumulator through every tile.
The continuation argument is not a sum of independently reduced partial dots.
A following full CTA copies histories and computes independent output elements,
keeping each element's generated coefficient order. Lane zero still owns the
augmented solve, normalization and dependent-history retirement. Uniform
broadcasts and CTA barriers replace warp synchronization only in this route.

The cache costs 8 * history_capacity^2 bytes per system; staged operands use
4 KiB of shared memory per CTA. No additional resident matrix or host/reference
consumer is introduced. First insertion skips the unused diagonal; insertion
two establishes it. Cached physical slots survive retirement and wraparound;
a count/head reset may leave cache bytes uncleared because newly live entries
are recomputed before use. Inactive or invalid histories preserve the existing
reset/fallback policy. Launch failures stop before the following mutation.

Small owners and ordinary HF retain their existing default. Explicit
`GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM=0` restores the serial route. Explicit
`1` retains the prior cooperative reducer unless its subordinate reduction
selector is `ordered`. Resource queries and construction share admission and
validate active selectors before allocation. A KS plan retains its chosen
reducer; it cannot reinterpret cached entries after an environment change.

## Rejected alternatives

Automatic cooperative reduction changed the floating-point sum order. Although
all independent energy/force gates passed, PBE/96 cold iterations changed from
11/11/11 to 11/16/13, and r2SCAN-3c/96 from 10/10/10 to 13/10/10. Corresponding
PBE cold times were 20.596/20.623/20.728 versus 19.845/23.878/21.515 seconds;
3c was 45.595/45.786/45.709 versus 50.606/44.321/44.301 seconds. This is not a
reliable default improvement. PBE0's isolated improvement cannot justify it.

The existing ordered incremental path kept all r2SCAN-3c trajectories at ten
iterations but reduced its median only from 46.202 to 45.627 seconds (1.24%).
This is below the predeclared 2% minimum robustness floor.

Staging operands while retaining the single-warp update preserved iterations
and reduced three-round medians by 2.62% (PBE), 1.72% (PBE0), and 2.43% (3c).
PBE0 does not meet the 2% floor. Its separate profile attributed 0.134488 seconds
to staged dots and 0.402713 to the remaining update, motivating the full-CTA
transport rather than changing DIIS numerics. These measurements precede the
latest master integration and are not evidence for its changed J/K defaults.

## Qualification boundaries

The older staged variant passed 111 focused host tests and real-GPU bitwise
state/output comparisons at 512/513 AOs, one/two spin blocks, normalized and
unnormalized windows, wraparound, retirement, resets, and poisoned inactive
systems. An independent long-double oracle checked raw Gram values. Filtered
memcheck and synccheck on the new pending-row kernel reported zero errors.
An earlier unfiltered sanitizer run of the old comprehensive history-64 suite
timed out at ten minutes; that run is not a pass.

The final candidate adds full-history-eight and large-invalid-state cases.
Both comprehensive native DIIS suites and all large cases pass; state/output
bits match the serial owner and raw Gram values pass the independent oracle.
Filtered memcheck and synccheck cover both pending-row and wide-update kernels
with zero errors. The integrated host selection passes 189 focused checks;
five public-Schwarz receipt-reader checks also pass.

Three alternating rounds on the integrated baseline reduce cold E/F medians
from 20.839021 to 19.895562 seconds (PBE, 4.53%), 24.073740 to 23.507947 (PBE0,
2.35%), and 46.060636 to 44.163734 (3c, 4.12%). All exceed the predeclared
`max(2%, 2 * (baseline MAD / median + candidate MAD / median))` floor. Every
cold trajectory retains 11/9/10 iterations and reported Fock builds respectively.
Warm medians change by +0.02%/-0.01%/+0.77%; no warm speedup is claimed.
A separate PBE0 profile measures DIIS at 0.815841 seconds before and 0.180795
after (0.133003 pending dots + 0.047793 update), across twelve updates including
three warm calls. Intrusive profiles are excluded from clean timings.

All 130 integrated cold/warm/moved E/F calls pass independent references,
including Standard 3/12/48-atom, Fast 96-atom and DF 96-atom holdouts. The Fast
3c/96 strict paired-force diagnostic still fails at 3.442e-8 against 1e-8 while
passing its independent scientific gate. The first Schwarz campaign separately
recorded comparable same-binary variation; its cause is not established. Kernel
bitwise parity is not a claim of bitwise complete molecular endpoints.
The 200 exploratory calls and eight final profile calls also independently
qualify; rejected trajectories remain separate, never pooled into performance.
Compact qualification and the rechecker are in
`benchmarks/results/ordered-diis-cold-20261023/`. Exact, losslessly compressed
receipt/source bundles remain at the recorded ignored raw locations, bound by
checksums in that summary; they are not duplicated into the nearly full tracked
evidence budget.

The previous manifest and complete samples remain separately retained under
`n1:/data/jzzeng/direct-dft-cold-next-20261023/`; the new integration campaign
uses `n1:/data/jzzeng/direct-dft-cold-integrated-20261023/`. The baseline includes
PR #2702's Schwarz fix plus master `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`;
its guarded cold incremental J/K and prepared block-screening changes were
qualified together with the new schedule. The benchmark driver revision denotes
this common base, not an unmodified candidate checkout. Do not pool old and integrated runs.

## Revisit when

Broaden size/hardware/functional coverage only with complete cold/warm/moved
E/F endpoints, independent references, unchanged final physical validation,
actual semantic work and bounded resource refusal. Parallel arithmetic needs
its own trajectory/endpoint qualification; bitwise transport parity does not
establish universal scientific or performance equivalence of other reducers.
