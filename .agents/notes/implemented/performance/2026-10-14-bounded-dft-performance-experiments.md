# Decision: keep bounded DFT scheduling experiments behind measured gates

Status: implemented (opt-in experiments; endpoint qualification pending)
Date: 2026-10-14

## Problem

Issues #2475–#2479 require current-source attribution before changing defaults.
The pre-#2450 force profile cannot establish the current ranking, and #2464's
point-expression sharing does not optimize four-jet matrix contractions.

## Decision

Keep the ordinary sampled-AO map owner and add a bounded device compaction
control. Both schedules call the same generated AO producer and sampled-jet
predicate. Stable warp-prefix compaction writes increasing AO IDs to the
already reserved map and downloads only CSR offsets per chunk. Dead work and
point-total scratch hold flags and offsets; no additional device allocation is
admitted. Publication is transactional and every failed borrower is drained.

Add a diagnostic-only executed-work census rather than inferring integrals from
unscreened shell cardinality. It distinguishes candidate traversal, screening,
source admission and actual recurrence work. Incomplete coverage and missing
counter sites are explicit. An optional low/order-six diagnostic split repeats
enumeration; its class times are evidence about that intrusive run only.

Reuse the existing class-domain materialized dddd kernel for the dd×dd
acquisition experiment. Do not substitute the older streaming force algebra.
The order-seven experiment changes only page width (8/16 versus 32), retaining
all 256 coefficient lanes, generated Coulomb equations and shared barriers.
Requests are captured with prepared geometry. All defaults remain unchanged.

## Evidence available before qualification

Unmodified master `36574095876e4d12a3945b39cef72f6c77895f00` passes all 12
canonical r²SCAN-3c direct 48- and 96-atom E+force gates on n1 RTX 5090, Slurm
8368. The 96-atom clean warm and moved-warm medians are 95.380 and 92.741 s;
cold complete time is 463.117 s and moved time is 314.350 s, with 26/16 actual
SCF builds. All warm calls have one build. These are complete endpoint times,
including preparation, without iteration normalization.

A separate n4 Nsight Systems diagnostic of the same native image (Slurm 889)
returns cold and one warm. Its warm public call takes 97.093 s. Kernel times
are 28.016 s order seven, 26.187 s combined low/order six, and 5.766 s
materialized order eight. These intervals motivate experiments, not speedup
claims. The Python wall ledger is a separate additive view. NCU hardware
counters are unavailable: the finite Slurm 890 probe returned
`ERR_NVGPUCTRPERM`. Static registers/stack data must not be called measured
occupancy, stalls or atomic contention.

AO discovery has independent CPU E/V and exact sorted-ID tests, including
spherical/cartesian s/p/d/f, empty/tail/full maps, multiple ballot waves,
tile-size-one scratch, capacity refusal and failed preparation followed by
retry. The n2 real-GPU test and Compute Sanitizer memcheck pass with zero
errors. Initial 48/96 host/device endpoint comparisons show a large transfer
reduction but small discovery-time savings; they do not justify promotion.
Different observed moved-SCF counts remain part of the complete endpoint cost.

## Rejected alternatives and invariants

- Reimplementing dd×dp enumeration would duplicate the already merged #2175
  owner. Reuse its canonical unsigned physical-pair ordering.
- Fewer CTA lanes would omit generated coefficients; vary queue width instead.
- Summing host, device, API and overlapping parent intervals overstates wall
  time. Preserve separate ledgers and reconcile exclusive host intervals.
- A new full-grid host bitmap or shell-quartet inventory violates bounded
  ownership. Keep existing resource refusal and generic fallbacks.
- Neither small kernel improvements nor a warm-only pilot qualifies a new
  default. Retain the unchanged 1800-s whole-point limit, independent
  1e-8 Eh / 1e-7 Eh/Bohr gates, all 12 calls and actual SCF/Fock histories.

## Revisit when

Promote a direct scheduling candidate only with a robust >=5% complete
96-atom warm and moved-warm improvement and no >5% cold/moved or 48-atom
regression. AO discovery requires repeatable cold/moved benefit without warm
regression. Four-jet work requires a separately identified post-#2464 control;
do not re-credit that PR's CSE or silently mix its source with master timings.

## Four-jet experiment admission

The separately qualified #2464 image at `786c31a481ad6cd743c14e8b2920f3a89b5ef8c6`
(library `bc2b09850386e26b67b439173ba7ce0fd367a042765c1d6c0e3452a054761fe0`)
was profiled again on n4 for M06-2X/direct/96. One diagnostic warm call records
0.924 s in `batch_density_products` and 1.342 s in `batch_local_potentials`;
the native SCF/publication host interval is 6.218 s. The force geometry-point
kernel is a distinct 1.247 s. These identify matrix work after shared-branch
CSE rather than repeating its old point-expression attribution.

Expose the existing compiler-owned 8/16/32 matrix schedule candidates through
CMake, retaining 16 by default. The 32 candidate trades more shared-panel reuse
for a larger CTA and possible occupancy loss. It needs no second bilinear,
vendor provider, additional panel allocation or dropped tau lane. A matched
post-CSE build must decide this tradeoff with complete endpoints; legal static
resource bounds are not performance evidence.
