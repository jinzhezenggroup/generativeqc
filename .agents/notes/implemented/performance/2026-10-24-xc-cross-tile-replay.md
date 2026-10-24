# Decision: defer FP64 cross range checks to the point-tile boundary

Status: implemented
Date: 2026-10-24

## Problem

After the public-AO Schwarz and ordered DIIS changes, Direct r2SCAN-3c/96
still spent substantial time in XC contraction. A cold-plus-warm profile of
`c1243a9d6ca693d7cb4058605ce1e191e2d8ce22` attributed 5.77 GPU seconds to
`batch_local_diagonal_potentials<double>`, including its cross terms. The
strict-double path checked the accumulator after every pair of nested FMAs.
Those checks preserve a real finite-domain requirement: a large accumulated
value can overflow on the first cross leg before its opposite leg cancels.

## Decision

Keep the exact ordered nested FMAs but save the incoming accumulator and
check only the end of each shared-memory point tile. A nonfinite accumulator
cannot recover to a finite value through subsequent FMAs. Consequently a
finite tile result proves that none of its earlier accumulator values needed
the old fallback. Otherwise replay that tile from its saved accumulator,
using the original per-term test and grouped two-product fallback.

This is a bounded scheduling change to the existing generated contraction,
not a new scientific equation. A failed tile performs at most one extra tile
pass. No shared-memory layout, allocation, active-domain policy, precision,
grid, screening, or solver policy changes. The FP32/mixed branches and the
range-sensitive diagonal fallback retain their existing expressions.

## Rejected alternatives

- Removing overflow handling loses finite cancellation cases and is incorrect.
- Reassociating the cross sum, splitting it into independent accumulators, or
  changing precision creates a new rounding contract that this optimization
  does not need.
- Claiming the 27–65% isolated contraction reduction as an endpoint gain hides
  the large integral-force and preparation costs. Complete E/F timing is the
  acceptance evidence.
- Repeating clean endpoint samples until favorable results appear is not a
  valid measurement protocol. The first PBE candidate cold sample was slower;
  it remains in the prespecified three-round population and its range.

## Invariants

Preserve the original per-output FMA order on finite tiles and replay every
nonfinite tile from its original incoming accumulator. Preserve both signs,
cross-leg orientation, point and AO tails, sparse public AO maps, and spin
channels. A kernel-level bitwise comparison does not imply endpoint bitwise
equivalence: independent block reductions and nonlinear SCF trajectories must
still be qualified at the observable boundary.

## Evidence

The [frozen report](../../../../benchmarks/results/xc-cross-replay-20261024/README.md)
binds the build, all populations, references and host-only rechecker. Three
rotating fresh-process rounds on RTX 5090 reduced Direct/96 cold medians by
8.09% (PBE), 5.87% (PBE0), and 3.50% (r2SCAN-3c). Native cold iteration/Fock
counts in that population remain 11/9/10 and warm counts remain one. The
paired intrusive 3c profile reduces the potential kernel from 5.7735 to
4.0725 GPU seconds over the same 2,904 launches; density contraction and
integral-force time remain essentially unchanged. These profiles are excluded
from clean endpoint statistics.

The CUDA numerical suite passes 144 cases, including 36 new cancellation
cases at and across tile boundaries. Memcheck and synccheck each pass 64
selected cases without errors. Four native suites cover point/ragged batches,
spatial grids and matrix scheduling, including independent E/V, resource
refusal and graph replay. The focused host selection passes 148 checks with
ten CUDA-only skips. Static dense-reference checks and displaced-geometry
references remain separate from production execution and timing.

One disabled-coarse-screening 3c cold regression call takes 13 iterations
instead of ten: tiny earlier roundoff differences precede a large DIIS
density excursion at iteration ten. Its final E/F passes the independent
gate but fails the optional much tighter paired-force diagnostic. The report
retains that call and a fixed two-round same-binary follow-up separately;
neither is pooled into default-setting timing. Do not infer that keeping the
local FMA order makes the entire solver deterministic.

## Consequences and revisit criteria

Ordinary finite tiles avoid most range-check overhead, with bounded extra
arithmetic for exceptional tiles. The remaining 3c cold gap to GPU4PySCF is
about 32% in this same-campaign default-grid comparison; integral forces,
density contraction and preparation remain significant targets. This is
sampled workload evidence, not a global fastest-configuration claim.

Revisit if exceptional tiles become common, compiler flags change the IEEE
nonfinite invariant, or a different tile/accumulator schedule is proposed.
Any DIIS robustness change needs its own independent solver qualification;
it should not be hidden in a contraction optimization.
