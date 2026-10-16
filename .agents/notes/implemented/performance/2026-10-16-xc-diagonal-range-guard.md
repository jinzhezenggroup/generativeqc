# Decision: guard diagonal XC contraction reuse, not coefficient preparation

Status: implemented; complete-endpoint qualification in progress
Date: 2026-10-16

## Problem

This supersedes the unpromoted full-panel layout in
[the original diagonal-contraction trial](2026-10-16-xc-diagonal-potential.md).
Generic generated XC inputs have no upstream magnitude bound. Moving a factor
of two into coefficient preparation changes finite-input overflow and underflow
behavior, even when sampled molecular endpoints pass an independent oracle.

## Decision

Always produce the original compiler-owned half-coefficient panels, in their
original arithmetic order. The strict-FP64 diagonal consumer prepares an exactly
doubled right panel in shared memory only when that individual panel element is
bounded. At the existing load barrier a uniform CTA vote admits one-sided reuse
only when the quadrature weights and all four AO/panel operands in that point
tile have magnitude at most `2**128`. The value/gradient jet always retains both
legs. Without an explicit weight witness it always uses the original two legs.

On admitted tiles doubling is exact, each new product is at most `2**257`, and
even a signed-64-bit point count cannot overflow their sum on its own. If a
canonical panel has underflowed before applying its weight, the admitted weight
and AO bound its amplification by `2**256`. Bounding only the final panels and
AOs is insufficient; a large weight can conceal amplified preparation underflow
inside an otherwise bounded panel. No new lower
magnitude cutoff, clipping, input screening or tolerance is introduced.
Ordinary FP64 reassociation still changes rounding; independent complete
energy/force gates remain mandatory, not an assertion of bitwise equivalence.

Unsafe tiles use both original products. A byte flag records whether each
right-panel entry was doubled, permitting exact inverse scaling before the
fallback. Large entries are never doubled. In particular, the fallback does not
attempt to halve an already-overflowed panel. Its producer and contraction match
the original finite-domain arithmetic.

The CTA vote replaces one existing barrier rather than adding a barrier to the
productive path. The flag panel costs `tile*(tile+1)` shared bytes and is included
in compiler resource admission. There is no new global allocation, CTA/page-width
choice, method selector, or approximation of Direct J/K.

## Evidence and invariants

The actual emitted host barrier simulation and CUDA tests exercise both review
counterexamples: a finite `2e208` result that the full-panel trial rejected, and
an exactly cancelling off-diagonal that it changed to `-2**25`. Unsafe outputs
must match the original two-legged consumer exactly, not just a relaxed
tolerance. CUDA qualification covers contraction with FMA both enabled (the
production build mode) and disabled, both spins, maps/tails, accumulation,
nonfinite rejection, and the unchanged FP32 consumer.
An additional control has bounded AOs and final panels but a `2**1000` weight;
it catches a guard that overlooks amplified intermediate underflow.

The 16x16 diagonal probe uses 8992 body shared bytes, 48 registers and zero
stack/spills. Compiler launch qualification includes the full flag panel.
Dynamic instruction counters remain unavailable (`ERR_NVGPUCTRPERM`), so logical
pair-point work and symbolic five-versus-eight productive multiplications must
not be labeled measured hardware instructions. The revised source receives
fresh complete matched populations; timings from the superseded source are not
silently reused as its qualification.

## Revisit when

A different contraction representation can remove more density/potential work
while retaining a sound input-domain witness, independent scientific gates and
bounded fallback. Any change to the range bound needs a new argument and the
extreme finite-domain controls, not just a successful molecular timing sample.
