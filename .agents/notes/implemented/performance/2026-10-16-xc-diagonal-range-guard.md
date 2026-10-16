# Decision: guard diagonal XC contraction reuse, not coefficient preparation

Status: implemented; frozen-source standalone and integration qualification complete
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

## Final complete-endpoint qualification

The final scientific source `5940546542d0cf77b60c8e59a2df32f8852f8d0f` completes
192 clean calls and eight separate diagnostic calls in finite node1 Slurm jobs
9053/9054, with the original 1800-second population limits, all five warm samples,
moving grids and independent geometry oracles. The clean maximum energy/force
errors are 2.546585164964199e-11 Eh / 6.738787197528573e-11 Eh/Bohr. All prescribed
populations complete; the earlier missing-AOT-sidecar attempts remain failed
receipts, not successful timings.

Standalone 96-atom cold E+F is 308.882366 -> 296.468955 s at matched 24/24
iterations/Fock builds; warm is 83.002791 -> 82.456957 s. The modest warm gain
does not reach the overall 10% objective alone, but the cold component is
independently useful and no 48-control regression exceeds 0.55% in these pairs.
The separate, actually measured overlay of the six #2558 force compiler files
at `719214668aeb69146174a7039490fd95d5607533` gives 315.298063 -> 254.163584 s
cold and 82.237315 -> 42.716903 s warm. Its cold trajectory changes 25/25 ->
24/24 and is explicitly not normalized away. Changed-geometry warm improves
82.349748 -> 42.753795 s at matched 1/1 work. These are frozen-source comparisons,
not newly measured current-master results or an attribution of the force stack
to this XC-only implementation.

The standalone diagnostic cold also changes 25/25 -> 24/24, unlike the clean
standalone pair. Profile-specific counts and mutually exclusive wall/device
ledgers remain separate. Integration warm order6/7/8 device seconds are
21.726820/27.807967/5.725769 -> 13.762362/1.987465/0.450183. Native order7/8
preparation/contraction counters demonstrate real work reduction without changed
enumeration; order6 preparation counters are unavailable, not zero. Cold remains
dominated by roughly 99.69 s Direct-J and 94.27 s XC device work at 24 builds.

The compact packet retains every force bit for the 24 candidate96 clean calls
and all eight profile calls. The other 168 clean calls retain all scalar samples,
actual work counts, independent-validation receipts and raw force/JSON hashes;
their force gates were checked at collection/packaging, not replayed from the
compact packet. Complete raw streams remain in ignored local campaign directories.
This explicit boundary avoids claiming 200 offline force rechecks. The offline
verifier lives in the validation tooling, with a small benchmark entry point.
Only this task's preceding MD narrative is losslessly compressed, with all its
bytes and scientific data retained; the global 64-MiB cap is unchanged.

See `benchmarks/results/xc-diagonal-potential-20261016/README.md` and PR #2567.
The guarded path is default on the branch under existing admission and fallback;
the composed force overlay is separately unmerged. No merge, tag or Release
is performed.
