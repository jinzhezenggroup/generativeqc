# Decision: coalesce admitted Direct packets and route DFT dddd values

Status: implemented (default-off qualification candidate)
Date: 2026-10-06

## Problem

The first primitive-pair recurrence schedule shared preparation inside one
256-AO packet. A dddd shell quartet can contain 1,296 Cartesian components and
six packets; repeating the recurrence for each packet still amplifies source
work. Its HF angular launcher also does not serve the DFT Direct provider,
whose dddd fallback is a separate class-specific persistent stream.

## Decision

Give each thread a finite register slot for each coalesced packet. Decode and
screen every component independently before publishing a common shared
recurrence. Coalescing changes the lifetime, not the Hermite/Coulomb equations
or primitive/component contraction order. Component values remain local and
each contracted ERI feeds the retained J/K symmetry scatter.

HF compaction admits all shell packets together on one precision/source route.
Its tile-zero worker can therefore consume the whole shell; other packets do
no source work. Balanced shell angular momenta bound the maximum component
product at each total order, so lower-order workers need fewer register slots
than ffff's 40. The existing exact-class generated mask still excludes all
components of a covered shell class consistently.

The DFT dddd stream changes to eight warps only for the frozen materialized
value request. Its shell admission, active systems and J/K identities remain
the incumbent's. Six slots per lane cover the full 6^4 domain, including the
16-component final tail. Whole-CTA publication/retirement fences replace warp
fences for this variant; screened empty claims also retire their shared state
before the leader claims the next shell. Derivatives retain their independent
single-warp owner. Optional test counters are borrowed; production allocates
none.

Missing pair storage and separately selected reachable/convolution schedules
retain the explicit incumbent. Positive K-only remains positive K; only the
combined HF scatter applies its restricted/unrestricted exchange factor.

## Qualification and limits

The whole-shell GPU fixture compares with the retained raw component evaluator
and an independent host symmetry orbit. Its two-primitive dddd case publishes
1,296 components after 16 Coulomb preparations; a screened-out shell performs
zero preparation. Same-pair, signed contractions, inactive systems, spherical
public projection, both spins and all three J/K identities are acceptance
domains. Native launcher work counters must distinguish execution from a
fallback; equal final matrices alone cannot do so.

Finite Slurm job 6146 qualified the native DFT launcher, all three value
identities, both spins, missing cache/offsets, reachable/convolution fallbacks
and inactive/screened empty streams. Memcheck, initcheck and synccheck reported
zero errors. A focused cross-warp shared-lifetime racecheck reported zero
hazards; the earlier all-order racecheck was stopped after more than ten
minutes and is not counted as passing evidence. The standalone owner was
compiled with normal NVCC `-O3`, without fast/split compilation, while the
full release library build remains a separate gate.

The original standalone fixture supplied explicit-shell ERI inputs but omitted
the native dispatcher's AO-to-shell map and shell-screening system prefixes.
The stream/fallback tests exposed those incomplete fixture preconditions;
supplying the production metadata contract passed the retained numerical
gates. Preserve the distinction between a component oracle's minimal inputs
and a prepared native stream's complete topology.

Complete 48/96-atom HF/PBE0 energy+force timing, solver histories and semantic
work are still required. Register spill/occupancy and recurrence serialization
can offset the work reduction. These counts alone do not justify a speedup
claim, default promotion or completion of #1892. Stationary derivative sharing
and uncovered DFT high-angular consumers remain subsequent slices.

## References

#1892; PR #2000; `docs/developer/direct_pair_recurrence.md`; the earlier
`2026-10-06-direct-pair-materialized-recurrence.md` records the packet foundation
and coefficient underflow decision.
