# Candidate: partition the long DF Hermite charge source traversal

Status: explicit candidate implemented and qualified; default promotion pending
Date: 2026-10-23

## Motivation

The 96-atom r2SCAN-3c DF endpoint cannot retain its complete packed three-center
tensor on the 32 GiB device. Its exact streamed Hermite charge contracts many
orbital primitives into a smaller auxiliary target domain. Nsight on source
35f67156 shows these charge kernels dominate repeated J work; the full
diagnostic endpoint spends approximately 1.455 seconds per J application.

## Candidate and ownership

`GENERATIVEQC_DF_MD_CHARGE_SPLITS=1..64` is frozen at owner creation. Unset
retains one CTA per target. More splits divide the original source groups into
contiguous partitions, then use a second kernel with one writer per auxiliary
Hermite component and ascending split order. This changes FP64 summation order
and therefore needs independent charge, projection, metric and force gates.
It does not change any source, threshold, screening, metric or precision.

Partials borrow the existing orbital primal-potential allocation, which is dead
during the auxiliary charge pass. The source density and raised orbital
response potential stay disjoint and live. Every complete split writes its
own partial, including zero tails; the reducer completes before the next
angular class on the same stream. A later orbital projection first clears or
refreshes the primal potential. Capacity is checked separately for every
angular class and derivative order; insufficient space retains the incumbent.
No new device allocation or graph-time environment lookup is introduced.

## Required qualification

Check nonsymmetric and changed densities, zero replay, graph replay, auxiliary
g and orbital f, independent displaced forces, tight budgets and sanitizer.
Then compare clean complete endpoints with the same binary at splits 1 and 8,
and qualify every retained energy/force sample. Kernel timing alone cannot
justify promotion. Keep the default unchanged if the complete endpoint fails
to improve or any scientific gate fails.

## Source-matched endpoint evidence

Source 4f6558075d6484ac65b6b6852f3f71765757f84a and all five standalone
stationary manifests were rebuilt from 5,984 authenticated inputs. Independent
metric/J/displaced-force tests pass at splits 8 and 32 (16 cases each), and eight
partition derivative/graph cases under Compute Sanitizer report zero errors.
Five alternating split-1/split-32 r2SCAN-3c DF96 processes give cold medians
52.552040/43.591161 s (17.051% faster) and per-process warm medians
14.898640/12.208637 s (18.055% faster). Both exceed the original MAD/2% floor.
All 104 native static and moved calls pass their independent gates; full raw
receipts, work counts, source/build provenance and offline reproduction are in
[the retained evidence](../../../benchmarks/results/df-md-charge-splits-20261023/README.md).
Earlier screen cohorts stay separate from repeated performance populations.
The default is still one partition. Promotion needs a bounded general admission
policy and its own exact-source default and fallback qualification; the measured
explicit control alone does not authorize relabeling default performance.
