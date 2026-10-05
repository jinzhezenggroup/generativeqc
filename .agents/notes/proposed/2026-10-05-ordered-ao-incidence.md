# Decision: retain ordered AO incidence during phased geometry response

Status: proposed; complete endpoint qualification pending
Date: 2026-10-05

## Problem

After computing each active AO's point gradient, the cooperative geometry kernel
lets each atom scan the entire active AO map to gather its own contributions.
The map is identical for all points in the tile. The retained 96-atom profile
attributes 2.7062 s to the complete AO/grid geometry kernel; it does not isolate
this selection loop from XC evaluation or the other reductions.

## Decision and lifetime

Before the geometry kernel, construct per-atom heads and per-active-AO next
indices. Reverse insertion produces increasing active-AO index order, including
arbitrary maps and noncontiguous atom membership. Each atom remains the sole
writer of its gradient. The point-motion owner keeps its existing full ordered
sum. No floating-point atomic operation, regrouping, cutoff or scientific formula
is introduced.

The existing admitted Becke pair panel is dead until AO response completes, so
it lends `atoms + active_aos` size_t words to this metadata. The following Becke
pair-primal phase overwrites those words only after the geometry kernel finishes
on the same stream. No allocation or reservation increases. Admission checks the
actual borrowed capacity and cooperative AO shared-panel capacity. Empty maps,
serial AO response, missing phased storage and capacity misses retain the dense
scan. Invalid map/atom indices set the existing sticky error and cannot publish a
successful force result. Rebuild at every tile submission; no identity or geometry
cache assumptions are added.

For a valid cooperative tile with P points, A atoms and M active AOs, the selected
reduction visits PM incident entries instead of PAM membership checks; metadata
preparation visits M entries and initializes A heads. AO pullback evaluation and
the ordered point-motion sum remain unchanged. The actual owner launch counter
includes the additional preparation kernel. These are source-derived loop counts,
not measured machine instructions or a new lower formal scaling for DFT.

## Initial evidence and limits

The final host gate passes 111 tests, including bitwise AO/grid comparisons
against serial evaluation, reversed AO maps, noncontiguous atom membership,
12..128 atoms, 2..1024 AOs, geometry lane tails, poisoned scratch, invalid maps
and atoms, failure/reset behavior and resource admission. The final shared-owner
CUDA gate passes 24 tests over 48/96 atoms, full/subset/empty maps, external seeds,
explicit/implicit owners, tails and moved/restored centers.

A prototype before the empty/serial-capacity guard was measured in n1 RTX 5090
Slurm job 5904. One process selected the libraries in
baseline/candidate/candidate/baseline order with fresh owners for each case.
Each row below is the median of three 256-point submissions (original, moved,
restored synthetic geometry). Events span the actual shared-owner enqueue,
including transfers, submission gaps and all Becke phases; this is neither an
AO-only kernel time nor a molecular E+F endpoint. Both arms contain #1950's pair
logarithm schedule. Integral derivatives are replaced by a fail-closed stub in
both isolated libraries because this gate only submits geometry work.

| Atoms / AOs | Baseline medians, ms | Incidence medians, ms |
| --- | ---: | ---: |
| 48 / 384 | 0.487168, 0.482720 | 0.384704, 0.386112 |
| 96 / 768 | 1.035552, 1.030592 | 0.727968, 0.731776 |

The complete probe checks all source arrays against the bounded route. Its
one-point tails and failure/recovery observations are retained, not folded into
the 256-point medians. Inputs are synthetic and do not qualify molecule-specific
profitability or the public active-AO distribution. Final-source sanitizers and
paired complete cold/warm/moved/moved-warm endpoints are required before promotion.

Artifacts, generated source, raw event/wall samples, the input transformation
and ccache/compiler receipts are retained under
`/data/jzzeng/qc-ao-incidence-1894/.artifacts/1894-ao-incidence/` locally and on n1.
An initial incorrectly generated seven-source LDA wrapper failed the eight-source
PBE0 fixture's output-size check. That configuration failure remains in the logs;
only the corrected PBE0 wrapper supplies the passing CUDA evidence above.

## References

- #1894, #1895; this is a separate scheduling slice above #1950.
- `benchmarks/results/pbe0-public-force-policy-20261005/README.md`.
