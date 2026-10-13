# Decision: retain the occupied projection across the HF-DF capacity crossover

Status: implemented; endpoint qualification pending
Date: 2026-10-13

## Problem

Issue #2421 reports a 96-atom warm HF-DF endpoint of 59.318 s, versus
0.673 s at 48 atoms. The default value allowance retains packed B but drops
the complete occupied projection. Increasing only the value allowance enables
occupied SCF, while the force still repeats dense fitted-panel projections.

## Decision

Reuse the method-owned resident budget policy for singleton restricted HF,
only when automatic storage/exchange and the actual RHF planner admit both B
and the complete occupied U. Charge DIIS and factor state in that final check.
Update the DF and one-electron response owners with the same resolved cap.
Explicit public/response budgets, UHF, batches and insufficient capacity retain
their previous bounded plans.

Select the existing fitted occupied force consumer automatically only for a
current full-rank metric and exact final-projection token. Revalidate the model,
geometry/solve/density generations and actual canonical density before borrowing
U. Stale/corrected states retain bounded fitted-panel response. A force attempt
still consumes the exclusive projection lease; another SCF must publish a new one.

## Evidence and rejected alternative

Diagnostic-only n1 RTX 5090 runs used library SHA256
`622722de6277147d512fe045f969b80c10b5db76d9aac0fbabd4f70326c5aac9`.
The default trace selected dense SCF, spent 115.008 s in 30 K builds, and
40.929 s in force response. A 4-GiB response-cap ablation enabled occupied SCF
but increased the generic force-response cost to about 76 s. Memory rebalancing
alone is therefore not a complete fix.

Slurm 8054 combined that cap with the existing explicit fitted occupied
consumer. Its four diagnostic calls took 36.039 / 3.447 / 28.622 / 3.422 s
(cold / warm / moved / moved-warm); force response took about 2.09 s and borrowed
the 3,649,044,480-byte final projection. These traced, single-repeat calls are
mechanism evidence, not the plotted population or an independent accuracy gate.
Formal default-route 12-call energy/force qualification remains required.

## Invariants and revisit conditions

Keep the independent 1e-8 Eh / 1e-7 Eh/Bohr endpoint gates, finite memory ownership,
explicit diagnostic overrides, and exact fallback. Revisit the response floor if
its live allocations or occupied response algebra change. Do not replace the
final actual-plan check with the shared resolver's weaker packed-B capacity test.
