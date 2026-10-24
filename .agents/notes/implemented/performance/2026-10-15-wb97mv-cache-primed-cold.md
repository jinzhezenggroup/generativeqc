# Decision: prime compilation caches before fresh small-system cold measurements

Status: implemented
Date: 2026-10-15

## Problem

The WB97M-V README cold curves rose at 3 atoms: direct 63.451 s and DF
44.671 s, versus 30.796/28.172 s at 12 atoms. Per-call telemetry attributes
58.070/39.210 s to force preparation, which includes code generation,
artifact loading and resource setup, not isolated compiler time. Warm calls
and independent energy/force gates are normal. Later sizes retain compiler
caches populated by earlier work, so the first point is not a fair estimate
of cached-artifact endpoint scaling.

## Decision and evidence

Run one complete 3/12-atom priming population per native mode, then measure
each size in a fresh process and Calculator with the unchanged public
initializer. Retain all twelve measured endpoints and all independent gates;
do not subtract preparation time, reuse converged densities or select a
minimum from retries. No production code or library is changed: the native
source remains the recorded `810f081ef293256ebe20beb7298f9d21996fec81`.

Finite Slurm jobs n5/1579 and n1/8493 complete 48 priming and 48 measured
calls, all independently checked. New cold 3/12 timings are direct
7.299/30.861 s and DF 7.779/29.126 s. Retain the large-system observations;
do not rerun or publish incomplete DF/96. Historical reference timing data
are not promoted to this new native compilation-cache policy.

Archive the original two 3-atom native populations losslessly with shared,
hash-bound unchanged references. Original 12-atom controls and full summaries
remain recoverable from existing Git history and ignored raw storage. All
new observations retain source/library identities and separate primer
receipts. SVG presentation sharing is pixel-identical, providing room for
retained scientific evidence without raising the aggregate storage budget.

## Boundaries and revisit conditions

This supersedes only the small-system cache policy from
[the prior force-budget decision](2026-10-14-wb97mv-readme-force-budget.md).
Cold still includes construction, resource preparation and the first full
SCF plus forces in the measured process. Priming runs are not timed cold
samples or reference speedup evidence. A future fully uncached campaign
must independently record its compiler/artifact state and retain every
endpoint's startup cost rather than mixing cache states within one curve.
