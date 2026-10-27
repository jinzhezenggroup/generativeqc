# Decision: discover XC sharing from specialized point structure

Status: implemented; catalog-wide execution remains explicit opt-in
Date: 2026-10-28

## Problem

The KTBM prepared-product optimization has useful numerical and endpoint evidence,
but source-entry membership cannot be the general discovery rule. Other scalar
parameters control powers, branches, zero terms and constant CSE. Treating every
`params_a_` binding as runtime data loses those specializations; an initial census
also encounters compile-time-exponent requirements in nine registrations.

## Decision

Keep the existing measured KTBM recipe as the qualified default. Add an independent
catalog-wide candidate path that starts after literal import, AD and algebra.
Lift only arithmetic constant leaves in the existing Graph. Keep predicate
dependency closures, exponent payloads, zero and binary scaling constants native.
Compare complete emitted general/restricted bodies and their layouts, work domains,
versions and ingredient contracts; source-entry names are not grouping keys.
Different constant alias/CSE structures deliberately split groups.

Restore group-common constants and deduplicate identical coefficient columns.
Bind independently discovered general and unpolarized constant lists into the same
immutable launch row. Preserve true LDA U1 RKS execution when the equal-spin
optimization is disabled; P2 is not an interchangeable LDA RKS worker. Reject
launch rows larger than 384 doubles, leaving room below the portable 4 KiB CUDA
parameter limit. Singletons and incompatible/resource-exceeding groups retain
ordinary literal execution.

`qualified` is the default selection policy; `discover` explicitly emits all
structurally compatible candidates. A machine-readable sharing report separates
candidate membership from actual selection. Planning caches canonical literal
providers within the existing package lifetime so discovery and emission do not
repeat their lowering. Neither policy changes public functional admission or the
canonical coefficient-bound CPU/CUDA mathematical descriptors.

## Evidence and default boundary

The full 197-registration structural census finds 12 groups covering 73
registrations, including the original 26 KTBM registrations. The remaining 124
retain specialization. This is an inventory of this source revision, not a fixed
enrollment table or a claim about all Libxc registrations.

The first SM120/PRO6000 paired pilot found near-neutral aggregate point timing
for most new groups, but approximately 2–3% regressions in some small paired
batched TH and BKL cases. Default promotion requires the point-case and complete
endpoint gates, not just a favorable catalog median or object-size result. The
general path therefore remains opt-in. Do not silently widen the default merely
because the discovery planner can construct a mathematically valid worker.

The repeated final point campaign confirms four cases above 1% (maximum 2.98%).
657 point/Fock and 23 independent public-KS GPU checks pass. The remaining MVSB
UKS cohort fails to converge with both shared and literal execution; all six
failed endpoint attempts are retained. The other 24 cohorts produce 720 complete
calls with matching semantic work, maximum energy difference 2.274e-13 Eh and
phase median time ratios 0.980498–1.010619. Source-synchronized publication
rebuilds additionally pass 15 complete smoke calls. Broad endpoint near-parity
does not erase the failed point-case gate or the nonconvergent cohort.

Source-matched numerical, build, point and endpoint evidence, including failed
performance gates, lives in
[`xc-structural-sharing-20261028`](../../../../benchmarks/results/xc-structural-sharing-20261028/README.md).
Permanent tests execute shared coefficient rows against independent Libxc,
exercise true U1/P2 layouts and invalid inputs, and verify selection/reporting.
Historical one-off default-byte comparisons remain evidence rather than required
CI source fingerprints.

## Rejected alternatives

- Adding a growing family-name whitelist as the discovery mechanism: misses
  structurally equivalent programs and cannot prove ABI compatibility.
- Making every scalar binding symbolic before AD: changes specialization and can
  violate compile-time exponent requirements.
- Promoting every compatible group: the retained small-batch regressions fail the
  existing per-case performance boundary despite substantial object-size savings.
- Merging different literal CSE shapes: requires an additional mathematically and
  performance-qualified transformation. The KTBM prepared-product recipe has that
  evidence; generic discovery does not assume it applies elsewhere.

## Revisit when

New source-matched point and complete-endpoint evidence qualifies additional
selection policies; or a general structural transformation can recover shared
workers across different CSE shapes without increasing executed work or losing
useful specialization. The current census and reports supply candidates, not
approval for automatic promotion.

## References

- [Original prepared-coefficient promotion](../performance/2026-10-28-xc-prepared-parameters.md)
- [Original sharing ownership](2026-10-27-xc-parameter-worker-sharing.md)
- [Current CUDA XC guide](../../../../docs/developer/xc_native_cuda.md)
