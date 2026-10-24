# Decision: preserve frozen-source CPU CBS error evidence

Status: implemented in this change
Date: 2026-10-23

## Problem

The existing 64 MiB aggregate benchmark-results cap could not admit the complete
CPU CBS reference receipts. The first successful native build stopped at its
capacity preflight before scientific execution. A later complete water fixture
passed, but its lossless raw bundle alone occupied 168,804 bytes. The actual
remaining 64 MiB headroom at that time was 154,631 bytes; this was a storage
failure, not a failed numerical comparison. The original outcomes remain retained.

## Decision

The user explicitly authorized increasing capacity ("允许扩大容量"). The aggregate
64-to-80 MiB change is a separate storage dependency, PR #2704, whose rationale
is owned by `2026-10-23-benchmark-evidence-capacity.md`. The CPU evidence slice
must consume the actually merged policy rather than treating a local proposed
value as effective upstream capacity.

On 2026-10-24, normal upstream lossless compaction restored room under the
unchanged 64 MiB cap. This CPU publication can consume that actual capacity
without waiting for PR #2704. Earlier capacity failures remain valid historical
outcomes for their original trees; a later fit does not rewrite them as PASS.

Keep the 1 MiB per-file cap, 2 MiB new-evidence review budget, independent
oracles, numerical gates and protected review/merge requirements. Preserve
complete endpoint returns, traces, logs, failed preflights, identities and file
modes rather than truncating logs, dropping samples or rewriting other owners'
historical records. No production or reference data changes are needed.

Do not invent a requirement for a mathematical absolute compressed-byte upper
bound on every future dynamic log. Reasonable complete-slice estimates and
bounded first-fixture execution can establish an empirical budget. If the
actual complete records exceed the cap, retain them unchanged and block
publication/dependent steps. The valid numerical result remains distinct from
the failed storage gate; no repeated calculation erases that history.

## Source and qualification boundaries

Measurements use the explicitly frozen Git LF source
`bbd41a42764e8d22d15c1348f70a633961d4dcf5`, source-matched native CPU library and
committed PySCF 2.14.0 reference fixtures. The results directory binds those
identities and retains the original build/attempt provenance. Later upstream
changes do not relabel that library as a current-master build. Advancing upstream
alone does not require repeating a valid frozen-source measurement.

The temporary observer wraps only the public CBS endpoint, returns the original
object, preserves scientific exceptions and restores the method in `finally`.
Raw results retain the executed observer and runner bytes. Independent replay
uses the committed PySCF records and recomputes strict absolute errors, including
missing/non-finite rejection; it never qualifies a missing H2 result from water.

CPU conventional exact CBS, the physical spherical H2 DZ/TZ pair and synthetic
water-algebra fixtures form this evidence slice. Correlation-only DF, full RI-JK,
CUDA, general SDK qualification and the unmeasured parent scope remain separate.
Use Refs #2681; this evidence does not justify closing the parent issue.

## Consequences

Storage acceptance and numerical acceptance remain independent. Replaying the
retained bytes can reproduce error tables and verify all inputs without running
a new scientific calculation. Native execution, CI, formal latest-head review
and protected delivery must retain their own evidence; a larger storage budget
does not satisfy any of those gates by itself.

## Evidence

`benchmarks/results/cbs-reference-errors-cpu-20261023/` contains the receipts,
independent replay and publication inventory. That inventory, rather than this
note, records the final measured outcomes and exact error values.
