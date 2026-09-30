# Decision: fail-closed inventory for performance default promotion

Status: implemented
Date: 2026-09-30

## Problem

Performance paths had accumulated across public precision policy, SCF controllers,
CUDA runtime selectors, TensorSchedule dimensions, and CUDA Graph execution.
Implementation or microbenchmark evidence alone was easy to confuse with a
decision to make one of those paths automatic.

## Decision

Issue #1598 owns one machine-readable default-promotion inventory under
`manifests/maintenance/`. A repository checker reverse-discovers controls from
the current production ownership surfaces and requires every discovered control
to appear exactly once with an owner issue, rationale, revisit condition, source,
and one of the shared promotion classifications.

The initial audit covers:
- experimental/incremental SCF option fields;
- the public `precision="auto"` policy;
- every CUDA HF/DF runtime-policy variable retained in resource identity;
- every boolean TensorSchedule dimension plus its reduction provider; and
- TensorIR CUDA Graph replay.

Registration is bookkeeping, not promotion. Guard misses retain the existing
production path, diagnostic controls remain diagnostic, and negative evidence is
kept explicitly rather than silently retried as a default.

## Rejected alternatives

A hand-maintained table without source discovery was rejected because a new
default-off production switch could bypass the table. Turning every implemented
optimization on was rejected because #1598 requires complete public-endpoint
evidence and exact workload/resource guards. Treating environment variables as
the product API was also rejected; explicit public policy remains authoritative
where one exists.

## Invariants

- A newly discovered audited control must fail CI until it is registered.
- Every registered control has an owner, rationale, and concrete revisit condition.
- Inventory membership never establishes numerical safety or performance benefit.
- Negative evidence remains durable and does not become an automatic candidate
  merely because the implementation still exists.
- Already-default rows describe automatic guarded policy; explicit overrides do
  not inherit production qualification automatically.

## Evidence

The checker currently resolves 59 controls into 16 decision groups. Regression
tests cover the current repository, missing metadata, duplicate registration,
missing controls, and a synthetic new TensorSchedule opt-in.

## Consequences

Adding a new audited performance control has a small explicit maintenance cost,
but the review now exposes whether it is diagnostic, negative, awaiting evidence,
ready for guarded promotion, already automatic, or ready to retire.

## Revisit when

Expand source discovery when a new production policy owner is introduced outside
the audited surfaces. Narrow or retire controls when their owner issues retain
enough endpoint evidence to make a durable promotion or negative decision.

## References

- #1598
- #375
- #406
- #507
- #508
- #509
- #783
- #971
- #990

Agent: ChatGPT
Model: GPT-5.6 Sol
