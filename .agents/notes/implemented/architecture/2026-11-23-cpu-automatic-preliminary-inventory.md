# Decision: Resolve automatic CPU cold starts in the native inventory

Status: implemented
Date: 2026-11-23

## Decision

Direct FP64 CPU HF/KS automatic MINAO and Hückel plans query the resolved
provider and its numeric inventory in one native call per item. The compact
query reuses the prepared target's topology gate and the explicit capacity
kernel. A target's spin channels come from its resolved model, not from its
multiplicity. Hückel borrows atom-local shell metadata only; geometry, basis
values and scientific state are not constructed by the query.

A target-domain miss returns no provider or resource reservation. A local-cap
skip preserves the provider ID and required bytes but reserves no source
storage. An aggregate host limit cannot silently change that decision. Both
the resolved provider and the disposition enter the resource schedule
identity; geometry-only replay can reuse that schedule.

The explicit v1 contract, default candidate discovery, and fitted/CUDA
planning are unchanged. A library lacking v2 retains the split system-policy
and capacity queries. Native refusal and inconsistent output do not fall back.
The CLI remains explicit; it does not interpret the Python default as part of
its public command contract.

## Evidence

On base `cae2959e1`, the native compact and normalized capacity queries agree
at the exact cap and one byte below for MINAO and Hückel. An unrestricted
singlet, ECP, nonsinglet and out-of-domain element all decline independently.
Installed Python tests cover a ragged He/water/Ca18+ batch, local and aggregate
one-byte boundaries, cold/warm/moved replay, and invalid receipts without
falling back. Native-only installed SDK PBE/H2 force results and receipts
match the equivalent explicit CLI policy for both providers, including local
budget skips. All 152 CPU CTests, 334 installed Python tests, and strict
Sphinx pass.

Seven interleaved before/after process pairs after warmup on one EPYC 7K62
core measured construction, resource planning, preparation, three PBE0 force
executions (cold/warm/moved), and teardown. Median endpoint changes were
+0.68% for MINAO water, +0.65% for Hückel water, and −0.30% for ragged Hückel.
Energy, forces, iteration/Fock counts, warm selection, and seed diagnostics
(excluding timing) matched exactly. The water target Fock counts were
13/4/12 for both providers. Hückel performed 7 cold preliminary Fock builds
for water and 2 for the admitted He in the ragged case. The ragged modeled
peak fell by 68,232 bytes; this is not an RSS claim. No GPU test was run.

Raw receipts, versioned library digests, and compiler-cache statistics are
under `/data/jzzeng/gqc-2227-sdk-JsefEq0x/python-automatic-preliminary-host`.

## References

- [Explicit CPU cold-start layout](2026-11-23-python-preliminary-host-admission.md)
- [CLI opt-in automatic admission](2026-11-23-cli-automatic-preliminary-admission.md) extends this decision to the CLI.
- [Resource planning](../../../../docs/maintainer/resource_planning.md)
