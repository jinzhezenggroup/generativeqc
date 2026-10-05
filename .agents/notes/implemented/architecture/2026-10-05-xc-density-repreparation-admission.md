# Decision: reject overlapping enabled XC density providers

Status: implemented
Date: 2026-10-05
Agent: dot

## Problem

`CudaXcPlan::prepare_density` permits binding replacement before evaluation.
It prepares a replacement provider before publishing the new tables and releasing
its predecessor. Each provider independently admits one matrix cache plus its
96 MiB opaque-provider allowance. A second enabled owner can therefore overlap
the first under the caller's single steady-state reservation. The numeric ledger
tracks matrix caches, not opaque provider allowances, and context preparation
precedes the replacement cache's ledger check. Even a ledger limited to one cache
does not prevent transient acquisition of the second context.

## Decision

Reject a nonzero-budget preparation while an enabled density provider exists,
before constructing temporary bindings or entering provider preparation. Preserve
the prior owner and tables on refusal. A zero-budget preparation remains a
transactional way to replace the tables and release an enabled provider; a later
nonzero preparation may then acquire a new one. Existing setup validation still
runs first. Absent, generated, and disabled incumbents remain replaceable.
Successful AO discovery already releases a dense provider before publishing the
indexed layout, so preparation after discovery remains supported. Empty maps
continue to require no optional owner.

## Rejected alternatives and limits

Releasing the incumbent before validating and constructing its replacement would
silently weaken failure recovery. Counting only cache bytes misses opaque
contexts. Allowing repeated enabled owners needs explicit peak host/device
admission and an API capable of distinguishing that reservation from the current
steady allowance; this repair does not introduce either.

Disabled provider wrappers retain host metadata. Their transactional replacement
can still overlap that metadata; this guard does not claim a universal peak-host
bound for all setup paths. It specifically prevents overlapping enabled optional
provider reservations and does not qualify a new production provider policy.

## Evidence

`tests/python/test_xc_density_repreparation.py` executes the actual owner methods
with host runtime/provider doubles. It covers early refusal with both ample and
one-cache numeric capacity, generated/disabled replacement, zero-budget release,
failed binding/provider preparation, strict/mixed full-tail tables, validation,
empty/response domains, and actual AO-discovery failure/reset control flow.
Against the pre-fix method, the overlap and one-cache cases reach two modeled
96 MiB context reservations; the repaired method stops at one without invoking
binding or provider construction. These are ownership/admission assertions, not
measurements of actual GPU allocations, GPU numerical qualification, or timings.
The single-preparation native KS path is unchanged.

## Revisit when

A caller needs atomic replacement of one enabled provider with another and can
explicitly reserve both owners' peak host/device costs.

References: `src/dft/cuda_xc.hpp`, `src/tensor/cuda_panel_product.cuh`,
`src/runtime/resource_cuda.cuh`, and the earlier
[KS provider-reservation decision](2026-10-05-ks-density-provider-reservations.md).
