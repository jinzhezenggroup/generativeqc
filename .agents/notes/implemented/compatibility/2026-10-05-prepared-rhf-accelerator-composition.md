# Compose prepared RHF contractions with optional response accelerators

Status: implemented
Date: 2026-10-05

## Decision and rationale

Reserve the complete five-table prepared contraction allowance before admitting
optional DF inverse and recycled-direction storage. This preserves the earlier
physical-owner provider priority while adding all descriptor storage to the
complete budget. Exact prepared capacity admits the tables; one byte less keeps
the scalar path. All optional accelerators continue to use only remaining space.

The recycled-direction identity includes the admitted prepared/scalar policy,
in addition to the exact frame, physical operator and device. Shared provider
setup can subsequently refuse its optional storage. In that case clear the
prepared-policy cache, remove its complete allowance and disable recycling for
this call before any physical action. Do not reuse or publish an image under a
policy the owner did not instantiate. A later scalar-budget call can admit its
own scalar-policy cache normally; no numerical work is retried by this repair.

## Existing active-cache peak repair

A caller-owned active cache remains live during optional inverse preparation.
The earlier admission order did not charge or retire it until after allocating
the inverse. A stale cache from a larger frame could therefore remain above the
current budget while fresh inverse storage was allocated. The endpoint caller
restores the cache bytes into the inclusive response maximum, so the physical
owner must still account for their coexistence itself.

Before inverse setup, clear stale or unaffordable active cache storage. Otherwise
reserve the existing cache payload while the transferred preconditioner data and
inverse copies coexist. Remove only this temporary charge immediately before
normal cache-plus-exact-image admission, which charges the final owner once.
Matching populated cache data remains available when the combined budget fits;
resource rejection retains the exact diagonal solve and normal cache fallback.

## Preserved gates

The provisional screened solve, optional inverse and initial guess still require
an independent scalar zero-screening final residual. Exact diagonal correction,
all attempted-work diagnostics, inactive-cache charging or release, once-only
whole-endpoint resource retry and publication after derivative gates remain.
No scientific equation, precision policy or accelerator default changes.

## Validation

Source-extracted admission tests cover inactive cache exact/short bounds,
prepared table priority and the final provider-policy fallback. Existing real
GMRES composition tests cover screening, inverse refusal, recycling, retained
attempt counts and propagation of physical/audit errors. Generated five-stage
oracles and ownership tests remain separately required. Host validation is not
fresh device qualification or an endpoint performance claim.

References: #1918, #1935, #1868
