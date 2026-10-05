# Decision: shared tables own cuBLASLt alongside affine providers

Status: implemented
Date: 2026-10-05

## Decision

Extend PreparedContractions with optional cuBLASLt execution of the same affine
request. Provider implementation identity remains below method/scientific APIs.
Build capability is explicit and defaults off; the macro and link dependency
propagate together so shared header layouts remain consistent.

Generalize optional preparation across cuTENSOR and cuBLASLt while retaining the
existing generated/cuBLAS behavior. Check aggregate simultaneous reservation
arithmetic before either family's plan allocation. Publish a shape only after
all plans succeed; failure in the second family drains the first family's plans
as well. Live cleanup failures propagate, and failed preparation leaves the
same shape available for an admitted retry.

The per-plan reservation must cover every selected provider in a mixed table.
This conservative interface avoids undercounting coexistence. Pointer storage
for enabled optional families is included in the existing descriptor budget.
Provider-typed provenance visitors preserve source compatibility with existing
cuTENSOR callbacks while reporting cuBLASLt algorithm configuration per slot.

## Qualification boundaries

The native probe exercises table execution and exact semantic work counters for
the 192 existing layout/precision/beta cases. It rejects the second cuBLASLt
preparation, retries the same shape, checks stale context and checked release,
and exercises mixed-family preparation rejection when both providers are built.
The build-disabled case retains explicit provider-unavailable rejection.

This does not add a cuBLASLt production resource profile or choose it for an
entire scientific region. Compiler eligibility remains restricted to supported
rank-2/3 matrix layouts; unsupported rank grouping is not silently packed.
Complete endpoint and opaque/lazy resource qualification remain #1888/#1886.
The standalone cached executor and rationale are in #1929.
