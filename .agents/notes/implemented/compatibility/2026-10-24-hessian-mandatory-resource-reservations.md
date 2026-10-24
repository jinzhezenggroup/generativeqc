# Decision: reserve mandatory Hessian storage before optional schedules

Status: implemented
Date: 2026-10-24

## Problem

PR #2594's optional one-electron parallel schedule and grid replay could turn an
admitted scalar request into a resource rejection. The KS adapter added mandatory
effective charges after selecting the optional integral arena. The aggregate
Hessian selected grid replay before charging its geometry AO outputs and other
mandatory response/source consumers. Optional host task-vector allocation failure
also escaped as endpoint OOM.

## Decision

Use the same normalized-topology one-electron planner for preparation and the
aggregate owner's scalar reservation. Its budget includes effective charges.
Reserve the complete aligned Hessian arena with scalar grid replay, plus scalar
one-electron storage, before preparing mandatory induced/DF/ECP/exact sources.
Only then admit grid replay and parallel integral storage from the surplus. Grid
payload capacity is rounded down to the enclosing 256-byte reservation boundary;
this padding is part of the aggregate budget, not free capacity.

Catch allocation failure locally while constructing optional task and graph
metadata. Move the validated task plan into the prepared owner rather than copying
its vector. Preparation caps planning to the caller's actual arena, so a scalar
workspace produced by a transient optional sizing failure remains usable when a
later allocation would succeed. Mandatory allocation, source closure, work,
provider, authentication and unrelated CUDA errors still fail; there is no broad
retry on INVALID_ARGUMENT.

## Evidence and boundaries

Host tests execute the production adapter, scheduler/preparation and complete
Hessian prepare path with isolated numerical providers. They cover charge and
alignment boundaries, RKS/UKS direct and DF/ECP consumer combinations, allocation
failure, cleanup and fatal-provider negatives. Original source and mutations that
omit reservations, option forwarding or local allocation fallback fail these
controls. These are resource/lifetime tests, not fresh CUDA numerical qualification.

Numerical kernels, ordered accumulation, scientific gates, component caps and the
180-second/24-GiB endpoint limits remain unchanged. No CPU numerical fallback or
new GPU campaign is introduced. The larger Hessian performance gap remains; this
repair does not expand the retained diagnostic-grid performance qualification.
The optional schedule is specific to the PR branch, not a defect attributed to
master without that feature.

## References

- PR #2594, reviews 5557517455, 5557543062 and 5558125746.
- `tests/python/test_native_one_electron_budget.py`
- `tests/python/test_native_one_electron_schedule_budget.py`
- `tests/python/test_native_hessian_budget.py`
