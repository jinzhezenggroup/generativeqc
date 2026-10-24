# Decision: Share CPU HF phase sizing across native and Python frontends

Status: implemented
Date: 2026-10-15

## Problem

The #2227 R2 migration already shared CPU KS sizing, but CPU RHF/UHF phase
capacities still had separate Python and native CLI implementations. Both
described the same Jet/preparation/iteration/finalization lifetimes, including
the conventional four-center tensors still retained by CPU DF preparation.
Keeping two production sizing owners allowed their bounds to drift.

## Decision

`scf::cpu_hf_host_resources` owns the four per-item phase capacities. The native
CLI calls it directly; Python discovers an additive private shape-only C query
on its already loaded library. Query rejection propagates without retrying
Python arithmetic. Schedule identity distinguishes native and portable sizing.
The bridge uses scalar/stack storage and performs no integral, SCF, allocation,
provider discovery or CUDA work on successful requests. Every intermediate is
checked in the signed-int64 byte domain; errors preserve the entire output.

## Compatibility and boundaries

Library-free dry runs and older libraries keep the original portable metadata
bound. Removing it would make a metadata-only API implicitly load a provider.
This fallback does not perform scientific execution. RHF/UHF, direct/DF,
Cartesian/public AO coexistence, ECP workspace and zero-history sizing retain
their previous capacities. ECP workspace continues to come from its existing
capacity owner rather than introducing another quadrature policy here.

Fleet retained state, candidate composition, admission and leases are not
migrated by this increment. Numeric/value-metadata bounds exclude allocator
and runtime overhead; sampled host observations are not complete endpoint
receipts. CUDA execution and scientific equations are unchanged.

## Evidence

Native tests pin the H2 lifetime capacities, cover spin/DF/history variants,
invalid schema/topology/angular domains, transactional overflow and the exact
signed-int64 capacity boundary. Python tests compare the independent portable
bound across through-g, spherical, ECP and DF shapes, poison portable arithmetic
after native discovery, reject fallback after native errors, and verify fleet
capacity parity with distinct legacy provenance. Native CLI tests compare the
same query against Python-free RHF/UHF direct/DF resource commands. Existing
CPU resource tests retain exact/below-limit admission and cold/warm endpoint
energy/force comparisons, with partial observation limits kept explicit.

## Revisit when

Complete native fleet admission/lease ownership or a provider-free installed
shape service can replace the remaining frontend composition/compatibility.

## References

#2227 R2/R5/R6; #933/#934;
`docs/maintainer/resource_planning.md`;
`src/scf/hf_host_resources.hpp`;
`python/generativeqc/resources_hf.py`;
`.agents/notes/implemented/architecture/2026-10-12-shared-cpu-ks-host-sizing.md`.

## Follow-up

The subsequent `2026-10-15-native-cpu-hf-fleet-sizing.md` note extends native
ownership to retained fleet capacity and serialized workspace aggregation.
The initial phase-only ABI remains an explicit compatibility boundary.
