# Decision: bind external stationary sources in the source cache recipe

Status: implemented
Date: 2026-10-28

## Problem

Integrating source reuse with the updated native force owner added an
`include_external` choice for range-exchange and VV10 derivative sources. The
uncached wrapper respected that choice, but the cache adapter initially omitted
it from both forwarding and recipe identity. A cached composite wrapper could
therefore have a different source layout from its uncached counterpart.

## Decision

Forward the selection through native, ordinary and cached wrapper generation,
and bind it alongside reduction and staged-point choices in the source recipe.
Preserve target-independent source reuse, downstream architecture-specific
binary identity, and integrity failures on altered cached bytes.

## Validation

The composite regression uses WB97M-V to prove that complete and semilocal-only
source layouts differ, each cached output equals direct generation, and a cache
hit preserves the complete output. Existing reconfiguration tests exercise the
20-second discovery limit with the restored catalog-enumeration optimization;
no timeout or numerical acceptance gate is raised. Historical AOT receipts stay
bound to their original compiler revision and are not relabelled as this merge.

## Rejected alternative

Relying on the scientific plan alone is insufficient: the same plan can be
lowered with external sources delegated or included. Both selected behavior
and the plan must participate in cache identity.
