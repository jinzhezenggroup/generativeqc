# Candidate: bounded default DF charge partitions for long SM 12.0 traversals

Status: proposed; default candidate under independent and endpoint qualification
Date: 2026-10-24

## Motivation and boundaries

Stage24's source-matched water96 r2SCAN-3c DF profile still contains long
single-target Hermite charge traversals. CUDA kernel timing identifies the
charge classes directly; inclusive CPU sampling of an XC entry point may also
include waiting for these preceding stream operations and is not exclusive XC
time. Earlier explicit split-32 measurements already establish a useful endpoint
candidate, but an explicit environment override is not a default work policy.

## Candidate policy

With no override, only SM 12.0 and source angular classes with at least 8192
orbital primitives may request 32 partitions. Such a class traverses at least
256 source groups with the existing 8..32 source lanes. Smaller classes and
other devices retain one partition. Freeze device identity and the override
at owner creation; graph replay performs no device query or environment read.

Cap the request by complete partials fitting the existing dead orbital-potential
allocation. Zero or one partial capacity retains the original auxiliary-output
consumer. The raised force potential and source density remain disjoint/live.
Explicit 1..64 overrides preserve their previous semantics, including rollback
with 1 and malformed-control rejection. No new device storage, larger budget,
integral omission, screening, metric, precision or convergence change is made.

The kernel and ordered partial reducer are unchanged. Different summation
grouping is a numerical change: long signed-contraction libcint charge/J and
displaced-derivative controls, graph zero/change replay, constrained ownership,
sanitizers and complete molecular E/F gates are required before promotion.
Measured source and paired whole-endpoint evidence must remain explicit.

## Revisit conditions

Broader devices, shorter traversals or other partition counts require their own
complete-endpoint evidence. Never infer useful speed from occupancy or a source
count alone, and retain negative endpoints alongside successful observations.
