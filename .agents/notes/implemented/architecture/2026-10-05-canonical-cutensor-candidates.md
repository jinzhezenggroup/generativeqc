# Decision: cuTENSOR eligibility preserves canonical contraction identity

Status: implemented
Date: 2026-10-05

## Problem

#1885 identified useful packing opportunities but created a separate
`tensor-contraction` request containing flattened GEMM dimensions and a
provider-specific descriptor hash. Its FP64 filter also removed other arithmetic
before shared precision selection. Such candidates could not compete with the
existing cuBLAS/generated implementations of the original TensorIR node.

## Decision

Reuse the canonical planned-node adapter for every provider, including physical
arena aliases and the original node/precision identities. Move that adapter into
the existing tensor lowering module so provider implementations do not depend on
one another. Preserve packing bytes and descriptor hashes as execution diagnostics;
they do not become mathematical semantics.

Discover materialized packed binary contractions independently of precision.
For each admitted variant the cuTENSOR adapter returns either a concrete homogeneous
FP32/FP64 offer or explicit rejection. Standalone mixed arithmetic, casts,
refinement and audit obligations remain unsupported until a qualified composite
implementation owns all their work. No variant disappears before selection.

Require explicit availability, matching cuTENSOR 2.x version, device workspace,
retained provider/cache and host-plan bounds. Existing common collection enforces
their simultaneous limits and capture/order requirements. The adapter does not
claim capture safety or exact reduction order. Virtual, overlapping, aliased,
diagonal, triangular and one-sided-reduction cases retain negative evidence.
Reuse one TensorLoweringAdapter across opportunity discovery to avoid repeatedly
hashing the whole program for each site.

## Evidence and limits

The provider/selection suites pass 52 host tests. Coverage includes exact request
equality with incumbent cuBLAS/generated provenance, FP32/FP64 physical views,
retained mixed rejection, absent/invalid resource facts, target/version mismatch,
simultaneous admission, aliases, capture and reduction-order rejection.
Compiler structure and pinned type checks pass. At o=2/v=3, the planning report
finds 75 RCCSD iteration and 160 Lambda-transpose packed sites, accounting for
76,768 and 152,832 bytes of semantic conversion traffic respectively.

This slice does not link or execute cuTENSOR, select a faster plan, or change
generated mathematics. Ready metadata requires native preparation before it can
become executable; complete phase timing remains unknown. Real-device plans,
warm replay, allocation/capture failure gates and production endpoint evidence
remain #1887 work. These traffic counts are not measured time or a speed claim.

## Rejected alternatives

A second operation identity prevents comparison even when equations match.
Dropping non-FP64 requests hides negative precision evidence. Assuming missing
plan/cache bounds are zero would undercharge preparation resources.
