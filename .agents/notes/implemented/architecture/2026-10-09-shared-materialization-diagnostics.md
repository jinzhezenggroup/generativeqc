# Decision: share materialization policy, preserve analyzer evidence ownership

Status: implemented
Date: 2026-10-09

## Problem

Native exact-support and Python source audits had separate output/advice, while
TensorIR complexity described only shape-derived storage. MP2 aggregate writes
are outside the native closed proof subset; a visible dense allocation and a
factorized alternative are not an aggregate alias, ABI, or routing proof.

## Decision

The stdlib-only `common.materialization` adapter owns one evidence-to-diagnostic
policy. Native and Python parsers remain the evidence owners. The common work
audit calls the existing strict structured audit path and retains its source and
scanner byte identities, unknown states, exact domains, union upper bounds, and
source-role census. All analyzers consume the same captured source bytes without
reopening paths. Filesystem state is explicitly a scan-time observation. The
adapter itself is hashed as consumed scanner code; its import-bound digest is
verified at report emission so an edited file cannot relabel stale executable
policy with new source bytes.

TensorIR's explicit `ComplexityReport.materialization_diagnostics()` reports
logical dense elements and retained-output layout. It never infers zeros from
shape or factorization. The ordinary summary remains unchanged because it is
already included in optional optimizer/preparation provenance.

## Rejected alternatives

- Another parser or support algebra: duplicate proof engines would diverge
- Whole-program MP2 alias/ABI reconstruction: unjustified by source-role anchors
- Treating unknown consumer layout as a dense requirement: absence is not proof
- Injecting diagnostics into equation/normal-preparation provenance: unnecessary
  identity churn for an advisory report
- Reusing the common audit's former weaker traversal: lost missing-path,
  containment, canonical-source, alias, and unknown-Git guarantees

## Invariants

Only exact single-address-domain certificates earn a conservative representation
review recommendation. Full-domain writes never earn sparse advice. Union bounds
are never converted to exact counts or ratios. MP2 role census proof flags remain
false; observed helper expressions never become certified element counts or
growth degrees from source-role anchors alone. No numerical threshold, automatic rewrite, native/scientific modification,
GPU qualification, or performance claim is introduced.

## Evidence

`tests/python/test_materialization_diagnostics.py` exercises common-entrypoint OV,
full-domain, lower-triangle, unsupported and union-bound sources; real MP2 and
changed/missing/malformed roles; shared Python and TensorIR policy; exact scanner
and source hashes; strict path selection; unknown Git evidence; stdlib-only CLI;
and equation, AD and CPU-preparation immutability.

## Revisit when

Structured TensorIR support or a new analyzer can supply a scoped, validated
certificate. Extend evidence inputs and the shared policy without silently
broadening an existing certificate or rewriting equations.

## References

- Issues #1631, #1574, and parent #1580
- Existing native proof rationale:
  `../performance/2026-10-07-native-structured-materialization-subset.md`
- Source-role census introduced by PR #2142
