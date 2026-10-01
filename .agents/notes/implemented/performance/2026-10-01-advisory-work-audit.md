# Decision: separate actionable source evidence from runtime performance claims

Status: implemented
Date: 2026-10-01

## Problem

The high-order-loop audit does not find repeated source preparation, helper
allocations, exact structured writes or CUDA synchronization inside replay. A
source inventory that reports every vector declaration or deep loop as a bug
would obscure the real opportunities and mislabel retained reference paths.

## Decision

Add a separate advisory work inventory. Native detection uses a bounded lexical
subset with explicitly located same-file call witnesses. Python structured
materialization uses the standard-library AST and fails closed on unsupported
mutation and escape. Preserve evidence, confidence, disposition, input/scanner
identity and semantic site fingerprints in JSON. CI retains artifacts without
ratcheting unqualified findings.

## Rejected alternatives

- Hard-gating all loop allocations or CUDA waits: setup, deliberate streaming,
  host publication and compatibility paths need different ownership contracts.
- Calling every default vector an allocation or every resize a reallocation:
  capacity is part of the proof; reused or unknown capacity is not a count.
- Claiming unrestricted C++ dataflow or purity from regex: unresolved scopes,
  overloads, reference escapes and floating-point state invalidate that claim.
- Reporting source locations as allocator events or source-copy counts as a
  measured amplification ratio: static and dynamic quantities are different.
- Hardcoding the known ERI filename or Coulomb formula into a general scanner.
  Native simplex support and MP2 relaxed-weight support remain explicit gaps.

## Evidence and invariants

The regression suites cover helper calls, ambiguous/scoped overloads, moved and
reserved vectors, capacity lifetimes, repeated arithmetic blocks, induction
mutation, alias escapes, exact dense negatives, triangles, structured NumPy
writes and line-stable distinct-site identity. An independent adversarial review
provided counterexamples that were retained as regressions.

The first retained scan and manual dispositions are in
`benchmarks/results/work-audit-20261001/`. Real candidates include repeated
streamed DF whitening and tile-local CPU RI-MP2 workspaces. A source-repeated
producer still requires unchanged-input, execution and retention proof; no
native algorithm, numerical expression, allocation lifetime or synchronization
is changed in this tooling PR.

## Revisit when

Promote a narrow endpoint-specific gate only after runtime ownership/counters
and role classification are qualified. Extend native support through a real
C++ AST or compiler-owned schedule metadata rather than expanding permissive
lexical proofs. Exact support must integrate with TensorIR, not introduce a
parallel scientific representation.

## References

- #1628, #1629, #1630, #1631, #1574
- `docs/maintainer/source_work_audit.md`
- `tools/audit_native_work.py`
- `tools/work_audit_python.py`
