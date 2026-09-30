# Decision: native high-order source audit fails only proven rank-2 matrix chains

Status: implemented
Date: 2026-09-30

## Problem

TensorIR can now report symbolic storage/work degree and legally reassociate
n-ary contractions, but native scientific C++ can still bypass that ownership.
Issue #1576 exposed three rank-2 AO/MO transforms written as quartic scalar loops.
A source-only regression tied to those three function names prevented exact
reintroduction but could not discover the same algebra elsewhere.

## Decision

Add a repository-wide native source audit over non-vendored `src/` code. The
audit inventories leaf loop nests of depth four or greater and applies a
conservative matrix-chain classifier. CI fails only when a high-order reduction
writes a rank-2 target from at least three pairwise-indexed inputs spanning four
loop indices and no three-/four-index source access is present.

All other high-order nests remain report-only. The inventory distinguishes
high-rank output materialization/permutation passes, genuine high-rank source
contractions, fixed-extent inner loops, and an unclassified fallback. It also
reports an effective depth with simple fixed extents removed, so constant spin or
Cartesian/component loops do not masquerade as additional asymptotic powers.
These classes are review prompts, not proof that method scaling can be reduced.
The generic compiler remains the owner for proof-carrying TensorIR reassociation.

## Rejected alternatives

Failing every new four-deep loop was rejected because exact ERI/J/K and other
method-defined four-index contractions are legitimate high-order work. Blind
source-to-source rewriting was rejected because native loop syntax does not
carry enough scientific semantics to prove an equivalent lower-order method.

## Invariants

- Never claim exact-exchange, CC, or ERI scaling was reduced from loop depth alone.
- A direct rank-2 matrix-chain regression must fail before merge.
- Genuine high-rank source accesses must remain reportable without being
  mislabeled as reducible.
- Vendored xTB native code is excluded by default and can be included explicitly
  for diagnostic scans.

## Evidence

`tests/python/test_native_complexity_audit.py` contains a synthetic quartic
`C^T A C` case that must be rejected and a genuine dense four-index ERI-to-Fock
contraction that must remain report-only. The production-tree test requires no
matrix-chain candidates. The pre-commit `native-complexity-audit` hook runs the
same repository scan on every change.

## Consequences

The check is intentionally heuristic and conservative. It can miss algebra
written through opaque helper calls, but false claims of reducibility are more
damaging than incomplete discovery. The JSON inventory provides a stable input
for future source-audit extensions.

## Revisit when

Extend or replace this scanner when native scientific regions carry explicit IR
metadata, when a C++ AST pass can recover index semantics safely, or when a new
reducible pattern is demonstrated with an independent algebraic oracle.

## References

- #1580
- #1576
- #1587
- #1591
- `docs/maintainer/performance_engineering.md`

---

Agent: ChatGPT
Model: GPT-5.6 Sol
