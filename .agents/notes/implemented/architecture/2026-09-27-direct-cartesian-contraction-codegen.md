# Decision: generate Direct Cartesian and generic contraction support

Status: implemented
Date: 2026-09-27

## Problem
The generic Cartesian primitive evaluator and generic contracted-ERI adapter
remained 325 nonblank/noncomment lines of maintained scientific CUDA after the
order-specific recurrence stack moved under compiler ownership.

## Decision
Emit both qualified headers at build time from one scientific-compiler owner.
Weighted-ERI and generated shell/source paths consume the generated Cartesian
header; J/K, cached-tensor, reference-force and packed-Fock consumers use the
generated contraction header.

## Rejected alternatives
Duplicating Cartesian evaluation into every consumer was rejected because it would
recreate formula ownership. Removing the generic contraction fallback was rejected
because unsupported/public-AO and reference consumers still require it.

## Invariants
- Preserve Cartesian/Hermite/Coulomb arithmetic and reduction order.
- Preserve range-separated behavior and mixed-precision fail-closed semantics.
- Do not change screening, selectors, queue topology, precision policy or schedules.
- Exact-head CUDA compile and endpoint qualification remain merge gates.

## Evidence
Structural regressions require all production consumers to use generated headers
and require both source-tree scientific owners to be absent.

## References
- #351
- #356
- #1468
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol
