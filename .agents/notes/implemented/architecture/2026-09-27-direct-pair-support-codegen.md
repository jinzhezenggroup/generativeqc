# Decision: generate Direct pair and Hermite support

Status: implemented
Date: 2026-09-27

## Problem
Low-order pair expansions and exact shell-pair Hermite recurrence remained three
maintained scientific CUDA headers after primitive order-2/3/4 recurrence moved
under compiler ownership.

## Decision
Emit the existing pair-order-2, pair-order-3 and shell-pair-Hermite helpers from
one build-time compiler owner. Generated primitive recurrence headers consume
these generated dependencies directly.

## Rejected alternatives
Duplicating the pair formulas inside each generated primitive-order header was
rejected because it would recreate source and compiler ownership duplication.
Removing the helpers entirely was rejected because current generated recurrence
still depends on their exact sparse/workspace semantics.

## Invariants
- Preserve sparse term ordering, Hermite workspace bounds and arithmetic order.
- Keep Boys policy, screening, selectors and schedules unchanged.
- Do not introduce Direct-only optimizer decisions.

## Evidence
Structural tests require generated recurrence output to contain no retired native
pair/Hermite includes and require the source-tree owners to be absent.

## References
- #351
- #356
- #1466
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol
