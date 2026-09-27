# Decision: generate Direct order-two shell contraction

Status: implemented
Date: 2026-09-27

## Problem
The exact order-two shell contraction remained a 225-line maintained scientific
CUDA owner after its primitive recurrence and pair expansions moved to generated
compiler ownership.

## Decision
Emit the existing order-two shell contraction at build time from the scientific
compiler and switch source-contraction and Fock order-two consumers to that
generated header.

## Invariants
- Preserve component masks, arithmetic/reduction order and workspace bounds.
- Preserve existing generated order-two primitive recurrence dependency.
- Do not change screening, selectors, queue topology, Boys policy or schedules.
- Exact-head CUDA and endpoint qualification remain merge gates.

## Evidence
Structural tests require both consumers to include the generated header and require
the former native owner to be absent.

## References
- #351
- #356
- #1467

Agent: ChatGPT
Model: GPT-5.6 Sol
