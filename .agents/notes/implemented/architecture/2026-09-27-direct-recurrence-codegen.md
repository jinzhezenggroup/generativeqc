# Decision: generate Direct order-2/3/4 recurrence support

Status: implemented
Date: 2026-09-27

## Problem
Order-2, order-3 and order-4 primitive ERI recurrence headers plus shell-class
dispatch remained about 600 nonblank/noncomment lines of handwritten scientific
CUDA even after force consumers and source contraction moved under compiler
ownership.

## Decision
Emit those four qualified headers at build time from one scientific-compiler owner.
The source bodies are migrated without changing arithmetic or workspace order.
Native pair/Hermite helpers remain explicit dependencies for later retirement.

## Rejected alternatives
A new Direct-only recurrence IR was rejected because #351/#682 already define the
shared recurrence/compiler direction. Replacing the fallback with only promoted
AOT classes was rejected because unsupported and Schwarz/Fock consumers still
need the qualified generic domain.

## Invariants
- Preserve order-2/3/4 arithmetic, workspace sizes and reduction order.
- Preserve existing pair/Hermite helper semantics and Boys numerical policy.
- Do not alter selector, screening, queue, precision or scheduling behavior.
- Exact-head CUDA compile, numerical parity and resource gates remain required.

## Evidence
Structural tests lock generated entry points, source-tree owner deletion, consumer
include migration and CMake generation dependencies.

## Revisit when
Move the remaining native pair/Hermite/cartesian support to shared compiler
representations only when dependent consumers have equivalent qualified lowering.

## References
- #351
- #356
- #1465
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol
