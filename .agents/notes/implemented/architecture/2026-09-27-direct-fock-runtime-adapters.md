# Decision: make order-two and quartet Direct-Fock adapters runtime-only

Status: implemented
Date: 2026-09-27

## Problem
After recurrence/source-contraction and Fock-scatter retirement, the order-two and
generic quartet adapters remained classified as scientific because they still
contained duplicated shell ordering and inline AO Schwarz threshold arithmetic.

## Decision
Reuse the compiler-generated stable shell canonicalizer for order-two tasks and
centralize the AO-level Schwarz gate in the existing scientific screening-policy
owner. The two Fock adapter headers then retain only task/tile decoding, component
mask packing, generated-class dispatch and calls into compiler-owned integral/Fock
math, so their ownership becomes runtime.

## Invariants
- Screening inequality and tolerance are unchanged.
- Generated class-mask semantics and task ordering are unchanged.
- Integral arithmetic and Fock scatter remain in their existing generated owners.
- No performance or capability selector changes are introduced.

## Evidence
Structural tests reject private Schwarz arithmetic and private order-two shell
canonicalization in these adapters and require runtime ownership.

## References
- #356
- #1470
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol
