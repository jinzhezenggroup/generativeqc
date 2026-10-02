# Decision: retain TensorIR diagnostic controls through lowering

Status: implemented
Date: 2026-10-01

## Problem

TensorIR planners and emitters call shared backend preparation on their inputs.
Exposing pass selection only on an earlier optimizer/preparation call therefore
does not suffice: a later default preparation silently runs the complete pipeline
and replaces the diagnostic selection. A stopped multiply was folded to a
constant by `plan_cuda` before code generation.

## Decision

Default backend preparation inherits recorded diagnostic pass controls from its
input program. It still invokes the shared pass manager, including validation of
pass names and dependencies. Explicit non-default controls replace the inherited
selection. Calling `optimize(program)` without controls explicitly resets to the
canonical pipeline. Programs without diagnostic selections retain the existing
production behavior and identity.

## Rejected alternatives

Bypassing shared preparation inside individual emitters would duplicate compiler
policy and weaken the production ownership boundary. Adding private optimizer
implementations to diagnostic consumers was likewise rejected.

## Invariants and evidence

Tests exercise repeated CPU/CUDA/portable/scalar preparation, actual CUDA planning,
CPU/scalar emitted multiplication, explicit canonical reset, and unknown-pass
rejection. Diagnostic metadata remains separate from mathematical equation
identity; optimizer identity continues to describe the selected pass sequence.
No numerical algorithm, backend capability, or default schedule is changed.

## Revisit when

Revisit propagation if the shared pass manager acquires non-idempotent stages or
an explicit prepared-artifact protocol replaces repeated backend preparation.

## References

- PR #1676 and issue #682
- `tests/python/test_compiler_pass_manager.py`

Agent: dot
