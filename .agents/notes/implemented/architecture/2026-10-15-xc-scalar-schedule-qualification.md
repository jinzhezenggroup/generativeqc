# Decision: qualify XC scalar optimizations by consumer metadata

Status: superseded by the generic consumer policy; original #2506 rationale retained below
Date: 2026-10-15

The user rejected moving a method-specific exception into manifest metadata as
insufficient generalization. The consumer-only replacement and its independent
qualification gates are tracked in
`.agents/notes/implemented/performance/2026-10-15-generic-xc-force-powers.md`. The original
admission decision below is historical, not the current schedule contract.

## Problem

PR #2505 independently admitted bounded integer-power lowering for FP64
M06-2X RKS stationary forces, but its activation was a method-name comparison
inside `geometry_cuda.py`. That coupling made further method/spin admission
require editing the force emitter and risked hiding the negative global SCF
experiment behind an overly broad compiler default.

## Decision

Retain `Graph.lower_small_integer_powers` and the existing
`emit_split_hybrid_device_body(..., lower_integer_powers=False)` default.
Declare the *qualified* consumer, method, spin and precision combination
in `manifests/cuda_split_hybrids.json`. Generate the immutable code-keyed
schedule set through the existing registry generator, then select the scalar
schedule with `xc/scalar_schedule.py` at the force consumer boundary.

The existing FP64 M06-2X RKS force schedule is the only admitted entry.
Unknown methods, SCF point consumers, UKS, float, and MN15 retain the original
scalar operation ordering. Invalid or duplicated manifest admissions fail
during generation. No user-visible CLI flag, new SCF path, or new math
transformation is introduced.

## Rejected alternatives

- Unconditionally enabling small-integer lowering for all split-hybrid point
  graphs: the retained historical DF96 frozen warm regression increased
  Fock builds from three to four and complete time by 5.8%.
- Relaxing convergence gates to restore the third DF96 build: unsupported;
  the original study does not isolate which SCF gate was crossed.
- Keeping the functional-name comparison in geometry emission: it turns
  each future qualification into a force-emitter code change.

## Invariants and evidence

- This is an admission/ownership refactor. The qualified FP64 force emitter
  receives the same boolean as #2505; the default SCF emitter receives false.
- Existing independent Libxc, finite-difference and full E+F qualification in
  #2505 remains the performance evidence. This change makes no new speed claim.
- The registry generation freshness test, manifest rejection tests, consumer
  schedule matrix and existing force scalar tests protect selection semantics.
- Changing the scalar arithmetic in SCF, admitting a new precision/spin/method,
  or addressing the DF96 iteration instability requires separate numerical and
  complete-endpoint validation.

## References

- PR #2505 and `.agents/notes/implemented/performance/2026-10-15-m06-rks-force-powers.md`
- `.agents/notes/rejected/2026-10-14-mgga-integer-power-defaults.md`
- `tools/generate_xc_split_hybrid_registry.py`
