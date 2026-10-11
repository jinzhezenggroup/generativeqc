# Decision: native per-item automatic cold starts

Status: implemented
Date: 2026-10-11

## Problem

The model/element query in the [earlier decision](2026-10-11-native-automatic-minao-policy.md)
still left Python selecting a shared explicit MINAO descriptor. One unsupported
neighbor disabled the entire batch. The planner also lacked population admission,
so a zero-electron system could be charged for a seed that native MINAO rejects.

## Decision

Add an AUTO_CPU provider ID and independently discoverable capability without
changing the options structure or schema. Shared native preparation resolves the
candidate to MINAO or no preliminary policy for each immutable system topology.
HF single plans, HF fleet items and KS single/batch plans use those resolved
controls for both execution and warm-retry diagnostics and allocation semantics.
Selection reuses explicit MINAO admission; projection and target equations are
unchanged. Invalid controls still fail before a domain fallback can hide them.

For that positively discovered CPU path, a basis package's unused ECP records
are not a constructor-wide veto. The actual target's ECP cores/terms determine
per-item admission. Explicit requests and CUDA/old-library constructors retain
their previous conservative basis-level ECP checks.

Add a population-aware allocation-free system query using the existing request
structure rather than extending its v1 layout. Python marshals normalized topology
and records input-ordered resolved kinds in the resource schedule identity. It
charges retained seeds only for selected systems, with the existing serialized
workspace bound. This is not complete native resource/lease authority.

## Rejected alternatives

- Infer per-item execution from schema 1 or MINAO's old bit: old providers use a
  shared explicit policy and would reject mixed batches rather than fall back.
- Apply this CPU gate to CUDA without real-device evidence: keep the separately
  qualified conservative CUDA path until scheduler-backed numerical/work gates.
- Ignore malformed controls on ineligible systems: that would hide user errors
  and make validation depend on neighboring topology.
- Copy the MINAO projection or target equations: existing native owners already
  provide scientific execution, density admission and bounded retries.

## Invariants

Explicit policies remain fail-closed. Existing/imported/warm density is
authoritative. One item's fallback never changes another item's policy, retries,
or retained state. Resource discovery performs no scientific/device work.
Target admission remains responsible for invalid methods, spin and basis.

## Evidence

Native policy tests cover discovery, populations, malformed descriptors and
controls. Installed-compatible C++ batch tests exercise mixed He/Ca2+ cold/warm
execution for RHF/PBE and explicit rejection. Python endpoint tests extend this
to RHF/PBE/PBE0 mixed ECP/all-electron bases, moved coordinates, failed neighbors and retained-state replay,
plus exact/below-limit resource admission and planner-query effect tests.
Acceptance is limited to this CPU policy slice, not the complete #2227 tracker.

## Revisit when

CUDA per-item policy is independently qualified, resource leases become native,
or a different automatic provider needs a new execution-policy version.

## References

- #2227 R2/R4/R6 and PR #2285
- `docs/user/initial_guesses.md`
- `src/scf/preliminary_guess.cpp`
- `tests/native/test_cpp_batch.cpp`
- `tests/python/test_minao_default.py`
