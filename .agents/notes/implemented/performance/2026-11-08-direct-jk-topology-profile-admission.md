# Decision: Resolve Direct J/K residency before topology packing

Status: implemented (structural migration; calibrated-profile and GPU qualification pending)
Date: 2026-11-08

## Problem

Issue #597's source inventory still identified three native performance constants.
`resolve_small_hf_profitability` already owned an AO crossover, but topology,
resource-only admission and execution independently repeated an AO=16 gate.
Changing only the resolver could therefore charge the wrong arena or omit a
spherical-to-Cartesian transform needed by the direct fallback. Persistent queue
allocation, reset and dispatch also duplicated compiler scheduling policy in a
native constants header.

## Decision

- Resolve resident-ERI eligibility in the actual host packer using the shared
  small-HF policy. Store the decision in `HostBatch` and compare it in immutable
  topology identity. Execution consumes that decision, retaining the existing
  energy-only/no-physical-reference guard.
- Query runtime target facts once at the bucket lifecycle boundary, before
  packing, through the numerical owner's provider bridge, and pass those same
  facts to execution. Preserve the lifecycle/provider import boundary and run
  provider-initialization preflight before this probe. Do not add a second
  visibility-sensitive device query on every replay.
- Retain the AO=16 no-calibration crossover. Bound resident ERI bytes by an
  explicit profile ceiling, the existing fixed-topology 1/32 device-memory
  policy, and the conservative 256-MiB unknown-target fallback. RHF/UHF share
  one tensor per system, not one per spin state.
- Emit persistent force/Fock order domains from a compiler-owned immutable
  profile alongside the resident-PSSS schedule. Allocation, reset and dispatch
  all consume those emitted constants. Keep the default seven/six domains;
  smaller test profiles retain fixed-grid fallbacks and cannot expand into
  unqualified higher-register-footprint orders.

## Rejected alternatives

Deleting the AO gate without migrating topology can produce a direct route with
no required Cartesian transform. Treating an n^4 byte estimate as performance
calibration ignores screened shell work and expected SCF reuse. Selecting another
queue schedule by GPU name or SM count alone is not profitability evidence.
Moving symbols into generated output does not establish runtime profile
activation; the inventory explicitly retains that remaining work.

## Invariants

Scientific screening, equations, FP64/FP32 policy and force accuracy gates are
unchanged. Device-inert resource queries do not probe CUDA. A route change
invalidates cached topology even when the basis and transform contents match.
The bounded staging cache is still retained explicitly, not misclassified as an
unresolved performance defect. No production profile or schedule is promoted.

## Evidence and limitations

Base source: `2bf3fd98d` (current origin/master at task start). Compiled host
tests exercise the real packer, policy, topology comparison and arena with RHF
and UHF, exact/rejected memory boundaries, batch tensor accounting, overflow,
unknown/synthetic targets and smaller compiler queue domains.

Qualification on 2026-11-08:

- 605 host tests pass; four explicit device opt-ins remain skipped in that
  host-only suite. Compiler/SCF ownership audits, the tuning inventory checker,
  Ruff and clang-format checks pass.
- CUDA 12.9.86 compiles the actual angular Fock, angular force and compaction
  translation units for `sm_80`. Default and reduced-domain compaction plus the
  native queue test compile/link for `sm_120`. Compilation uses verified
  sccache 0.16.0; before/after statistics and logs remain in ignored
  `build/issue597/`.
- Local `node3` is drained and is not used for GPU work. The permitted remote
  nodes are queried separately. On `node5`, Slurm job 2023 runs the actual
  Fock/optional-FP32 queue reset with default force/Fock domains 7/6 and reduced
  domains 3/3. Both pass, including unchanged guard words and omitted-FP32
  storage. Job 2024 repeats both under Compute Sanitizer memcheck: zero errors
  for each. Both jobs inherit `CUDA_VISIBLE_DEVICES=0` from Slurm and request
  `main`, `gpu:5090:1`, one task and an explicit three-minute limit.

This is queue/storage validation, not independent energy/force or endpoint
performance qualification. Issue #597 must remain open until canonical
calibrated-profile admission/activation and matched complete RTX 5090 endpoint
qualification are delivered. Synthetic selection and device queue tests are not
replacements for those measurements.

## Revisit when

A compatible measured profile includes actual screened-quartet work, expected
Fock reuse, geometry rebuild frequency, provider setup/workspace and complete
memory accounting, with independent numerical gates and matched endpoint
evidence for the selected route. Extend queue domains only with the corresponding
force/Fock endpoint evidence and retained bounded fallback.

## References

- Issue #597.
- [Earlier target-resource policy](2026-09-21-direct-jk-target-resource-policy.md).
- [Current policy contract](../../../../docs/maintainer/performance_engineering.md#direct-jk-policy-ownership).
- `manifests/maintenance/direct_jk_tuning_inventory.json`.
