# Decision: exact-profile overrides of shared Rys-K defaults

Status: implemented
Date: 2026-10-10

## Problem

PR #2228 established a shared nine-class Rys-K preference for compiled,
non-portable targets that satisfy the target's capability/resource gates.
PR #2226 independently removed architecture-name checks using exact-tuned
manifest preferences, but otherwise selected no preference. The shipped sm_120
and empty portable profiles produced identical artifacts under both policies;
future/custom profiles exposed the incompatible defaults.

## Decision

Preserve the landed shared policy and support optional exact-tuned overrides.
An absent `preferred_rys_task_fock_shell_classes` field means use the shared
nine-class default. A present list replaces that set; an empty list explicitly
opts out. Only exact tuned profiles carry overrides. Compatible/non-tuned
profiles discard the source profile's override and use the shared default.
Portable profiles always retain the incumbent.

Explicit override classes must all occur in the actual target's generated
candidate inventory, or generation fails. Shared defaults continue to intersect
that inventory, retaining incumbent fallback when resources/classes are absent.
This deliberately supersedes #2226's proposed rule that unqualified profiles
always retain the incumbent. It does not undo #2228's target-independent policy
or require repeated full scientific qualification for each GPU marketing model.

## Rejected alternatives

- Selecting either side of conflict markers would silently discard the other
  behavior; choosing the older exact-only policy would reverse landed defaults.
- Treating an empty list as absent would defeat explicit opt-out.
- Unioning an override with shared defaults would defeat measured subsets.
- Carrying overrides through compatibility would transfer target-specific
  performance choices without evidence.

## Invariants and validation

Keep actual-target scheduling, portable fallback, explicit unsupported-class
rejection, and shared unsupported-class fallback. No architecture-name branch,
GPU probing, new recurrence mathematics, or runtime autotuning is introduced.
Regression tests cover missing/empty/subset/additional-capable exact overrides,
compatible and non-tuned resolution, portable profiles, unavailable classes,
and unsupported target resources. Promotion-inventory mutation controls audit
both the shared policy and override contract. Existing shipped manifest class
choices and generated arithmetic remain unchanged.

## Revisit when

Measured complete-endpoint evidence justifies changing a specific profile's
class choices, or capability/resource changes alter admission. Device-specific
performance claims still require matching evidence.

## References

- PR #2226: https://github.com/jinzhezenggroup/generativeqc/pull/2226
- PR #2228: https://github.com/jinzhezenggroup/generativeqc/pull/2228
- Current contract: `docs/developer/direct_rys_tasks.md`
