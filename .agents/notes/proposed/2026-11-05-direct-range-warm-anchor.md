# Proposal: qualify dense-Direct warm-anchor deferral

Status: opt-in implementation; real-device qualification pending
Date: 2026-11-05

## Problem

The retained A100 water96 full-Direct WB97M-V energy/force endpoint spends
13.210521 seconds rebuilding the incumbent warm-energy baseline after physical
SCF convergence. The accepted immutable range source can already serve warm
replay, but the deferral predicate admits only fitted J/K owners with an exact
occupied factor. Direct owners instead freeze their accepted dense density.

The existing rebasing fallback already restores that density and its own XC
scalars before evaluating the incumbent provider. It does not require a fitted
factor. The opportunity is avoiding an unused provider schedule, not changing
the physical SCF or treating two summation schedules as numerically identical.

## Decision

Extend explicit `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY=1` to full-Direct RKS.
Keep unset Direct owners eager, fitted defaults unchanged and partially fitted
owners excluded. All existing strict-FP64, device-fused XC, source-policy,
identity/lifetime, and following-force-memory guards remain authoritative.

The retained source's own energy and compensation certify its frozen density.
If that source is lost, restore that same density and its saved XC values and
rebuild the incumbent anchor as counted preparation work. Never revoke the
current stationary final-state lease during optional cache eviction.

## Rejected alternatives

- Deleting the incumbent fallback or comparing different summation schedules at
  sub-ulp energy gates would change numerical acceptance.
- Unconditional deferral can merely transfer cold work to the first warm call.
  Preserve the force-envelope guard and inspect each warm call, not only a median.
- GPU-name dispatch is unnecessary: eligibility follows the owner, precision,
  immutable source, and actual available memory.
- Default promotion before matched endpoint and independent science evidence is
  not justified. This proposal changes only explicit opt-in behavior.

## Qualification contract

Use current-master-based, source-authenticated SM80/SM120 Release binaries with
fast-compile off and verified compiler caching. Compare one matched cold and
three-warm lifecycle per policy on actual scheduled A100 and RTX 5090 devices.
Retain public work ledgers and diagnostic traces, including failures. Do not
reuse the old measured image as qualification of current master.

Independent same-grid GPU4PySCF small-system full-Direct E/F gates cover eager,
opt-in, and unset owners, frozen/updating warm seeds, retained replay, policy and
capacity invalidation, and force eviction. Preserve fitted-owner coverage.
Matched large endpoints retain complete construction/preparation/SCF/force/host
export timing, common independent observable gates and actual Fock/VV10 work.
One diagnostic population is not a statistical default-promotion result.

## Evidence

- Initial expanded host lifetime/admission/failure suite: 171 passed, six real
  GPU cases explicitly skipped outside an allocation.
- Independent GPU and complete endpoint qualification is pending.

## References

- Engineering memory: issue #2976; implementation is separate from evidence-only
  PR #3002.
- Existing fitted policy and the rejected first-warm regression:
  [guarded default anchor](../implemented/performance/2026-11-01-default-range-warm-anchor.md).
- Original measured opportunity: A100 LSF job 225489, retained raw trace and
  audit under task-local `.artifacts/issue2987/`.
