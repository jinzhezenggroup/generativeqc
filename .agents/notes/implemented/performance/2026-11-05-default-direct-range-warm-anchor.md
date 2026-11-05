# Decision: enable qualified dense-Direct warm-anchor deferral by default

Status: implemented; promoted-image endpoint qualification pending
Date: 2026-11-05

Supersedes the default-selection policy in
[the original opt-in trial](2026-11-05-direct-range-warm-anchor.md).

## Problem

The matched A100 and RTX 5090 trial removes one unused incumbent Fock from
water96 full-Direct WB97M-V without transferring it to the first warm endpoint.
Cold times improve 9.222% and 7.349%; cold-plus-three-warm times improve 6.152%
and 5.700%. All warm observations remain one iteration/one Fock and pass the
predeclared 1.05 timing guard. Independent energy/force gates pass. Requiring
users to discover an environment switch leaves this qualified benefit unused.
The user explicitly requests default selection after reviewing this evidence.

## Decision

Unset `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY` and explicit `1` both admit dense
full-Direct seeds under the existing guards. Explicit `0` restores the eager
path; other explicit values also conservatively retain eager preparation.
Fitted defaults and partial-fitting exclusions do not change.

This is resource- and owner-based selection, not A100 model-name dispatch.
Require strict FP64 RKS, device-fused XC, an immutable audited replayable range
source, compatible policy/lifetime, and sufficient actual following-force
headroom. Source loss reconstructs the saved density's matching incumbent energy
as counted preparation; force eviction preserves the current stationary token.
No CPU/reference work enters production. Do not change the Hamiltonian,
convergence criteria, force tile, or Direct-versus-fitting contract.

## Evidence and limits

The original [compact publication](../../../../benchmarks/results/direct-range-warm-anchor-20261105/README.md)
retains the matched explicit-on/eager trial and independent reference. It is one
instrumented population per policy/device, not a statistical zero-regression
claim. RTX 5090 warm ratios are 1.0022–1.0084; the A100 native cold endpoint still
trails GPU4PySCF. Do not reinterpret those timings as measurements of the new
default binary.

Qualification of the promoted image must authenticate the changed source and
loaded ELF, exercise unset/explicit-on/explicit-off small-system controls on both
actual scheduled devices, and run one unset-default water96 cold/three-warm
lifecycle per device. Reuse the authenticated independent large reference from
the trial, with provenance stated explicitly. Do not repeat the successful
matched populations or relax numerical/work gates.

## Revisit when

A qualified owner transfers work into a later endpoint, violates independent
scientific gates, or shows a reproducible endpoint regression. Retain explicit
disable and bounded eager fallback rather than broadening unsupported owners.

## References

- Engineering memory: issue #2976; implementation PR #3006.
- The previous fitted default and headroom rationale:
  [guarded fitted anchor](2026-11-01-default-range-warm-anchor.md).
