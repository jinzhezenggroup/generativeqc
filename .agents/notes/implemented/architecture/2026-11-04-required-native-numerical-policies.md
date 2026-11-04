# Decision: require selected native numerical policy queries

Status: implemented
Date: 2026-11-04

## Problem

Native owners already determine budget-Fock controls, completed mixed work,
strict returned-state replay, standard fallback and correlated stopping limits.
Missing query symbols still selected duplicate Python arithmetic. CBS metadata
also maintained a separate optional-library discovery/cache, so installation age
could change the authority used before actual solver admission.

## Decision

Require the existing precision-controls, precision-admission and correlated
convergence queries in their consumers. Delete all four Python precision-policy
fallbacks and both correlated stopping-policy implementations. CBS standalone
planning uses the normal native loader; Calculator-owned queries use only their
selected library. No old-library alias, scientific whitelist or silent strict
metadata fallback remains for these policies.

Fast verifies native admission support during construction, before molecular
work, and verifies the selected query again when consuming returned evidence.
Unsupported numerical/provider domains with a current native owner still use
the existing bounded strict or independent-audit paths. Missing protocol support
is an installation mismatch, not evidence that an unsupported arithmetic route
has been scientifically admitted.

Unresolved KS options have no admitted model or plan facts. The controls adapter
passes that absence to the native owner instead of reading a plan property that
would require compiler resolution just to establish non-admission.

## Validation boundaries

- Preserve native C++/compiler equations, thresholds and caller ceilings.
- Compare correlated metadata against a frozen test-only pre-cutover oracle,
  including randomized allowances and exact one-ULP bracket neighbors. This is
  metadata equivalence, not an independent chemical error certificate.
- Exercise actual compiled host predicates with injected molecular observations;
  unit fixtures may not duplicate scientific admission arithmetic.
- Reject missing protocols, selected-library replacement, native refusal and
  malformed receipts without rediscovery or a Python decision.
- Preserve unknown counter states, returned-state/generation binding, explicit
  observable audits and separate native correlated execution capability gates.
- Retain deleted locations as historical source evidence, not live anchors.

## Remaining ownership

This removes numerical policy duplicates but does not move precision
candidate/auditor/fallback execution, observable comparison, independent seed
history, CBS composition, universal resource/provider lifetime or general
response/HVP into the native owner. Parent #2227 remains incomplete. Real-device
execution is not established by host CUDA metadata or synthetic endpoint tests.

## Rejected alternatives

- Keep unreachable compatibility functions: obsolete scientific policy would
  remain an attractive second authority and hinder provable retirement.
- Missing-query FP64 audit: changes ownership according to installation age and
  conceals an unsupported runtime rather than an unsupported numerical domain.
- Source-body/hash tests: constrain refactoring without proving actual dispatch,
  failure propagation, lifetime or scientific endpoint behavior.

## References

- #2227 R4/R6; the separate SCF-default retirement in #2968.
- `tests/python/test_native_correlation_policy.py`
- `tests/python/test_native_precision_controls.py`
- `tests/python/test_native_precision_admission.py`
- `docs/user/ccsdt_cbs.md`
- `docs/user/application_precision.md`
