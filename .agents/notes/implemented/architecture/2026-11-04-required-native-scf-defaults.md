# Decision: require the selected native SCF-default policy

Status: implemented
Date: 2026-11-04

## Problem

The native catalog/grid default-controls query already owns the admitted policy,
but a missing query still selected a duplicate Python model/grid policy through
an older version-only capability. Calculator and precision consumers could
therefore use different policy owners depending on installed library age. The
duplicate also resolved scientific model identities in the Python runtime.

## Decision

Require `generativeqc_scf_defaults_query_v1` in the selected library. Migrate
Calculator construction, precision validation/fallback and native precision
context transport directly to `_scf_defaults_native.default_controls`. Delete
the obsolete `_scf_convergence` controller and its model/grid selector rather
than retaining an alias or an unreachable compatibility implementation.

Missing protocol support is an actionable installation error. Native query
failure and invalid receipts remain errors; no strict-default or version-only
Python retry exists. Native qualified and strict thresholds, explicit caller
overrides, copied-answer caching and physical final-state checks are unchanged.
The native physical-convergence capability symbol itself is not removed by this
Python ownership change.

## Historical source evidence

The ownership inventory retains removed locations under `retired_python_sites`,
with their last-present source revision, replacement and rationale. They are not
live call sites and do not require obsolete functions/files to remain in the
current tree. The original baseline SHA and all tracked domains are retained;
retirement does not imply full R4/R6 or parent #2227 acceptance.

## Rejected alternatives

- Keep a wrapper or reference-only copy of the former policy: no production
  consumer needs that interface and it is not an independent numerical oracle.
- Silently use strict defaults for an old library: it conceals installation
  mismatch and preserves a second scientific policy authority.
- Replace native admission with a Python hash/model whitelist: complete-model
  identity and default-grid selection already belong to the native owners.

## Invariants

- Every successful default selection comes from the selected native library.
- Explicit tolerances and unsupported-domain native strict controls are preserved.
- Failure, library/query replacement and malformed receipts cannot use cached or
  legacy answers as a fallback.
- Test metadata doubles inject fixed receipts, not another model/grid policy.
- Precision candidate/auditor scheduling, general resource/lease authority and
  remaining response/HVP migration are not delivered by this change.

## Validation

Behavioral tests cover missing support despite a legacy capability, native
failure, malformed receipts, library/query replacement, detached cached answers,
explicit overrides and precision profile consumers. Existing compiled adapter
tests retain real native model/grid admission and concurrent binding retirement.
Independent physical and endpoint qualification remains separate from these
host-only metadata/ownership checks.
