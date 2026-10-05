# Decision: append denominator selection after landed response controls

Status: implemented
Date: 2026-10-05
Agent: dot

## Problem

The derived-denominator branch used benchmark argument ten for its representation
selector. Landed master now assigns ten through twelve to orbital screening,
J/K profiling and the nuclear-response schedule. Reusing ten would silently
reinterpret a valid master command.

## Decision

Keep DIIS at eight, CCSD Q batch at nine and all response controls at ten through
twelve. Append DERIVED_DENOMINATORS_0_OR_1 at thirteen, default one. The native
force function likewise appends the representation Boolean after the existing
response-options argument. Its RCCSD state call retains the reference-plan
pointer and Q-batch selector before representation. Explicit zero stays available.

Historical source-bound command and timing receipts remain byte-exact. Adapting
an older denominator-branch command requires supplying the three response
controls before its representation token. No numeric-value or arity heuristic
attempts to distinguish incompatible historical layouts.

## Invariants and evidence

Canonical rounding, admission thresholds, fingerprints, fallback and equations
are unchanged. Physical response audits and final exact weights are unchanged.
Live extracted endpoint/default and forwarding fixtures protect existing slots,
all nuclear schedules, explicit true/false, omitted suffixes and malformed tokens.
The current-state CLI is documented in docs/developer/df_ccsdt_gradient.md.

## References

- PR #1916: canonical denominator representation
- PR #1832: landed RHF response controls
- The earlier force-endpoint-diis-response-cli decision records preceding layouts
