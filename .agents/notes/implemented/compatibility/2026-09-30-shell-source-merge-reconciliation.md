# Decision: retain one stationary shell-source reduction

Status: implemented
Date: 2026-09-30

## Problem
PR #1573 and the landed full-range producer overlap in the ordinary stationary driver. Combining source seeding with the older host addition would count J/K twice. Resident nonlocal and geometry-only-reset additions must also survive reconciliation.

## Decision and invariants
Use the checked, one-shot source-seed join and the existing final device reduction. Retain the published full-range compatibility producer, resident density and nonlocal seed APIs. Do not reconstruct exchange coefficients or integral mathematics. Preserve missing-capability/NOT_IMPLEMENTED fallback, token checks, bounded work admission, and failure isolation. A geometry-only reset closes the source-seed gate; only a full density reset opens it.

## Evidence
Run the prepared-shell adapter/driver tests, native seed probe, direct-owner guards, geometry reset, bounded executor, merge-boundary and all capacity mutation gates. Update only the inspected endpoint-owner and whole-native-header fingerprints; all numerical limits and individual source-block contracts remain unchanged. No NVIDIA numerical or performance qualification is claimed.

## References
PR #1573; issues #1477 and #1423.
