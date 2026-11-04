# Decision: automatically admit bounded occupied-XC initializers

Status: implemented
Date: 2026-11-05

## Problem

The recovered shortcut in the [audited-proposal decision](2026-11-04-occupied-xc-audited-proposals.md)
remained an opt-in even after complete-endpoint qualification found a profitable
domain. Requiring an environment override would leave ordinary eligible users on
the slower matrix initializer. Default promotion must change selection, not
silently weaken physical acceptance or relabel historical opt-in measurements.

## Decision

Unset `GENERATIVEQC_CUDA_KS_OCCUPIED_XC` now permits automatic admission, as
does exact `1`. Explicit `0`, empty and malformed values retain matrix-only
work. The existing native owner still decides whether a canonical frame,
strict-FP64 meta-GGA layout, occupied rank, scratch lease and residual regime
permit proposals. Unsupported and near-stationary cases keep original-D work.

The one-panel non-certifying projection and bounded initializer lifecycle are
unchanged. A permanent original-D target handoff starts a fresh energy epoch;
only the strict physical/canonical gates certify convergence and force
publication. StrictAudit and response consumers never use provisional features.
Neither default selection nor explicit `1` bypasses these invariants.

## Rejected alternatives

- Retaining an opt-in after qualification hides the recovered benefit.
- Restricting the default to a hardcoded molecule size or benchmark method would
  substitute an example for the actual admission conditions.
- Calling historical explicit-1 timings default-path evidence obscures which
  native selection branch was exercised. New receipts record actual absent
  environment values and retain backward-compatible v1 replay.
- Requiring per-point full matrix recomputation restores the cost removed by
  the optimization, without replacing strict endpoint acceptance.

## Evidence and revisit conditions

The earlier source-bound opt-in qualification remains historical evidence, not
a measurement of this new selector. Default-path qualification must compare
actual unset controls with explicit `0`, include complete native cold/warm/
moved E+F and semantic work, and preserve independent and paired scientific
gates. Host execution tests cover unset, exact 1, 0, empty and malformed values;
native tests exercise automatic and explicit admission and a strict matrix tail.

Revisit automatic admission on a credible endpoint regression, not an isolated
kernel timing. Retain the explicit rollback and original-D publication contract.
This supersedes only the opt-in selection policy in the audited-proposal note,
not its scientific boundaries or the lower-level exact-identity contract.
